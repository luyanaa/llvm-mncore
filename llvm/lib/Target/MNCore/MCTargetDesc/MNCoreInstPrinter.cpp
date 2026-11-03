//===- MNCoreInstPrinter.cpp - MN-Core2 instruction printer ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreInstPrinter.h"
#include "MNCoreMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "MNCoreGenAsmWriter.inc"

void MNCoreInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}

// M1 scalar registers are GRF0/GRF1/T slots. `.vsm` names them as physical
// PE operands, not compiler registers: GRF0@A -> $lr<A>v, GRF1@A ->
// $ls<A>v, the T register -> $llt. All M1 scalar ALU ops are longword
// accesses, so the `l` width and the `v` address mode are fixed here; the
// assemble3-side 2-longword alignment rule is checked when the instruction
// is emitted (MNCoreAsmPrinter::emitInstruction). MASM-level packed
// operands do not go through this path — they are i64 immediates printed by
// printPEOperand.
void MNCoreInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  StringRef N = getRegisterName(Reg);
  auto NumAfter = [&](StringRef Prefix, int64_t &Out) -> bool {
    if (!N.starts_with(Prefix))
      return false;
    return !N.drop_front(Prefix.size()).getAsInteger(10, Out);
  };
  int64_t Addr = 0;
  if (NumAfter("grf0_", Addr)) {
    OS << "$lr" << Addr << 'v';
    return;
  }
  if (NumAfter("grf1_", Addr)) {
    OS << "$ls" << Addr << 'v';
    return;
  }
  if (NumAfter("t_", Addr)) {
    OS << "$llt"; // T has no address; always 2-longword (3.6.1.12)
    return;
  }
  if (NumAfter("omr_", Addr)) {
    OS << "$omr" << Addr;
    return;
  }
  OS << '$' << N;
}

void MNCoreInstPrinter::printOperand(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return;
  }
  if (MO.isImm()) {
    O << MO.getImm();
    return;
  }
  if (MO.isExpr()) {
    MAI.printExpr(O, *MO.getExpr());
    return;
  }
  llvm_unreachable("unsupported operand kind");
}

void MNCoreInstPrinter::printTagOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  int64_t Tag = MI->getOperand(OpNum).getImm();
  // .vsm tag literal: 'i' followed by a two-digit hex number.
  O << format("i%02llx", Tag);
}

void MNCoreInstPrinter::printImmLiteral(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  int64_t Imm = MI->getOperand(OpNum).getImm();
  // .vsm immediate literal: a typed single-word value, `i"<s32>"`.
  // The selection pattern only matches constants that fit a signed
  // 32-bit word, so a plain signed-decimal rendering is always legal.
  O << "i\"" << (int32_t)Imm << "\"";
}

// Typed immediate literal (3.2.2). Operand packing:
//   [3:0]  type    : 0=i 1=s 2=ui 3=us 4=f 5=h
//   [35:4] payload : the 32-bit word value (half types: the 16-bit value
//                    replicated twice; float types: the raw bit pattern).
// `u` (xoxo) is an opcode variant (immu), so it is not encoded here.
void MNCoreInstPrinter::printImmTyped(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  unsigned Type = V & 0xf;
  uint32_t Word = (uint32_t)((V >> 4) & 0xffffffffULL);
  switch (Type) {
  case 1: // signed half integer: value replicated into both halves
    O << "s\"" << (int16_t)(Word & 0xffff) << '"';
    return;
  case 2: // unsigned single integer
    O << "ui\"0x" << format_hex_no_prefix(Word, 8, /*Upper=*/false) << '"';
    return;
  case 3: // unsigned half integer
    O << "us\"0x" << format_hex_no_prefix(Word & 0xffff, 4, false) << '"';
    return;
  case 4: { // single-precision float, printed with round-trip precision
    float F;
    std::memcpy(&F, &Word, sizeof(F));
    O << "f\"" << format("%.9g", (double)F) << '"';
    return;
  }
  case 5: { // half-precision float
    APFloat H(APFloat::IEEEhalf(), APInt(16, Word & 0xffff));
    bool LosesInfo = false;
    H.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &LosesInfo);
    O << "h\"" << format("%.9g", H.convertToDouble()) << '"';
    return;
  }
  default: // i
    O << "i\"" << (int32_t)Word << '"';
    return;
  }
}

void MNCoreInstPrinter::printBFMantOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  // hbfn/<n>, hbfe/<n>: mantissa length, an integer 6..9 (3.6.12.12).
  O << '/' << MI->getOperand(OpNum).getImm();
}

// --- MV instruction operand helpers (section 3.5) ---------------------
//
// The MV memory operand packs its memory kind, address and the optional
// @group / .l2b qualifiers into one i64 (see MNCoreInstrMV.td for the
// field layout). printMVOperand unpacks it back to the .vsm text.
namespace {
enum MVKind : uint64_t { MVK_PDM = 0, MVK_DRAM = 1, MVK_L2BM = 2, MVK_DRAMI = 3 };
}

void MNCoreInstPrinter::printMVOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  uint64_t Kind = V & 0x3;
  uint64_t Addr = (V >> 4) & 0xffffffffULL;
  bool HasGrp = (V >> 40) & 1;
  unsigned Grp = (V >> 41) & 0x3;
  bool HasL2b = (V >> 44) & 1;
  unsigned L2b = (V >> 45) & 1;

  switch (Kind) {
  case MVK_PDM:   O << "$p";  break;
  case MVK_DRAM:  O << "$d";  break;
  case MVK_L2BM:  O << "$lc"; break;
  case MVK_DRAMI: O << "$di"; break;
  }
  O << Addr;
  // The '@' qualifier block. L2BM uses "@<group>.<l2b>", "@.<l2b>" or
  // nothing; PDM/DRAM use "@<group>" or nothing.
  if (HasGrp || HasL2b) {
    O << '@';
    if (HasGrp)
      O << Grp;
    if (HasL2b)
      O << '.' << L2b;
  }
}

// The "/n<size>[i<tag>][nd<N>][p<prio>]" parameter block. -1 in an
// optional field means the parameter is omitted.
void MNCoreInstPrinter::printMVParams(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  uint64_t Size = V & 0xffffffffULL;
  int64_t Tag = (int64_t)((V >> 32) & 0xff);
  int64_t Nd  = (int64_t)((V >> 40) & 0xff);
  int64_t Prio = (int64_t)((V >> 48) & 0xff);

  O << "/n" << Size;
  if (Tag != 0xff)
    O << format("i%02llx", Tag);
  if (Nd != 0xff)
    O << "nd" << Nd;
  if (Prio != 0xff)
    O << "p" << Prio;
}

// --- L2BM / L1BM expression operand helpers (sections 3.6.7/3.6.8) ----
//
// Each of these decodes a packed i64 whose field layout is documented
// next to the operand type in MNCoreInstrL2BM.td / MNCoreInstrL1BM.td.

// "$lc<addr>" or "$lc<addr>@.<l2b>" (l2bmdars form).
void MNCoreInstPrinter::printLCOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  O << "$lc" << ((V >> 4) & 0xffffffffULL);
  if ((V >> 40) & 1)
    O << "@." << ((V >> 41) & 1);
}

// "$lb<addr>" — the plain L1BM operand of L2BM expressions.
void MNCoreInstPrinter::printLBOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  O << "$lb" << ((V >> 4) & 0xffffffffULL);
}

// "$dar<addr>".
void MNCoreInstPrinter::printDAROperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  O << "$dar" << ((V >> 4) & 0xffffffffULL);
}

// The "@<l1bset>" opcode qualifier of L2BM expressions. An absent
// qualifier prints nothing; otherwise "@<l1badr>/<immode>" (mask form)
// or "@[<list>]" when the list form was written.
void MNCoreInstPrinter::printL1BSetOperand(const MCInst *MI,
                                           unsigned OpNum,
                                           raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  if (!((V >> 7) & 1))
    return;
  O << '@';
  if ((V >> 6) & 1) {
    uint64_t Mask = (V >> 16) & 0xff;
    O << '[';
    bool First = true;
    for (unsigned I = 0; I < 8; ++I) {
      if (!((Mask >> I) & 1))
        continue;
      if (!First)
        O << ',';
      O << I;
      First = false;
    }
    O << ']';
    return;
  }
  unsigned Adr = V & 0xf;
  unsigned Immode = (V >> 8) & 0x7;
  O << Adr;
  if (Immode != 0)
    O << '/' << Immode;
}

// The packed PE-side operand (section 3.6.1 names), both stream modes.
void MNCoreInstPrinter::printPEOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  unsigned Kind = V & 0xf;
  unsigned Width = (V >> 4) & 0x3;
  bool EFlag = (V >> 6) & 1;
  bool Neg = (V >> 7) & 1;
  bool Rd = (V >> 8) & 1;
  bool Flat = (V >> 9) & 1;

  // LM base-address-register write operands (3.6.1.7 / 3.6.1.9):
  // $[l]mb / $[l]nb. Marked by peoperand bit 49; output-only.
  if (((V >> 49) & 1) && !((V >> 9) & 1) && (Kind == 2 || Kind == 3)) {
    O << '$' << (Width == 0 ? "" : "l") << (Kind == 2 ? "mb" : "nb");
    goto done;
  }
  if (Neg)
    O << '-';
  switch (Kind) {
  case 5:  O << "$mauf";    goto done;
  case 6:  O << "$aluf";    goto done;
  case 7:  O << "$lbf";     goto done;
  case 8:  O << "$mreadf";  goto done;
  case 9:  O << "$nowrite"; goto done; // dummy output sink
  case 4: // T register: the (l|ll) prefix is accepted but ignored (3.6.1.12)
    O << (Width == 2 ? "$llt" : Width == 1 ? "$lt" : "$t");
    goto done;
  // Section 3.6.1.20 fixed-value inputs (ALU first operand only).
  case 10: O << "$l2bid";   goto done;
  case 11: O << "$l1bid";   goto done;
  case 12: O << "$mabid";   goto done;
  case 13: O << "$peid";    goto done;
  case 14: O << "$subpeid"; goto done;
  case 15: O << "$msb1";    goto done;
  default: break;
  }
  // Memory names $[(l|ll)](r|s|m|n)... — a pure width prefix:
  // word $r/$s/$m/$n, longword $l*, double-longword $ll*.
  O << '$';
  if (Width == 1)
    O << 'l';
  else if (Width == 2)
    O << "ll";

  if (Flat) {
    bool TInd = (V >> 62) & 1;
    bool HasJ = (V >> 58) & 1;
    unsigned Madpe = (V >> 59) & 0x7;
    O << "rsmn"[Kind];
    if (TInd)
      O << 't';
    O << '[';
    for (unsigned I = 0; I < 4; ++I) {
      if (I)
        O << ',';
      O << ((V >> (10 + I * 12)) & 0xfff);
    }
    O << ']';
    if (HasJ)
      O << 'j' << Madpe;
  } else {
    uint64_t Addr = (V >> 10) & 0xffffULL;
    bool HasV = (V >> 26) & 1;
    bool HasAdri = (V >> 27) & 1;
    uint64_t Adri = (V >> 28) & 0xffffULL;
    bool HasJ = (V >> 44) & 1;
    unsigned Madpe = (V >> 45) & 0x7;
    bool TInd = (V >> 48) & 1;
    O << "rsmn"[Kind];
    if (TInd)
      O << 't';
    // T-indirect may omit the inner address entirely ($lmt); when an
    // explicit v<adri> is present the addr prints ($lmt8v4).
    if (!TInd || Addr != 0 || HasV || HasAdri)
      O << Addr;
    if (HasV) {
      O << 'v';
      if (HasAdri)
        O << Adri;
    }
    if (HasJ)
      O << 'j' << Madpe;
  }
done:
  if (EFlag)
    O << 'e';
  if (Rd)
    O << 'r';
}

// "$l(x|y)<addr>" (bit 9 selects the $llx/$ll 2-lw spelling).
void MNCoreInstPrinter::printMATOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  O << '$' << (((V >> 9) & 1) ? "ll" : "l") << ((V & 1) ? 'y' : 'x')
    << ((V >> 1) & 0xff);
}

// "$lx" / "$ly" — whole-face matvec operand.
void MNCoreInstPrinter::printMATSideOperand(const MCInst *MI,
                                            unsigned OpNum,
                                            raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  O << "$l" << ((V & 1) ? 'y' : 'x');
}

// Section 3.6.2 mask application, the shared "/(...)":
//   "/(ll)?(<4-bit pattern>|$imr<adr>)" followed by an optional 't'/'p'
// word-length hint. The operand packs:
//   [0]     present
//   [1]     ll     : 2-longword mask word length
//   [2]     var    : variable entry ($imr) instead of a fixed pattern
//   [15:3]  value  : fixed-entry 4-bit pattern, or variable-entry address
//   [17:16] tp     : 0 none, 1 't' (truncate), 2 'p' (pad)
void MNCoreInstPrinter::printMaskOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  if (!(V & 1))
    return;
  bool LL = (V >> 1) & 1;
  bool Var = (V >> 2) & 1;
  unsigned Value = (V >> 3) & 0xf;
  bool Zero = (V >> 18) & 1;
  O << '/';
  if (LL)
    O << "ll";
  if (Var)
    O << "$imr" << Value;
  else
    for (int I = 3; I >= 0; --I) // 4-bit pattern, MSB = cycle 1
      O << ((Value >> I) & 1);
  // Word-length hint (3.6.2.1): 't' when the mask is 2-lw but the masked
  // PE destination is not, 'p' when the destination is 2-lw but the mask
  // is not. The suffix changes no behavior, but assemble3 rejects a
  // missing one and also rejects an unnecessary one, so derive it from
  // the destination, which sits immediately before the mask operands
  // (TSFlags 14/15 say how many mask operands this def carries).
  const MCInstrDesc &D = MII.get(MI->getOpcode());
  unsigned NMasks = (((D.TSFlags >> 14) & 1) + ((D.TSFlags >> 15) & 1));
  unsigned DstIdx = MI->getNumOperands() - NMasks - 1;
  unsigned TP = 0;
  if (NMasks && DstIdx < MI->getNumOperands()) {
    const MCOperand &Dst = MI->getOperand(DstIdx);
    if (Dst.isImm()) {
      uint64_t P = Dst.getImm();
      if ((P & 0xf) <= 4) { // a PE-memory operand
        unsigned DstW = (P >> 4) & 0x3; // 1=l, 2=ll
        if (LL && DstW != 2)
          TP = 1;
        else if (!LL && DstW == 2)
          TP = 2;
      }
    }
  }
  (void)Zero;
  if (TP == 1)
    O << 't';
  else if (TP == 2)
    O << 'p';
}

// `d get <memory> <nwords>` operand (3.4.3). Packed:
//   [3:0] mem, [20:4] addr, [28:21] nwords.
void MNCoreInstPrinter::printDGetOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  unsigned Mem = V & 0xf;
  uint64_t Addr = (V >> 4) & 0x1ffff;
  unsigned N = (unsigned)((V >> 21) & 0xff);
  switch (Mem) {
  case 1: O << "$lr" << Addr; break;
  case 2: O << "$ls" << Addr; break;
  case 3: O << "$lm" << Addr; break;
  case 4: O << "$ln" << Addr; break;
  case 5: O << "$t"; break;
  case 6: O << "$lb" << Addr; break;
  case 7: O << "$lc" << Addr; break;
  case 8: O << "$p" << Addr; break;
  case 9: O << "$d" << Addr; break;
  case 10: O << "$omr" << Addr; break;
  case 11: O << "$lx" << Addr; break;
  default: O << "$lr" << Addr; break;
  }
  O << ' ' << (N ? N : 1);
}

void MNCoreInstPrinter::printDSetData(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  O << format_hex_no_prefix((uint64_t)MI->getOperand(OpNum).getImm(), 16,
                            /*Upper=*/true);
}

// Multi-line write-mask directive (3.6.2.1):
//   mask[l|ll][r][s][t][m][n][k] <addr>
// Packed: [0] ll, [6:1] enables in r,s,t,m,n,k order, [13:7] address.
void MNCoreInstPrinter::printMaskDirective(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  if (V & 1)
    O << "ll";
  static const char L[] = {'r', 's', 't', 'm', 'n', 'k'};
  for (unsigned I = 0; I < 6; ++I)
    if ((V >> (1 + I)) & 1)
      O << L[I];
  O << ' ' << ((V >> 7) & 0x7f);
}

// "$omr<addr>" — mask-register write destination (3.6.1.13). Packed as
// (addr | marker<<32) so the PE-operand decoders can tell it apart from a
// peoperand immediate.
void MNCoreInstPrinter::printOMROperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  O << "$omr" << (MI->getOperand(OpNum).getImm() & 0x3f);
}

// "$(l|ll)b(<addr>|i)" — the L1BM operand of L1BM expressions.
void MNCoreInstPrinter::printLB2Operand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  bool Turn = V & 1;
  bool LL = (V >> 1) & 1;
  O << '$' << (LL ? "llb" : "lb");
  if (Turn)
    O << 'i';
  else
    O << ((V >> 2) & 0xffffULL);
}

// "@<mabadr>" opcode qualifier; absent prints nothing.
void MNCoreInstPrinter::printMABSetOperand(const MCInst *MI,
                                           unsigned OpNum,
                                           raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  if ((V >> 4) & 1)
    O << '@' << (V & 0xf);
}

// "[<mabdiff>]" inside the l1bmd mnemonic; absent prints nothing.
void MNCoreInstPrinter::printMABDiffOperand(const MCInst *MI,
                                            unsigned OpNum,
                                            raw_ostream &O) {
  uint64_t V = MI->getOperand(OpNum).getImm();
  if (!((V >> 5) & 1))
    return;
  O << (((V >> 4) & 1) ? '-' : '+') << (V & 0xf);
}
