//===- MNCoreInstPrinter.cpp - MN-Core2 instruction printer ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreInstPrinter.h"
#include "MNCoreMCTargetDesc.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "MNCoreGenAsmWriter.inc"

void MNCoreInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}

void MNCoreInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  OS << '$' << getRegisterName(Reg);
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

  if (Neg)
    O << '-';
  switch (Kind) {
  case 5:  O << "$mauf";    goto done;
  case 6:  O << "$aluf";    goto done;
  case 7:  O << "$lbf";     goto done;
  case 8:  O << "$mreadf";  goto done;
  case 9:  O << "$nowrite"; goto done; // dummy output sink
  case 4:  O << "$llt";     goto done; // T register: always 2-lw
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
