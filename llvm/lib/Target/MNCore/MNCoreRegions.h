//===- MNCoreRegions.h - PE-operand access decoding -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decodes the PE-operand (GRF0/GRF1/LM0/LM1/T) accesses of one PE
// instruction expression, and the section 3.6.4 per-step conflict rules
// over them. Shared by the packetizer (which refuses to co-issue a
// conflicting pair) and available to the hazard pass.
//
// Only the addressable PE memories are tracked here. L1BM/L2BM accesses
// are a different address space with their own direction rules and are
// handled by MNCoreHazard.
//
// The operand layout per expression family (dst is last in the .vsm
// input-then-output order; see the .td files):
//
//   l1dir 1 (L1B->PE):  the LAST immediate is the PE destination (write)
//   l1dir 2 (PE->L1B):  the second-to-last immediate is the PE source (read)
//   mauGrp 1 (calc):    last peoperand writes; a leading matrix operand is
//                       skipped when the expression touches a face
//   mauGrp 2 (mwrite):  the first immediate is the PE source (read)
//   mauGrp 3 (mread):   the last immediate is the PE destination (write)
//   otherwise (ALU):    last peoperand writes, the rest read
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREREGIONS_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREREGIONS_H

#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
namespace mncore {

enum MemSpace { MS_GRF0 = 0, MS_GRF1, MS_LM0, MS_LM1, MS_T, MS_OMR };

// One access of an addressable PE memory.
struct PEAccess {
  int Space;     // MemSpace
  int64_t Addr;  // base address (the peoperand address field)
  bool Write;
};

// The parts of a peoperand that the section 3.6.4 vfma/vmul-vs-mwrite rule
// compares: which memory is read (space + address) and the three input
// modifiers — sign inversion '-', one-precision-up extend 'e', and
// one-precision-down reduce 'r'. Bit positions match printPEOperand.
struct PEInput {
  bool Valid = false; // false for a non-memory immediate (matrix, etc.)
  int Space = 0;
  int64_t Addr = 0;
  bool Neg = false, Ext = false, Red = false;
};

inline PEInput peInput(uint64_t X) {
  PEInput P;
  unsigned Kind = X & 0xf;
  if (Kind > 4)
    return P; // forwarding / matrix / dummy sink: not a memory operand
  P.Valid = true;
  P.Space = (int)Kind;
  P.Addr = ((X >> 9) & 1) ? (int64_t)((X >> 10) & 0xfff)
                          : (int64_t)((X >> 10) & 0xffff);
  P.Ext = (X >> 6) & 1;
  P.Neg = (X >> 7) & 1;
  P.Red = (X >> 8) & 1;
  return P;
}

// Nth immediate operand of MI (0-based among immediates).
inline uint64_t nthImm(const MachineInstr &MI, unsigned N) {
  unsigned I = 0;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isImm()) {
      if (I == N)
        return MO.getImm();
      ++I;
    }
  return ~0ULL; // no such imm: decodes as an invalid PEInput
}

// Decode MI's PE-operand accesses into Acc.
inline void peAccesses(const MachineInstr &MI,
                       SmallVectorImpl<PEAccess> &Acc) {
  uint64_t TS = MI.getDesc().TSFlags;
  unsigned L1 = (TS >> 8) & 0x3;
  unsigned Grp = (TS >> 1) & 0x3;
  unsigned Face = (TS >> 3) & 0x3;

  SmallVector<uint64_t, 6> V;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isImm())
      V.push_back(MO.getImm());
  // A masked def appends [.., $zmask, $wmask] after its PE operands
  // (TSFlags bits 14/15). Those are mask specifiers, not memory accesses.
  if ((TS >> 15) & 1) {
    if (!V.empty())
      V.pop_back(); // $wmask
    if ((TS >> 14) & 1 && !V.empty())
      V.pop_back(); // $zmask
  } else if ((TS >> 14) & 1 && !V.empty()) {
    V.pop_back(); // $zmask only
  }
  // An imm command's first ins operand is the literal, not a PE operand.
  if (((TS >> 16) & 1) && !V.empty())
    V.erase(V.begin());

  // Mask-register destinations ($omr, 3.6.1.13) carry marker bit 32.
  // They are always writes; keep them out of the positional peoperand
  // decode and record them as MS_OMR writes (so two expressions writing
  // the same entry in one step are still caught).
  {
    SmallVector<uint64_t, 6> F;
    for (uint64_t X : V) {
      if ((X >> 32) & 1)
        Acc.push_back({MS_OMR, (int64_t)(X & 0x3f), /*Wr=*/true});
      else
        F.push_back(X);
    }
    V = std::move(F);
  }

  auto push = [&](uint64_t X, bool Wr) {
    unsigned Kind = X & 0xf;
    int Sp;
    switch (Kind) {
    case 0: Sp = MS_GRF0; break;
    case 1: Sp = MS_GRF1; break;
    case 2: Sp = MS_LM0; break;
    case 3: Sp = MS_LM1; break;
    case 4: Sp = MS_T; break;
    default: return; // forwarding / fixed value / matrix / non-memory
    }
    int64_t Addr = ((X >> 9) & 1) ? (int64_t)((X >> 10) & 0xfff)
                                  : (int64_t)((X >> 10) & 0xffff);
    Acc.push_back({Sp, Addr, Wr});
  };

  // Multi-destination form (section 3.6.1: <dst_0> [<dst_1>..]). The
  // write count is in TSFlags[20:18]; the last ndst PE operands write
  // and everything before them (after a leading matrix operand) reads.
  if ((TS >> 17) & 1) {
    unsigned NDst = (TS >> 18) & 0x7;
    int N = (int)V.size();
    int SrcEnd = N - (int)NDst;
    if (SrcEnd < 0)
      SrcEnd = 0;
    int Lo = 0;
    if (L1 == 1)
      Lo = 1;                          // skip the L1B source operand
    else if (Grp == 1 && Face != 0)
      Lo = 1;                          // skip the leading matrix operand
    for (int I = Lo; I < SrcEnd; ++I)
      push(V[I], /*Wr=*/false);
    for (int I = SrcEnd; I < N; ++I)
      push(V[I], /*Wr=*/true);
    return;
  }

  if (L1 == 1) {                       // L1B->PE: last imm is the PE write
    if (!V.empty())
      push(V.back(), /*Wr=*/true);
    return;
  }
  if (L1 == 2) {                       // PE->L1B: second-to-last is the PE read
    if (V.size() >= 2)
      push(V[V.size() - 2], /*Wr=*/false);
    return;
  }

  if (Grp == 2) {                      // mwrite: first imm is the PE read
    if (!V.empty())
      push(V[0], /*Wr=*/false);
    return;
  }
  if (Grp == 3) {                      // mread: last imm is the PE write
    if (V.size() >= 2)
      push(V.back(), /*Wr=*/true);
    return;
  }

  // mau-calc (grp 1, skip a leading matrix operand when a face is touched)
  // and the ALU (grp 0): the last peoperand is the destination write, the
  // preceding peoperands are reads.
  int Lo = (Grp == 1 && Face != 0) ? 1 : 0;
  int Dst = (int)V.size() - 1;
  for (int I = Lo; I < (int)V.size(); ++I)
    push(V[I], /*Wr=*/I == Dst);
}

// True when the two expressions violate a section 3.6.4 operand rule and
// so must not share a step:
//   * two expressions write the same PE operand (same space + address);
//   * two expressions read the same addressable space at different
//     addresses (a read set must be uniform across the step);
//   * an LM space is both read and written but at different addresses
//     (a same-address read+write in one step is explicitly allowed);
//   * an `imm` is issued while LM0 is accessed.
//
// The imm-ness is read from TSFlags bit 16, so every imm/immu spelling
// (isel `IMM`, typed `IMM_P`, `IMMU_P`) is covered.
inline bool peConflict(const MachineInstr &A, const MachineInstr &B) {
  SmallVector<PEAccess, 6> AA, BA;
  peAccesses(A, AA);
  peAccesses(B, BA);

  bool ImmA = (A.getDesc().TSFlags >> 16) & 1;
  bool ImmB = (B.getDesc().TSFlags >> 16) & 1;

  for (const PEAccess &X : AA)
    for (const PEAccess &Y : BA) {
      if (X.Space != Y.Space)
        continue;
      // Two writes of the same operand.
      if (X.Write && Y.Write && X.Addr == Y.Addr)
        return true;
      // Reads of one space must all use the same address.
      if (!X.Write && !Y.Write && X.Addr != Y.Addr)
        return true;
      // LM read+write in one step must cover the same address.
      if ((X.Space == MS_LM0 || X.Space == MS_LM1) &&
          X.Write != Y.Write && X.Addr != Y.Addr)
        return true;
      // `imm` forbids any LM0 access in the same step.
      if ((ImmA || ImmB) && X.Space == MS_LM0)
        return true;
    }
  return false;
}

// \xC2\xA73.6.4 vfma/vmul + mwrite sub-rule. When a `vfma`/`vmul` (vector
// forms, TSFlags bit 13) and an `mwrite` co-issue, the vector expression's
// *second* input and the mwrite's input must read the same PE operand and
// agree on sign inversion (-\xC2\xA73.6.9.22), precision extend ('e',
// \xC2\xA73.6.9.23) and precision reduce ('r', \xC2\xA73.6.9.24). Returns true
// (a conflict) when exactly one side is vfma/vmul + the other is mwrite
// and the shapes disagree.
inline bool vfmaMwriteShapeConflict(const MachineInstr &A,
                                    const MachineInstr &B) {
  auto IsVfma = [](const MachineInstr &MI) {
    return (MI.getDesc().TSFlags >> 13) & 1;
  };
  auto IsMwrite = [](const MachineInstr &MI) {
    return ((MI.getDesc().TSFlags >> 1) & 0x3) == 2; // mauGrp == mwrite
  };

  const MachineInstr *V = nullptr, *W = nullptr;
  if (IsVfma(A) && IsMwrite(B)) { V = &A; W = &B; }
  else if (IsVfma(B) && IsMwrite(A)) { V = &B; W = &A; }
  else return false;

  // vfma and vmul both carry the second input in immediate 1 (sx, sy, ...);
  // mwrite carries its single input in immediate 0 (src).
  PEInput VI = peInput(nthImm(*V, 1));
  PEInput WI = peInput(nthImm(*W, 0));
  if (!VI.Valid || !WI.Valid)
    return false; // a non-memory input has no shape to compare
  if (VI.Space != WI.Space || VI.Addr != WI.Addr)
    return true;
  return VI.Neg != WI.Neg || VI.Ext != WI.Ext || VI.Red != WI.Red;
}

// Mask specifiers carried by a masked def (section 3.6.2). The operands
// pack: [0] present, [1] ll (2-longword word length), [2] var ($imr),
// [15:3] value, [17:16] tp, [18] zero (zero-flush vs write mask).
struct PEMask {
  bool Present = false;
  bool LL = false;   // 2-longword mask word length
  bool Zero = false; // zero-flush (3.6.2.2) rather than a write mask
  bool Var = false;  // variable entry ($imr<adr>) rather than a pattern
  unsigned Value = 0; // fixed 4-bit pattern, or the variable entry address
};

// Collect the present mask specifiers of MI (0, 1 or 2 of them).
inline void peMasks(const MachineInstr &MI, SmallVectorImpl<PEMask> &Out) {
  uint64_t TS = MI.getDesc().TSFlags;
  bool HasZ = (TS >> 14) & 1;
  bool HasW = (TS >> 15) & 1;
  if (!HasZ && !HasW)
    return;
  SmallVector<uint64_t, 6> V;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isImm())
      V.push_back(MO.getImm());
  // Layout: [.., $zmask, $wmask]; take from the tail.
  uint64_t W = HasW ? V.back() : 0;
  if (HasW)
    V.pop_back();
  uint64_t Z = HasZ ? V.back() : 0;
  for (uint64_t X : {Z, W}) {
    PEMask M;
    M.Present = X & 1;
    M.LL = (X >> 1) & 1;
    M.Zero = (X >> 18) & 1;
    M.Var = (X >> 2) & 1;
    M.Value = (unsigned)((X >> 3) & 0xf);
    if (M.Present)
      Out.push_back(M);
  }
}

// Per-instruction grammar checks that do not fit the pairwise step rules:
//   * section 3.6.1.20 fixed-value inputs ($l2bid/$l1bid/$mabid/$peid/
//     $subpeid/$msb1) and 3.6.1.18 $mreadf are legal only as the ALU's
//     first input operand;
//   * 3.6.1.19 $nowrite must be the *only* output operand, so it cannot
//     appear in a multi-destination form.
inline bool peInstructionConflict(const MachineInstr &MI) {
  uint64_t TS = MI.getDesc().TSFlags;
  bool Imm = (TS >> 16) & 1;
  unsigned Grp = (TS >> 1) & 0x3;
  unsigned L1 = (TS >> 8) & 0x3;
  bool MDst = (TS >> 17) & 1;
  unsigned NDst = (TS >> 18) & 0x7;

  SmallVector<uint64_t, 8> V;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isImm())
      V.push_back(MO.getImm());
  if ((TS >> 15) & 1) {
    if (!V.empty())
      V.pop_back();
    if ((TS >> 14) & 1 && !V.empty())
      V.pop_back();
  } else if ((TS >> 14) & 1 && !V.empty()) {
    V.pop_back();
  }
  if (Imm && !V.empty())
    V.erase(V.begin());

  if (MDst) {
    if (V.size() < NDst)
      return true;
    unsigned Nowrite = 0;
    for (unsigned I = (unsigned)V.size() - NDst; I < V.size(); ++I)
      if ((V[I] & 0xf) == 9)
        ++Nowrite;
    if (Nowrite && NDst > 1)
      return true;
  }

  // ALU expressions only (grp 0, not an L1BM/L2BM transfer).
  if (Grp == 0 && L1 == 0) {
    for (unsigned I = 1; I < V.size(); ++I) {
      unsigned Kind = V[I] & 0xf;
      if (Kind == 8 || (Kind >= 10 && Kind <= 15))
        return true; // fixed value / mreadf outside the first input
    }
  }
  return false;
}

// \xC2\xA73.6.4 step-wide mask rules over the whole set of co-issued
// expressions. Returns true when the step is illegal:
//   * a zero-flush mask (3.6.2.2) appears on more than one expression;
//   * two present mask specifiers disagree on the mask word length
//     (longword vs 2-longword, 3.6.2);
//   * two present mask specifiers read different mask-register entries
//     (3.6.2.1: the entry address must be uniform across a step; a fixed
//     pattern names a fixed entry, a `$imr<adr>` a variable one, so both
//     the pattern and the variable address take part in the comparison).
// Unlike the operand rules this cannot be decided pairwise, so it takes the
// full set of expressions that would share the step.
template <typename RangeT>
inline bool maskStepConflict(const RangeT &MIs) {
  unsigned ZeroCount = 0;
  int WordLen = -1; // -1 unset, else 0 longword / 1 2-longword
  int EntryVar = -1, EntryVal = -1;
  bool Bad = false;
  for (const MachineInstr *MI : MIs) {
    SmallVector<PEMask, 2> Ms;
    peMasks(*MI, Ms);
    for (const PEMask &M : Ms) {
      if (M.Zero && ++ZeroCount > 1)
        Bad = true;
      int WL = M.LL ? 1 : 0;
      if (WordLen < 0)
        WordLen = WL;
      else if (WordLen != WL)
        Bad = true;
      int Var = M.Var ? 1 : 0;
      if (EntryVar < 0) {
        EntryVar = Var;
        EntryVal = (int)M.Value;
      } else if (EntryVar != Var || EntryVal != (int)M.Value) {
        Bad = true;
      }
    }
  }
  return Bad;
}

} // namespace mncore
} // namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREREGIONS_H
