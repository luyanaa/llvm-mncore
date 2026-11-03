//===- MNCoreHazard.cpp - MN-Core2 hazard spacing ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Inserts the `nop` steps section 3.6.3 requires between PE instruction
// expressions. The packetizer only decides which expressions may *share*
// a step; it never widens the gap between two steps, so it cannot by
// itself produce the spacing a data race or port collision demands. This
// pass runs AFTER the packetizer, walks the final bundled-step layout
// (a BUNDLE group is one step; a solo MachineInstr is one step), and
// emits NOP MachineInstrs — each an exclusive whole step — wherever a
// following step would otherwise issue too soon after a conflicting
// earlier one.
//
// NOP is exclusive (it shares a step only with wait), so N consecutive
// NOPs insert exactly N whole steps of separation — the `nop`/`nop/N`
// idiom the manual uses. The pass computes, for each step, the maximum
// gap required against every earlier conflicting access, then emits
// that many NOPs before it.
//
// Minimum gaps, in *empty steps between* writer and reader (the manual's
// "Nステップ空ける" — write at step s, earliest read at s+N+1), from
// section 3.6.3 ("always safe" bound; matches the assembler's checker):
//
//   GRF0/GRF1 write -> read     1   (data race, same addr)
//   LM0/LM1   write -> read     2   (port collision, any addr)
//   T         write -> read     1   (data race, same addr)
//   PE->L1B   ->  L1B->PE       2   (port collision, any addr; the
//                                  turnaround drain is exempt)
//   PE->L1B   ->  L1B->L2B/mc   3   (data race, same L1B addr)
//   L2B->L1B  ->  L1B->PE       2   (data race, same L1B addr)
//   L2B->L1B  ->  L1B->L2B/mc   2   (port, assume L1B overlap)
//   int-mcast ->  L1B->PE       3   (data race, same L1B addr)
//   int-mcast ->  L1B->L2B/mc   3   (port, assume L1B overlap)
//   L1B->L2B  ->  L2B->L1B      3   (port collision, any addr; this is a
//                                  read->write ordering)
//   L1B->L2B  ->  MV reads L2B  1   (data race; MV handled by tag)
//
// Multi-MV constraints (section 3.5.9) are not modelled: the manual
// defers their exact form to another document / the assembler's checker.
// A lone `wait` step (section 3.6.13 requires a co-issued expression) is
// not rejected here either — packetization cannot force a bundle.
//
// Not yet modelled (see mncore-backend-status): MV hazards (MVs carry
// tag/wait sync, not step spacing); write-mask (/imr) narrowing of the
// write cycle; L1B-subset (@<l1bset>) narrowing of the port-collision
// "when L1B overlaps" condition (we conservatively assume overlap);
// sub-word / MAB-granular address overlap.
//
//===----------------------------------------------------------------------===//

#include "MNCore.h"
#include "MNCoreInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mncore-hazard"

static cl::opt<bool> HazardReport(
    "mncore-hazard-report", cl::Hidden,
    cl::desc("Report each hazard gap and the rule that forced it"));

namespace {

enum MemSpace { MS_GRF0 = 0, MS_GRF1, MS_LM0, MS_LM1, MS_T, MS_L1B, MS_L2B };

// A memory region an expression accesses: space, base address (the
// packed immediate's address field), and whether the access writes.
struct Region {
  int Space;
  int64_t Addr;
  bool Write;
  // L1B-subset membership (3.6.7.1), an 8-bit L1B mask. 0xff = all L1Bs
  // (any expression without an @<l1bset> qualifier).
  unsigned Set = 0xff;
};

// One tracked access: the region plus the direction context of the
// expression that produced it (needed for the L1B direction-pair rules).
struct Access {
  Region R;
  unsigned D1;   // l1dir: 1=L1B->PE, 2=PE->L1B
  unsigned D2;   // l2dir: 1=L1B->L2B, 2=L2B->L1B, 3=L1B-internal/DAR
  bool Turn;     // turnaround ($lbi) form
};

// peoperand kind (low nibble) -> memory space, -1 for non-memory kinds.
static int peSpace(unsigned kind) {
  switch (kind) {
  case 0: return MS_GRF0;
  case 1: return MS_GRF1;
  case 2: return MS_LM0;
  case 3: return MS_LM1;
  case 4: return MS_T;
  default: return -1;
  }
}

// Decode a peoperand immediate -> (space, addr). Flat uses a0.
static bool decPE(uint64_t V, Region &R, bool write) {
  int sp = peSpace(V & 0xf);
  if (sp < 0)
    return false;
  R.Space = sp;
  R.Addr = ((V >> 9) & 1) ? (int64_t)((V >> 10) & 0xfff)
                        : (int64_t)((V >> 10) & 0xffff);
  R.Write = write;
  return true;
}
// lb2operand: [0]turn [1]ll [17:2]addr. Turnaround ($lbi) -> Addr -1.
static Region lb2(uint64_t V, bool write) {
  return {MS_L1B, (V & 1) ? -1 : (int64_t)((V >> 2) & 0xffff), write};
}
// lboperand/lcoperand (L2BM): [35:4]addr.
static Region lb(uint64_t V, bool write) {
  return {MS_L1B, (int64_t)((V >> 4) & 0xffffffffULL), write};
}
static Region lc(uint64_t V, bool write) {
  return {MS_L2B, (int64_t)((V >> 4) & 0xffffffffULL), write};
}

// Decode the L1B-subset qualifier (3.6.7.1) into an 8-bit L1B mask.
// Packing: [7] present, [6] isList, [3:0] l1badr, [10:8] immode,
// [31:16] list mask. Absent -> all L1Bs.
static unsigned l1bSetMask(uint64_t X) {
  if (!((X >> 7) & 1))
    return 0xff;
  if ((X >> 6) & 1)
    return (unsigned)((X >> 16) & 0xff);
  unsigned Adr = (unsigned)(X & 0xf) & 0x7;
  unsigned Immode = (unsigned)((X >> 8) & 0x7);
  unsigned NotI = (~Immode) & 0x7;
  unsigned M = 0;
  for (unsigned B = 0; B < 8; ++B)
    if ((B & NotI) == (Adr & NotI))
      M |= 1u << B;
  return M;
}

// TSFlags helpers (layout in MNCoreInstrFormats.td / the .td files).
static bool isPEExpr(const MachineInstr &MI) {
  return MI.getDesc().TSFlags & 0x1;
}
static unsigned l1dir(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 8) & 0x3;
}
static unsigned l2dir(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 10) & 0x3;
}
static unsigned mauGrp(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 1) & 0x3;
}
static bool isTurn(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 12) & 1;
}
static bool isMV(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 21) & 1;
}

// Collect the immediates of MI in order (skipping non-imm operands).
static void imms(const MachineInstr &MI, SmallVectorImpl<uint64_t> &Out) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isImm())
      Out.push_back(MO.getImm());
}

class MNCoreHazard : public MachineFunctionPass {
public:
  static char ID;
  MNCoreHazard() : MachineFunctionPass(ID) {
    initializeMNCoreHazardPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "MN-Core2 Hazard Spacing"; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const MNCoreInstrInfo *TII = nullptr;
  void regions(const MachineInstr &MI, SmallVectorImpl<Region> &Out);
};

} // end anonymous namespace

// ---------------------------------------------------------------------------
// Region extraction. Operand order is fixed per instruction family in
// the .td files; we use the TSFlags direction + mau group to know which
// immediate is which memory side and whether each is a read or write.
//
// ALU / MAU (no l1dir/l2dir): the FIRST peoperand is the destination
// (write), the rest are sources (read). For mwrite the peoperand is the
// written-to-matrix source; for mread it is the destination. The matrix
// side ($lx/$ly) is not a hazard-tracked space here.
// ---------------------------------------------------------------------------
void MNCoreHazard::regions(const MachineInstr &MI,
                           SmallVectorImpl<Region> &Out) {
  SmallVector<uint64_t, 6> V;
  imms(MI, V);
  unsigned d1 = l1dir(MI), d2 = l2dir(MI);
  uint64_t TS = MI.getDesc().TSFlags;

  // Trailing mask specifiers are not PE accesses (TSFlags 14/15).
  if ((TS >> 15) & 1) {
    if (!V.empty())
      V.pop_back();
    if ((TS >> 14) & 1 && !V.empty())
      V.pop_back();
  } else if ((TS >> 14) & 1 && !V.empty()) {
    V.pop_back();
  }
  // The imm command's first operand is its literal (TSFlags 16).
  if (((TS >> 16) & 1) && !V.empty())
    V.erase(V.begin());

  // A mask-register destination ($omr, marker bit 32) is not a tracked
  // memory space here (no read operands exist for it).
  {
    SmallVector<uint64_t, 6> F;
    for (uint64_t X : V)
      if (!((X >> 32) & 1))
        F.push_back(X);
    V = std::move(F);
  }

  // L1B-subset qualifier of an L2BM expression. L2BM->L1B forms carry
  // @<l1bset> (3.6.7.1); the L1B->L2B point form names a single L1B via
  // @<l1badr> (mabsetoperand, present bit 4), which is a one-element set.
  unsigned SetMask = 0xff;
  if (d2 == 1) {
    if (!V.empty() && ((V[0] >> 4) & 1))
      SetMask = 1u << ((unsigned)V[0] & 0x7);
  } else if (d2) {
    if (!V.empty())
      SetMask = l1bSetMask(V[0]);
  }

  // Multi-destination forms (TSFlags 17): the last NDst PE operands write.
  bool MDst = (TS >> 17) & 1;
  unsigned NDst = (TS >> 18) & 0x7;

  if (d1) {
    // L1BM. l1dir 1 = L1B->PE (broadcast/distribute): lb2 rd, pe wr.
    //       l1dir 2 = PE->L1B (collect/individual/reduce): pe rd, lb2 wr.
    if (d1 == 1) {
      // imms: [ (diff) lb2 pe ] or [ lb2 pe ]
      int lbidx = V.size() - 2;             // second-to-last is lb2
      int peidx = V.size() - 1;             // last is pe
      if (MDst) {
        lbidx = 0;                          // lb2 leads the operand list
        peidx = -1;
      }
      if (lbidx >= 0) Out.push_back(lb2(V[lbidx], /*w=*/false));
      if (MDst) {
        for (unsigned I = (unsigned)lbidx + 1; I < V.size(); ++I) {
          Region R;
          if (decPE(V[I], R, /*w=*/true)) Out.push_back(R);
        }
      } else if (peidx >= 0) {
        Region R;
        if (decPE(V[peidx], R, /*w=*/true)) Out.push_back(R);
      }
    } else {
      // PE->L1B: operand tail is (pe rd, lb2 wr). The lb2 dst is last;
      // the pe source is the second-to-last immediate. For reduce forms
      // there's no @/diff qualifier — still (pe, lb2) at the tail.
      if (V.size() >= 2) {
        Region R;
        if (decPE(V[V.size()-2], R, /*w=*/false)) Out.push_back(R);
        Out.push_back(lb2(V[V.size()-1], /*w=*/true));
      } else if (V.size() == 1) {
        Out.push_back(lb2(V[0], /*w=*/true));
      }
    }
    return;
  }

  if (d2) {
    // L2BM. Operand tails end in the address operand(s). l2dir:
    //   1 = L1B->L2B   (lb rd -> lc wr)   l2bm@ / l2bmd / l2bmr[2]
    //   2 = L2B->L1B   (lc rd -> lb wr)   l2bmb / l2bmb2 / l2bmd
    //   3 = L1B-int/DAR (lb<->lb or lc->dar)
    if (d2 == 1) {
      // last two: lb rd, lc wr (l1bset/@ qualifier may lead; take tail)
      if (V.size() >= 2) {
        Region R = lb(V[V.size()-2], /*w=*/false);
        R.Set = SetMask;
        Out.push_back(R);
        Out.push_back(lc(V[V.size()-1], /*w=*/true));
      } else if (V.size() == 1) {
        Out.push_back(lc(V[0], /*w=*/true));
      }
    } else if (d2 == 2) {
      if (V.size() >= 2) {
        Out.push_back(lc(V[V.size()-2], /*w=*/false));
        Region R = lb(V[V.size()-1], /*w=*/true);
        R.Set = SetMask;
        Out.push_back(R);
      }
    } else { // d2 == 3: multicast lb->lb or dar stage lc->dar
      if (V.size() >= 2) {
        // l2bmi: set, lb rd, lb wr ; l2bmdars: lc rd, dar
        Region R0 = lb(V[V.size()-2], /*w=*/false);
        R0.Set = SetMask;
        Out.push_back(R0);
        Region R1 = lb(V[V.size()-1], /*w=*/true);
        R1.Set = SetMask;
        Out.push_back(R1);
      }
    }
    return;
  }

  // ALU / MAU / other PE exprs (no L1B/L2B direction). Layout by group:
  //
  //   grp 1  mau-calc   : .vsm puts the destination LAST.
  //         vector (face 0):           V = [pe... , pe_dst]
  //         matvec (face X/Y):         V = [mat, pe... , pe_dst]
  //   grp 2  mau-mwrite : V = [pe_src, mat]  (writes the matrix face —
  //                                           not a tracked space)
  //   grp 3  mau-mread  : V = [mat, pe_dst]  (dst is the PE write)
  //   grp 0  other      : scalar ALU dst is a GRF *register* (not a
  //                       peoperand immediate), so nothing here; any
  //                       peoperand immediates are reads.
  //
  // A matside/matoperand immediate is a small row/column index whose
  // low bits can alias a GRF0 peoperand, so we never decode it — its
  // position (first for matvec/mread, last for mwrite) is fixed.
  unsigned grp = mauGrp(MI);
  unsigned face = (MI.getDesc().TSFlags >> 3) & 0x3;

  if (grp == 2) {           // mwrite: only the pe source is a read.
    if (!V.empty()) {
      Region R;
      if (decPE(V[0], R, /*w=*/false)) Out.push_back(R);
    }
    return;
  }
  if (grp == 3) {           // mread: last peoperand is the write.
    Region R;
    if (V.size() >= 2 && decPE(V.back(), R, /*w=*/true)) Out.push_back(R);
    return;
  }
  if (grp == 1) {
    // mau-calc: last peoperand writes, the rest read; skip a leading
    // matrix operand when the expression touches a face. A multi-dst
    // form writes its last NDst PE operands.
    int lo = (face != 0) ? 1 : 0;          // skip mat operand
    int srcEnd = (int)V.size();
    if (MDst)
      srcEnd -= (int)NDst;
    if (srcEnd < lo)
      srcEnd = lo;
    for (int i = lo; i < (int)V.size(); ++i) {
      Region R;
      bool Wr = MDst ? (i >= srcEnd) : (i == (int)V.size() - 1);
      if (decPE(V[i], R, /*w=*/Wr)) Out.push_back(R);
    }
    return;
  }

  // grp 0: no matrix side. The isel-facing scalar ALU writes a GRF
  // *register* (its packed operands are all reads); every assembly-level
  // ALU expression (ALU1P/ALU2P/IMM_P, no register defs) writes its last
  // peoperand, and a multi-destination form writes its last NDst.
  bool RegDefs = MI.getDesc().getNumDefs() != 0;
  int srcEnd0 = (int)V.size();
  if (MDst)
    srcEnd0 -= (int)NDst;
  for (int i = 0; i < (int)V.size(); ++i) {
    Region R;
    bool Wr = MDst ? (i >= srcEnd0)
                   : (!RegDefs && i == (int)V.size() - 1);
    if (decPE(V[i], R, /*w=*/Wr)) Out.push_back(R);
  }
}

// ---------------------------------------------------------------------------
// Gap requirements. For each pair (earlier access E at step se, this
// step's access N) return the extra steps the manual requires between
// them, else 0. L1B rules key off the direction of the expression that
// produced each access, so the direction flags travel with the Region.
// ---------------------------------------------------------------------------

// data-race gap: same space + same address, writer then reader.
static int dataRaceGap(const Access &W, const Access &N,
                       const char **Rule = nullptr) {
  if (!W.R.Write || N.R.Write || W.R.Space != N.R.Space ||
      W.R.Addr != N.R.Addr)
    return 0;
  switch (W.R.Space) {
  case MS_GRF0: case MS_GRF1:
    if (Rule) *Rule = "GRF data-race (3.6.3.9)";
    return 1;
  case MS_T:
    if (Rule) *Rule = "T data-race (3.6.3.9)";
    return 1;
  case MS_LM0: case MS_LM1:   return 0; // LM is port, handled separately
  case MS_L1B:                return 0; // L1B via direction rules
  default:                    return 0;
  }
}

// L1B direction-pair rules (section 3.6.3). The writer/reader direction
// is taken from the expression's l1dir/l2dir; the region must be the
// L1B-side access of that expression.
static int l1bPairGap(const Access &E, const Access &N,
                      const char **Rule = nullptr) {
  if (E.R.Space != MS_L1B || N.R.Space != MS_L1B)
    return 0;

  // Earlier wrote L1B (PE->L1B, L2B->L1B, or internal multicast).
  if (E.R.Write) {
    if (N.R.Write)
      return 0; // write->write not in the table
    bool nToPE   = (N.D1 == 1);         // reader drains to PE
    bool nToL2B  = (N.D2 == 1 || N.D2 == 3); // reader feeds L2B / int-mc
    if (E.D1 == 2) {                    // PE->L1B write
      if (nToPE) {
        if (N.Turn) return 0; // turnaround drain is exempt
        if (Rule) *Rule = "PE->L1B then L1B->PE port (3.6.3.8)";
        return 2;
      }
      if (nToL2B) {
        if (E.R.Addr != N.R.Addr) return 0;
        if (Rule) *Rule = "PE->L1B then L1B->L2B race (3.6.3.7)";
        return 3;
      }
    } else if (E.D2 == 2) {             // L2B->L1B write
      if (nToPE) {
        if (E.R.Addr != N.R.Addr) return 0;
        if (Rule) *Rule = "L2B->L1B then L1B->PE race (3.6.3.6)";
        return 2;
      }
      if (nToL2B) {
        if (!(E.R.Set & N.R.Set)) return 0;
        if (Rule) *Rule = "L2B->L1B then L1B->L2B port (3.6.3.3)";
        return 2;
      }
    } else if (E.D2 == 3) {             // internal multicast write
      if (nToPE) {
        if (E.R.Addr != N.R.Addr) return 0;
        if (Rule) *Rule = "int-mcast then L1B->PE race (3.6.3.5)";
        return 3;
      }
      if (nToL2B) {
        if (!(E.R.Set & N.R.Set)) return 0;
        if (Rule) *Rule = "int-mcast then L1B->L2B port (3.6.3.4)";
        return 3;
      }
    }
    return 0;
  }

      // Earlier read L1B for an L2B-bound transfer; a later L2B->L1B write
      // collides on the port regardless of address.
  if (!E.R.Write && E.D2 == 1 && N.R.Write && N.D2 == 2) {
    if (!(E.R.Set & N.R.Set)) return 0;
    if (Rule) *Rule = "L2B->L1B read then L2B->L1B write port (3.6.3.3)";
    return 3;
  }

  return 0;
}

// L2BM-side rules (3.6.3.1 and 3.6.3.2), keyed off the L2BM access of the
// L1B<->L2B transfer expressions and of MV commands reading L2BM.
static int l2bPairGap(const Access &E, const Access &N,
                      const char **Rule = nullptr) {
  if (E.R.Space != MS_L2B || !E.R.Write || E.D2 != 1)
    return 0; // only an L1B->L2B write starts these hazards
  // 3.6.3.1: an MV that reads L2BM must wait one empty step (the checker
  // assumes the MV touches its whole region on issue, so assume overlap).
  if (N.R.Space == MS_L2B && !N.R.Write && N.D2 == 0) {
    if (Rule) *Rule = "L1B->L2B then MV reads L2BM (3.6.3.1)";
    return 1;
  }
  // 3.6.3.2: a following L2B->L1B transfer collides on the L2BM port.
  // (The L1B side of the same pair is handled in l1bPairGap.)
  if (N.R.Space == MS_L2B && !N.R.Write && N.D2 == 2) {
    if (Rule) *Rule = "L1B->L2B then L2B->L1B port (3.6.3.2)";
    return 3;
  }
  return 0;
}

// port-collision gap: same LM space, writer then reader (any address).
static int portGap(const Access &W, const Access &N,
                   const char **Rule = nullptr) {
  if ((W.R.Space == MS_LM0 || W.R.Space == MS_LM1) && W.R.Write &&
      !N.R.Write && W.R.Space == N.R.Space) {
    if (Rule) *Rule = "LM port (3.6.3.9)";
    return 2;
  }
  const char *R1 = nullptr, *R2 = nullptr;
  int G1 = l1bPairGap(W, N, &R1), G2 = l2bPairGap(W, N, &R2);
  if (G1 >= G2) {
    if (Rule) *Rule = R1;
    return G1;
  }
  if (Rule) *Rule = R2;
  return G2;
}

static const char *spaceName(int Space) {
  switch (Space) {
  case MS_GRF0: return "GRF0";
  case MS_GRF1: return "GRF1";
  case MS_LM0: return "LM0";
  case MS_LM1: return "LM1";
  case MS_T: return "T";
  case MS_L1B: return "L1B";
  case MS_L2B: return "L2B";
  default: return "?";
  }
}

bool MNCoreHazard::runOnMachineFunction(MachineFunction &MF) {
  if (!TII)
    TII = static_cast<const MNCoreInstrInfo *>(
        MF.getSubtarget().getInstrInfo());

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // History of prior accesses: (access, step).
    struct Hist { Access A; int Step; };
    std::vector<Hist> Hist_;
    int Step = 0;

    // MachineBasicBlock::instr_iterator visits EVERY MI, including the
    // bundled members that the plain ::iterator skips.
    for (auto It = MBB.instr_begin(); It != MBB.instr_end();) {
      // Find the extent of this emitted step. A BUNDLE pseudo heads a
      // multi-expression step (the following isInsideBundle MIs are its
      // members); any other MI is a step by itself.
      auto StepFirst = It;
      auto StepEnd = std::next(StepFirst);
      if (It->isBundle())
        while (StepEnd != MBB.instr_end() && StepEnd->isInsideBundle())
          ++StepEnd;

      // Collect this step's tracked accesses.
      SmallVector<Access, 8> Cur;
      DebugLoc DL;
      for (auto J = StepFirst; J != StepEnd; ++J) {
        if (isMV(*J)) {
          // MV operand order is (params, src, dst); an L2BM operand (kind
          // 2) is read as src and written as dst (3.6.3.1/.2).
          SmallVector<uint64_t, 4> V;
          imms(*J, V);
          for (unsigned K = 1; K < V.size() && K <= 2; ++K) {
            if ((V[K] & 0x3) != 2)
              continue;
            Region R{MS_L2B, (int64_t)((V[K] >> 4) & 0xffffffffULL),
                     /*write=*/K == 2};
            Cur.push_back({R, /*D1=*/0, /*D2=*/0, /*Turn=*/false});
          }
          continue;
        }
        if (!isPEExpr(*J))
          continue;
        if (!DL)
          DL = J->getDebugLoc();
        SmallVector<Region, 6> Rs;
        regions(*J, Rs);
        for (const Region &r : Rs)
          Cur.push_back({r, l1dir(*J), l2dir(*J), isTurn(*J)});
      }

      // Required gap = max over (history, this-access) pairs of
      // (se + g + 1) - Step: 'g' empty steps must elapse, so the
      // earliest issue step for this access is h.Step + g + 1.
      int need = 0;
      const char *BestRule = nullptr;
      int BestStep = -1;
      const Region *BestE = nullptr, *BestA = nullptr;
      for (const Hist &h : Hist_)
        for (const Access &a : Cur) {
          const char *R1 = nullptr, *R2 = nullptr;
          int G1 = dataRaceGap(h.A, a, &R1);
          int G2 = portGap(h.A, a, &R2);
          int g = std::max(G1, G2);
          const char *R = G1 >= G2 ? R1 : R2;
          if (g <= 0)
            continue;
          int Cand = (h.Step + g + 1) - Step;
          if (Cand > need) {
            need = Cand;
            BestRule = R;
            BestStep = h.Step;
            BestE = &h.A.R;
            BestA = &a.R;
          }
        }
      if (HazardReport && need > 0)
        errs() << "mncore-hazard: " << MF.getName() << ": step " << Step
               << ": insert " << need << " nop(s) for " << BestRule
               << "; writer step " << BestStep << " "
               << spaceName(BestE->Space) << "@" << BestE->Addr << " -> "
               << spaceName(BestA->Space) << "@" << BestA->Addr << "\n";

      // Emit 'need' NOPs before the step (each = one empty step). Insert
      // at the plain-iterator position of the step's first MI (the BUNDLE
      // pseudo for a packet) so the NOPs precede it.
      for (int i = 0; i < need; ++i) {
        BuildMI(MBB, StepFirst->getIterator(), DL, TII->get(MNCore::NOP));
        ++Step;
        Changed = true;
      }

      // Record this step's accesses at its (post-insertion) step index.
      for (const Access &a : Cur)
        Hist_.push_back({a, Step});

      ++Step;
      It = StepEnd;
    }
  }
  return Changed;
}

char MNCoreHazard::ID = 0;
INITIALIZE_PASS(MNCoreHazard, "mncore-hazard", "MN-Core2 Hazard Spacing",
                false, false)

FunctionPass *llvm::createMNCoreHazard() { return new MNCoreHazard(); }
