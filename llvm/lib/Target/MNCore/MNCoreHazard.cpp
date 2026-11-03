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
// All memory accesses come from the shared `mncore::miAccesses` decoder
// (MNCoreRegions.h), the same description the packetizer and the MC
// verifier use. This pass only owns the *spacing rules*; it does not
// re-decode operands.
//
// Minimum gaps, in *empty steps between* writer and reader (the manual's
// "Nステップ空ける" — write at step s, earliest read at s+N+1), from
// section 3.6.3:
//
//   GRF0/GRF1 write -> read     1   (data race, same addr)
//   LM0/LM1   write -> read     2   (port collision, any addr)
//   T         write -> read     1   (data race, same addr)
//   PE->L1B   ->  L1B->PE       2   (port collision, any addr; the
//                                  turnaround drain is exempt)
//   PE->L1B   ->  L1B->L2B/mc   3   (data race, same L1B addr)
//   L2B->L1B  ->  L1B->PE       2   (data race, same L1B addr)
//   L2B->L1B  ->  L1B->L2B/mc   2   (port; only when the L1B sets overlap)
//   int-mcast ->  L1B->PE       3   (data race, same L1B addr)
//   int-mcast ->  L1B->L2B/mc   3   (port; only when the L1B sets overlap)
//   L1B->L2B  ->  L2B->L1B      3   (port collision; only on set overlap)
//   L1B->L2B  ->  MV reads L2B  1   (data race; MV handled by tag)
//
// Multi-MV constraints (section 3.5.9) are not modelled: the manual
// defers their exact form to another document / the assembler's checker.
// A lone `wait` step (section 3.6.13 requires a co-issued expression) is
// not rejected here either — packetization cannot force a bundle.
//
// Not yet modelled: sub-word / MAB-granular address overlap; the
// write-mask narrowing of hazard ports.
//
//===----------------------------------------------------------------------===//

#include "MNCore.h"
#include "MNCoreInstrInfo.h"
#include "MNCoreRegions.h"
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
using mncore::MemAccess;

#define DEBUG_TYPE "mncore-hazard"

static cl::opt<bool> HazardReport(
    "mncore-hazard-report", cl::Hidden,
    cl::desc("Report each hazard gap and the rule that forced it"));

namespace {

static bool isPEExpr(const MachineInstr &MI) {
  return MI.getDesc().TSFlags & 0x1;
}
static bool isMV(const MachineInstr &MI) {
  return (MI.getDesc().TSFlags >> 21) & 1;
}

static const char *spaceName(int Space) {
  switch (Space) {
  case mncore::MS_GRF0: return "GRF0";
  case mncore::MS_GRF1: return "GRF1";
  case mncore::MS_LM0: return "LM0";
  case mncore::MS_LM1: return "LM1";
  case mncore::MS_T: return "T";
  case mncore::MS_OMR: return "OMR";
  case mncore::MS_LM0BAR: return "LM0BAR";
  case mncore::MS_LM1BAR: return "LM1BAR";
  case mncore::MS_L1B: return "L1B";
  case mncore::MS_L2B: return "L2B";
  case mncore::MS_PDM: return "PDM";
  case mncore::MS_DRAM: return "DRAM";
  default: return "?";
  }
}

class MNCoreHazard : public MachineFunctionPass {
public:
  static char ID;
  MNCoreHazard() : MachineFunctionPass(ID) {
    initializeMNCoreHazardPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "MN-Core2 Hazard Spacing"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

// ---------------------------------------------------------------------------
// Gap requirements. For each pair (earlier access E at step se, this
// step's access N) return the extra steps the manual requires between
// them, else 0. L1B rules key off the direction of the expression that
// produced each access, so the direction flags travel with the access.
// ---------------------------------------------------------------------------

// data-race gap: same space + same address, writer then reader.
static int dataRaceGap(const MemAccess &W, const MemAccess &N,
                       const char **Rule = nullptr) {
  if (!W.Write || N.Write || W.Space != N.Space)
    return 0;
  if (!mncore::regionsOverlap(W.Addr, W.Width, N.Addr, N.Width))
    return 0;
  switch (W.Space) {
  case mncore::MS_GRF0:
  case mncore::MS_GRF1:
    if (Rule) *Rule = "GRF data-race (3.6.3.9)";
    return 1;
  case mncore::MS_T:
    if (Rule) *Rule = "T data-race (3.6.3.9)";
    return 1;
  default:
    return 0;
  }
}

// L1B direction-pair rules (section 3.6.3).
static int l1bPairGap(const MemAccess &E, const MemAccess &N,
                      const char **Rule = nullptr) {
  if (E.Space != mncore::MS_L1B || N.Space != mncore::MS_L1B)
    return 0;

  if (E.Write) {
    if (N.Write)
      return 0; // write->write not in the table
    bool nToPE = (N.Dir1 == 1);                 // reader drains to PE
    bool nToL2B = (N.Dir2 == 1 || N.Dir2 == 3); // feeds L2B / int-mc
    if (E.Dir1 == 2) {                          // PE->L1B write
      if (nToPE) {
        if (N.Turn) return 0; // turnaround drain is exempt
        if (Rule) *Rule = "PE->L1B then L1B->PE port (3.6.3.8)";
        return 2;
      }
      if (nToL2B) {
        if (!mncore::regionsOverlap(E.Addr, E.Width, N.Addr, N.Width))
          return 0;
        if (Rule) *Rule = "PE->L1B then L1B->L2B race (3.6.3.7)";
        return 3;
      }
    } else if (E.Dir2 == 2) {                   // L2B->L1B write
      if (nToPE) {
        if (!mncore::regionsOverlap(E.Addr, E.Width, N.Addr, N.Width))
          return 0;
        if (Rule) *Rule = "L2B->L1B then L1B->PE race (3.6.3.6)";
        return 2;
      }
      if (nToL2B) {
        if (!(E.Set & N.Set)) return 0;
        if (Rule) *Rule = "L2B->L1B then L1B->L2B port (3.6.3.3)";
        return 2;
      }
    } else if (E.Dir2 == 3) {                   // internal multicast write
      if (nToPE) {
        if (!mncore::regionsOverlap(E.Addr, E.Width, N.Addr, N.Width))
          return 0;
        if (Rule) *Rule = "int-mcast then L1B->PE race (3.6.3.5)";
        return 3;
      }
      if (nToL2B) {
        if (!(E.Set & N.Set)) return 0;
        if (Rule) *Rule = "int-mcast then L1B->L2B port (3.6.3.4)";
        return 3;
      }
    }
    return 0;
  }

  // Earlier read L1B for an L2B-bound transfer; a later L2B->L1B write
  // collides on the port when their L1B sets overlap.
  if (!E.Write && E.Dir2 == 1 && N.Write && N.Dir2 == 2) {
    if (!(E.Set & N.Set)) return 0;
    if (Rule) *Rule = "L2B->L1B read then L2B->L1B write port (3.6.3.3)";
    return 3;
  }

  return 0;
}

// L2BM-side rules (3.6.3.1 and 3.6.3.2).
static int l2bPairGap(const MemAccess &E, const MemAccess &N,
                      const char **Rule = nullptr) {
  if (E.Space != mncore::MS_L2B || !E.Write || E.Dir2 != 1)
    return 0; // only an L1B->L2B write starts these hazards
  if (N.Space == mncore::MS_L2B && !N.Write && N.Dir2 == 0) {
    if (Rule) *Rule = "L1B->L2B then MV reads L2BM (3.6.3.1)";
    return 1;
  }
  if (N.Space == mncore::MS_L2B && !N.Write && N.Dir2 == 2) {
    if (Rule) *Rule = "L1B->L2B then L2B->L1B port (3.6.3.2)";
    return 3;
  }
  return 0;
}

// port-collision gap: same LM space, writer then reader (any address).
static int portGap(const MemAccess &W, const MemAccess &N,
                   const char **Rule = nullptr) {
  if ((W.Space == mncore::MS_LM0 || W.Space == mncore::MS_LM1) && W.Write &&
      !N.Write && W.Space == N.Space) {
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

bool MNCoreHazard::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // History of prior accesses: (access, step).
    struct Hist { MemAccess A; int Step; };
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

      // Collect this step's tracked accesses through the shared decoder.
      SmallVector<MemAccess, 8> Cur;
      DebugLoc DL;
      for (auto J = StepFirst; J != StepEnd; ++J) {
        if (!isPEExpr(*J) && !isMV(*J))
          continue;
        if (!DL)
          DL = J->getDebugLoc();
        SmallVector<MemAccess, 8> As;
        mncore::miAccesses(*J, As);
        for (const MemAccess &A : As)
          Cur.push_back(A);
      }

      // Required gap = max over (history, this-access) pairs of
      // (se + g + 1) - Step: 'g' empty steps must elapse, so the
      // earliest issue step for this access is h.Step + g + 1.
      int need = 0;
      const char *BestRule = nullptr;
      int BestStep = -1;
      const MemAccess *BestE = nullptr, *BestA = nullptr;
      for (const Hist &h : Hist_)
        for (const MemAccess &a : Cur) {
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
            BestE = &h.A;
            BestA = &a;
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
        BuildMI(MBB, StepFirst->getIterator(), DL,
                MF.getSubtarget().getInstrInfo()->get(MNCore::NOP));
        ++Step;
        Changed = true;
      }

      // Record this step's accesses at its (post-insertion) step index.
      for (const MemAccess &a : Cur)
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
