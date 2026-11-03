//===- MNCorePacketizer.cpp - MN-Core2 VLIW packetizer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Groups PE instruction expressions into the 4-cycle steps they issue
// together in, by bundling MachineInstrs (isBundledWithPred/Succ). The
// AsmPrinter then renders each step as a ';'-separated .vsm line.
//
// The per-group slot reservations (section 3.6.4 "at most one
// expression per group per step") are enforced by the generated
// DFAPacketizer via the itinerary functional units. This pass adds the
// checks the DFA resource model cannot express:
//
//   * at most 2 of the 3 MAU groups (mau-calc, mau-mwrite, mau-mread)
//     per step;
//   * nop is exclusive with every expression except wait;
//   * the same matrix-register face ($lx/$ly) appears at most once;
//   * two expressions never write the same PE operand, reads of one
//     addressable space all use the same address, an LM space read and
//     written in one step covers the same address, and `imm` never
//     co-issues with an LM0 access;
//   * a vfma/vmul's second input matches the co-issued mwrite's input in
//     source operand, sign inversion, extend and reduce;
//   * a zero-flush mask appears on at most one expression per step, and
//     every mask specifier in a step shares one mask word length
//     (the last three groups in MNCoreRegions.h).
//
// (The face rule is also partly covered by MATX/MATY functional units;
// isLegalToPacketizeTogether is the authoritative check and also guards
// the cases the itinerary cannot see, e.g. mwrite's implicit face.)
//
//===----------------------------------------------------------------------===//

#include "MNCoreInstrInfo.h"
#include "MNCoreRegions.h"
#include "MNCoreSubtarget.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mncore-packetizer"

namespace {

class MNCorePacketizerList : public VLIWPacketizerList {
public:
  MNCorePacketizerList(MachineFunction &MF, MachineLoopInfo &MLI)
      : VLIWPacketizerList(MF, MLI, /*AA=*/nullptr),
        TII(static_cast<const MNCoreInstrInfo &>(
            *MF.getSubtarget().getInstrInfo())) {}

  // MV instructions and target pseudos never join a PE step.
  bool isSoloInstruction(const MachineInstr &MI) override {
    return !isPEExpression(MI);
  }

  // Pseudos and MV lines are not scheduled into a packet at all.
  bool ignorePseudoInstruction(const MachineInstr &I,
                               const MachineBasicBlock *MBB) override {
    return I.isPseudo() || I.getOpcode() == MNCore::BUNDLE ||
           !isPEExpression(I);
  }

  // The extra legality checks that the resource DFA cannot express.
  bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) override;

  // isLegalToPacketizeTogether is called only to decide whether the
  // packetizer may *drop* a scheduling-dependency edge that the resource
  // model already guards — it is NOT a way to waive a structural
  // illegality. If we returned true unconditionally, the pair would be
  // bundled even when isLegalToPacketizeTogether rejected it (e.g. a
  // nop next to an ALU op). An illegal pair must never packetize, so
  // refuse to prune.
  bool isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ) override {
    return false;
  }

private:
  const MNCoreInstrInfo &TII;

  // True for the MachineInstrs that become PE instruction expressions
  // in one step (i.e. the PEExpr TableGen family). MV lines and
  // pseudos return false.
  bool isPEExpression(const MachineInstr &MI) const;

  // Group membership helpers for the §3.6.4 rules.
  static bool isNop(const MachineInstr &MI) {
    return MI.getOpcode() == MNCore::NOP;
  }
  static bool isWait(const MachineInstr &MI) {
    return MI.getOpcode() == MNCore::WAIT;
  }
  // MAU group: mau-calc, mau-mwrite, mau-mread.
  enum MAUGroup { NoMAU, MAUCalc, MAUWrite, MAURead };
  MAUGroup mauGroup(const MachineInstr &MI) const;
  // Which matrix-register face does MI touch, if any? 0=none, 1=X, 2=Y.
  unsigned matrixFace(const MachineInstr &MI) const;
  // Precision letter of a MAU expression (d/f/g/h), 0 if not MAU —
  // the §3.6.4 "same precision" sub-rule applies when two of the three
  // MAU groups co-issue. Read from the opcode text.
  char mauPrecision(const MachineInstr &MI) const;

  // L1BM / L2BM transfer direction (TSFlags [9:8] / [11:10]), used for
  // the §3.6.4 co-issue rule and the §3.6.3 hazard latencies.
  //   l1dir: 0=none 1=L1B->PE(read) 2=PE->L1B(write) 3=turnaround
  //   l2dir: 0=none 1=L1B->L2B 2=L2B->L1B 3=L1B-internal/DAR
  static unsigned l1Direction(const MachineInstr &MI) {
    return (unsigned)((MI.getDesc().TSFlags >> 8) & 0x3);
  }
  static unsigned l2Direction(const MachineInstr &MI) {
    return (unsigned)((MI.getDesc().TSFlags >> 10) & 0x3);
  }
  static bool isL1BM(const MachineInstr &MI) { return l1Direction(MI) != 0; }
  // Turnaround ($lbi) form — set on the *_T twin via TSFlags bit 12.
  static bool l1IsTurn(const MachineInstr &MI) {
    return (MI.getDesc().TSFlags >> 12) & 1;
  }
};

bool MNCorePacketizerList::isPEExpression(const MachineInstr &MI) const {
  // TSFlags bit 0 is set by the TableGen PEExpr class on every PE
  // instruction expression; MV lines and pseudos leave it clear.
  return MI.getDesc().TSFlags & 0x1;
}

// TSFlags bit layout for PE expressions (see MAUExpr in
// MNCoreInstrFormats.td):
//   [0]    isPEExpr
//   [2:1]  mauGrp : 0=none 1=mau-calc 2=mau-mwrite 3=mau-mread
//   [4:3]  face   : 0=none 1=X 2=Y
//   [7:5]  prec   : 0=none 1=d 2=f 3=g 4=h
static uint64_t mncoreFlags(const MachineInstr &MI) {
  return MI.getDesc().TSFlags;
}

MNCorePacketizerList::MAUGroup
MNCorePacketizerList::mauGroup(const MachineInstr &MI) const {
  return (MAUGroup)((mncoreFlags(MI) >> 1) & 0x3);
}

unsigned MNCorePacketizerList::matrixFace(const MachineInstr &MI) const {
  return (unsigned)((mncoreFlags(MI) >> 3) & 0x3);
}

char MNCorePacketizerList::mauPrecision(const MachineInstr &MI) const {
  static const char P[] = {0, 'd', 'f', 'g', 'h'};
  unsigned p = (unsigned)((mncoreFlags(MI) >> 5) & 0x7);
  return p < 5 ? P[p] : 0;
}

bool MNCorePacketizerList::isLegalToPacketizeTogether(SUnit *SUI,
                                                    SUnit *SUJ) {
  const MachineInstr &I = *SUI->getInstr();
  const MachineInstr &J = *SUJ->getInstr();

  // nop is exclusive with everything except wait.
  if ((isNop(I) || isNop(J)) && !(isWait(I) || isWait(J)) &&
      !(isNop(I) && isNop(J)))
    return false;

  // At most 2 of the 3 MAU groups per step; when two different MAU
  // groups co-issue their precision letters must match (§3.6.4).
  // isLegalToPacketizeTogether is called pairwise, so to enforce the
  // "2 of 3 groups" cap we also count the groups already in the
  // accumulating packet (CurrentPacketMIs) plus the candidate pair.
  MAUGroup GI = mauGroup(I), GJ = mauGroup(J);
  if (GI != NoMAU || GJ != NoMAU) {
    unsigned GroupMask = 0;
    auto setBit = [&](MAUGroup g) {
      if (g != NoMAU)
        GroupMask |= 1u << (unsigned)g;
    };
    setBit(GI);
    setBit(GJ);
    for (MachineInstr *MI : CurrentPacketMIs)
      setBit(mauGroup(*MI));
    unsigned Groups = 0;
    for (unsigned B = GroupMask; B; B &= B - 1)
      ++Groups;
    if (Groups > 2)
      return false;
    // Same-precision sub-rule applies to the two issued MAU groups.
    char PI = mauPrecision(I), PJ = mauPrecision(J);
    if (PI != 0 && PJ != 0 && GI != GJ && PI != PJ)
      return false;
    // Check candidate against existing packet members too.
    for (MachineInstr *MI : CurrentPacketMIs) {
      MAUGroup G = mauGroup(*MI);
      char P = mauPrecision(*MI);
      if (G == NoMAU || P == 0)
        continue;
      if (GI != NoMAU && GI != G && PI != 0 && PI != P)
        return false;
      if (GJ != NoMAU && GJ != G && PJ != 0 && PJ != P)
        return false;
    }
  }

  // The same matrix-register face may be referenced at most once.
  unsigned FI = matrixFace(I), FJ = matrixFace(J);
  if (FI != 0 && FI == FJ)
    return false;

  // §3.6.4 L1BM co-issue: at most one l1bm and one l1bm-turnaround slot
  // per step (the DFA enforces that), and the two may co-issue only when
  // they move data in *opposite* directions — a PE->L1B write that fills
  // the turnaround register plus the L1B->PE read that drains it. Two
  // same-direction L1BM ops always collide on the encoding field
  // (broadcast+broadcast or collect+collect => "field conflict").
  if (isL1BM(I) && isL1BM(J)) {
    unsigned DI = l1Direction(I), DJ = l1Direction(J);
    if (DI == DJ)
      return false; // same direction => field conflict
  }

  // \xC2\xA73.6.4 operand rules the resource DFA cannot see. The packetizer calls
  // this once per instruction already in the packet, so judging the single
  // pair (SUI, SUJ) covers the whole step: no two expressions write the
  // same PE operand, reads of one addressable space share one address, an
  // LM space read and written in one step covers the same address, and
  // `imm` forbids any LM0 access.
  if (mncore::peConflict(I, J))
    return false;

  // The vfma/vmul + mwrite sub-rule: when those two co-issue, the vector
  // expression's second input and the mwrite input must read the same PE
  // operand and agree on sign inversion / extend / reduce.
  if (mncore::vfmaMwriteShapeConflict(I, J))
    return false;

  // \xC2\xA73.6.4 step-wide mask rules: a zero-flush mask on at most one expression
  // per step, and every present mask specifier agreeing on the mask word
  // length. These are not pairwise, so check the prospective step: the
  // packet already accumulated plus the instruction being added (SUI). Do
  // not add SUJ as well — it is already one of CurrentPacketMIs, and
  // counting a member twice would double-count its masks.
  {
    SmallVector<const MachineInstr *, 4> Step;
    Step.push_back(&I); // SUI — the newcomer
    for (MachineInstr *MI : CurrentPacketMIs)
      Step.push_back(MI);
    if (mncore::maskStepConflict(Step))
      return false;
  }

  return true;
}

class MNCorePacketizer : public MachineFunctionPass {
public:
  static char ID;
  MNCorePacketizer() : MachineFunctionPass(ID) {
    initializeMNCorePacketizerPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "MN-Core2 Packetizer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    MNCorePacketizerList Packetizer(MF, MLI);
    assert(Packetizer.getResourceTracker() && "Empty DFA table!");

    for (MachineBasicBlock &MBB : MF) {
      // Per-instruction grammar rules (fixed-value/mreadf placement,
      // $nowrite exclusivity) are not repairable by reordering, so reject
      // them outright rather than silently emitting illegal .vsm.
      for (MachineInstr &MI : MBB)
        if ((MI.getDesc().TSFlags & 0x1) &&
            mncore::peInstructionConflict(MI))
          report_fatal_error(
              "MN-Core2: illegal PE operand placement (fixed-value/$mreadf "
              "outside the first ALU input, or $nowrite with another output)",
              false);
      MachineBasicBlock::iterator Begin = MBB.begin();
      MachineBasicBlock::iterator End = MBB.end();
      if (Begin != End)
        Packetizer.PacketizeMIs(&MBB, Begin, End);
    }
    return true;
  }
};

char MNCorePacketizer::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(MNCorePacketizer, "mncore-packetizer",
                "MN-Core2 Packetizer", false, false)

FunctionPass *llvm::createMNCorePacketizer() {
  return new MNCorePacketizer();
}
