//===- MNCoreSchedule.td - MN-Core2 scheduling model ----------*- tablegen -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduling model for the MN-Core2 PE instruction word.
//
// The PE executes a single VLIW "step" every 4 cycles.  Each step is a
// bundle of PE instruction *expressions*, one per functional-unit group
// defined by the parallel-execution rules of section 3.6.4 of the
// MN-Core2 hardware manual:
//
//   Group Name        Issued expressions
//   nop               nop
//   noforward         noforward
//   l2bm              any L2BM expression other than l2bmdarw
//   l2bmdarw          l2bmdarw
//   l1bm              any non-turnaround L1BM expression
//   l1bm-turnaround   any turnaround L1BM expression
//   mau-calc          mfma, mmul, vfma, vmul, vadd, vpassa
//   mau-mwrite        mwrite
//   mau-mread         mread
//   alu               any ALU expression
//   wait              wait
//
// Each group is a single functional unit; the packetizer's DFA reserves
// one slot per group per step, which is exactly the "at most one
// expression per group per step" constraint. The matrix-register face
// units MATX/MATY are reserved by an extra InstrStage on face-touching
// MAU forms. The remaining section 3.6.4 rules (nop exclusivity, at
// most 2 of 3 MAU groups, matching operand shapes) are not resource
// capacities and are enforced in the packetizer's
// isLegalToPacketizeTogether instead.
//
// Hazard avoidance (section 3.6.3) is captured through the operand
// latency columns of the itinerary data.  The numbers below are the
// *safe* upper bounds the assembler uses; the manual sometimes allows a
// tighter spacing for specific address overlaps, but using the safe
// bound never produces illegal code.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units: one slot per parallel-execution group.
//
// (TargetItinerary.td / TargetSchedule.td are already included by
// Target.td.)
//===----------------------------------------------------------------------===//

// The "issue" FU is a degenerate reservation used for pseudo/control
// instructions that are emitted outside the VLIW step (MV commands and
// target pseudo instructions).
def MNCoreUnitIssue : FuncUnit;

def MNCoreUnitNop        : FuncUnit;  // nop
def MNCoreUnitNoforward  : FuncUnit;  // noforward
def MNCoreUnitL2BM       : FuncUnit;  // l2bm*
def MNCoreUnitL2BMDARW   : FuncUnit;  // l2bmdarw
def MNCoreUnitL1BM       : FuncUnit;  // l1bm* (non-turnaround)
def MNCoreUnitL1BMT      : FuncUnit;  // l1bm* turnaround
def MNCoreUnitMAUCalc    : FuncUnit;  // mfma/mmul/vfma/vmul/vadd/vpassa
def MNCoreUnitMAUWrite   : FuncUnit;  // mwrite
def MNCoreUnitMAURead    : FuncUnit;  // mread
def MNCoreUnitALU        : FuncUnit;  // ladd/lsub/... and l* ALU exprs
def MNCoreUnitWait       : FuncUnit;  // wait iXX

// Extra functional units that encode the parallel-execution rules of
// section 3.6.4 which a single slot per group cannot express by
// itself.
//
//   * MATX / MATY: the same matrix-register face ($lx or $ly) may be
//     referenced at most once per step. A mau-calc reads one face and a
//     mwrite/mread writes/reads one face, so every instruction that
//     touches a face reserves the matching unit via a second InstrStage.
//     Variants of the MAU itinerary classes (below) select the right
//     face.
//
// "nop is exclusive with every expression except wait" is NOT a
// resource-capacity rule — it cannot be encoded by a shared unit,
// because any unit every expression reserves would also block two
// different groups from co-issuing. It is a forbidden-pair constraint
// and is checked in the packetizer's isLegalToPacketizeTogether, along
// with "at most 2 of the 3 MAU groups" and the same-precision /
// operand-shape rules that resource reservations cannot express without
// wrongly forbidding legal pairs.
def MNCoreUnitMATX : FuncUnit;   // $lx referenced this step
def MNCoreUnitMATY : FuncUnit;   // $ly referenced this step

//===----------------------------------------------------------------------===//
// Itinerary classes.
//===----------------------------------------------------------------------===//

// An "expression" itinerary carries the single-cycle reservation on its
// functional unit plus the read/def operand cycles that encode the
// hazard rules of section 3.6.3/3.6.3.9.
//
// IMPORTANT: an InstrStage's unit list is a *choice* of units (OR), not
// a set that is reserved together. To make one expression consume two
// units in the same step — e.g. its group slot AND the nop-exclusivity
// unit NOPX, or a MAU group slot AND a matrix-face unit MATX/MATY — the
// itinerary lists one InstrStage per unit. The DFA packetizer consumes
// one unit per stage, so listing N stages reserves all N units.
//
// OperandCycles layout: [def0, def1, ..., use0, use1, ...] for the
// operands as they appear in the MachineInstr's operand list (outs
// first, then ins). A value of N means the operand is
// defined/read N cycles after issue.  We express the rules in *cycles*
// (1 step = 4 cycles); the table below uses the "always safe" bounds.

// Scalar ALU: GRF write is visible to a reader 1 step later (4 cycles);
// the ALU reads its inputs at issue.
def IIC_alu : InstrItinClass;

// Assembly-level ALU expressions whose operands are all packed peoperand
// immediates (MNCoreInstrALU.td). They reserve the same ALU slot, but they
// have no register defs or uses, so the itinerary carries no operand
// cycles: the read-after-write spacing for the memories they touch is
// handled by the hazard pass, not by the scheduler. Giving them IIC_alu's
// [4, 0, 0] cycles would stamp a latency onto an input immediate and invent
// a scheduling edge that blocks legal bundling.
def IIC_alupe : InstrItinClass;

// Constants written to the register file (zero / imm) follow the same
// "1 step later" read-after-write rule.
def IIC_const : InstrItinClass;

// NOP / NOFORWARD occupy their own slot for one step but have no
// operands.  Each gets its own class because the itinerary class fixes
// the functional-unit reservation.
def IIC_nop : InstrItinClass;
def IIC_noforward : InstrItinClass;

// WAIT only reserves the wait slot; the MV-side synchronisation is
// handled by the tag, not by the step scheduler.
def IIC_wait : InstrItinClass;

// L1BM->PE broadcast (l1bmp): the L1BM read happens at issue, the GRF
// write becomes visible 6 cycles later (the manual's "write completes
// in 6 cycles" rule for PE memory), giving a 2-step safe gap before a
// dependent reader.
def IIC_l1bmp : InstrItinClass;

// Collective L1BM/L2BM traffic reserves only its group slot.
def IIC_l1bm  : InstrItinClass;
def IIC_l2bm  : InstrItinClass;
def IIC_l1bmt : InstrItinClass;   // turnaround ($lbi) forms
def IIC_l2bmdarw : InstrItinClass; // l2bmdarw's own group

// MAU expressions. There is one itinerary class per (group, face)
// combination so the DFA can enforce the "same matrix-register face at
// most once per step" rule. Instructions that do not touch a matrix
// face use the plain group class.
def IIC_mau    : InstrItinClass;  // vector ops, no matrix face
def IIC_mauX   : InstrItinClass;  // mau-calc reading $lx
def IIC_mauY   : InstrItinClass;  // mau-calc reading $ly
def IIC_mwr    : InstrItinClass;  // mwrite, no face (n/a) — placeholder
def IIC_mwrX   : InstrItinClass;  // mwrite -> $lx
def IIC_mwrY   : InstrItinClass;  // mwrite -> $ly
def IIC_mrdX   : InstrItinClass;  // mread  <- $lx
def IIC_mrdY   : InstrItinClass;  // mread  <- $ly

// Pseudo / MV instructions never issue inside a PE step.
def IIC_pseudo : InstrItinClass;

//===----------------------------------------------------------------------===//
// Itinerary data.
//===----------------------------------------------------------------------===//

def MNCoreItins : ProcessorItineraries<
  [MNCoreUnitIssue,
   MNCoreUnitNop, MNCoreUnitNoforward,
   MNCoreUnitL2BM, MNCoreUnitL2BMDARW,
   MNCoreUnitL1BM, MNCoreUnitL1BMT,
   MNCoreUnitMAUCalc, MNCoreUnitMAUWrite, MNCoreUnitMAURead,
   MNCoreUnitALU, MNCoreUnitWait,
   MNCoreUnitMATX, MNCoreUnitMATY],
  [],
  [
    // Scalar ALU: one cycle on the ALU slot; def at cycle 4 (1 step),
    // uses read at cycle 0.
    InstrItinData<IIC_alu,
                  [InstrStage<1, [MNCoreUnitALU]>],
                  [4, 0, 0]>,
    // Packed-immediate ALU: same slot, no register operand cycles.
    InstrItinData<IIC_alupe,
                  [InstrStage<1, [MNCoreUnitALU]>],
                  []>,
    // Constants (zero/imm) are ALU expressions too — same group slot.
    InstrItinData<IIC_const,
                  [InstrStage<1, [MNCoreUnitALU]>],
                  [4]>,
    // nop / noforward occupy their own single-slot groups.
    InstrItinData<IIC_nop,
                  [InstrStage<1, [MNCoreUnitNop]>],
                  []>,
    InstrItinData<IIC_noforward,
                  [InstrStage<1, [MNCoreUnitNoforward]>],
                  []>,
    // wait: its own group; allowed alongside nop.
    InstrItinData<IIC_wait,  [InstrStage<1, [MNCoreUnitWait]>], [0]>,
    // l1bmp: L1BM read at issue, GRF write visible 6 cycles later.
    InstrItinData<IIC_l1bmp,
                  [InstrStage<1, [MNCoreUnitL1BM]>],
                  [6, 0]>,
    // Collective L1BM / L2BM traffic reserves only its group slot.
    InstrItinData<IIC_l1bm,
                  [InstrStage<1, [MNCoreUnitL1BM]>],
                  []>,
    InstrItinData<IIC_l1bmt,
                  [InstrStage<1, [MNCoreUnitL1BMT]>],
                  []>,
    InstrItinData<IIC_l2bm,
                  [InstrStage<1, [MNCoreUnitL2BM]>],
                  []>,
    // l2bmdarw: own group, co-issues with any l2bm expression.
    InstrItinData<IIC_l2bmdarw,
                  [InstrStage<1, [MNCoreUnitL2BMDARW]>],
                  []>,
    // MAU: stage 1 reserves the group slot; the face-touching forms add
    // a second stage reserving the matching MATX/MATY unit so two
    // references to the same face cannot share a step.
    InstrItinData<IIC_mau,
                  [InstrStage<1, [MNCoreUnitMAUCalc]>],
                  []>,
    InstrItinData<IIC_mauX,
                  [InstrStage<1, [MNCoreUnitMAUCalc]>,
                   InstrStage<1, [MNCoreUnitMATX]>],
                  []>,
    InstrItinData<IIC_mauY,
                  [InstrStage<1, [MNCoreUnitMAUCalc]>,
                   InstrStage<1, [MNCoreUnitMATY]>],
                  []>,
    InstrItinData<IIC_mwr,
                  [InstrStage<1, [MNCoreUnitMAUWrite]>],
                  []>,
    InstrItinData<IIC_mwrX,
                  [InstrStage<1, [MNCoreUnitMAUWrite]>,
                   InstrStage<1, [MNCoreUnitMATX]>],
                  []>,
    InstrItinData<IIC_mwrY,
                  [InstrStage<1, [MNCoreUnitMAUWrite]>,
                   InstrStage<1, [MNCoreUnitMATY]>],
                  []>,
    InstrItinData<IIC_mrdX,
                  [InstrStage<1, [MNCoreUnitMAURead]>,
                   InstrStage<1, [MNCoreUnitMATX]>],
                  []>,
    InstrItinData<IIC_mrdY,
                  [InstrStage<1, [MNCoreUnitMAURead]>,
                   InstrStage<1, [MNCoreUnitMATY]>],
                  []>,
    // Pseudos / MV lines take the generic issue slot and never join a
    // PE step.
    InstrItinData<IIC_pseudo,[InstrStage<1, [MNCoreUnitIssue]>],[]>
  ]>;

//===----------------------------------------------------------------------===//
// Machine model.
//===----------------------------------------------------------------------===//

def MNCoreModel : SchedMachineModel {
  // One expression per group per step -> 12 issue slots, but the
  // dominant constraint is the per-group single-slot reservation, so a
  // generous issue width simply avoids the packetizer capping early.
  let IssueWidth = 12;
  let Itineraries = MNCoreItins;
  // In-order, non-pipelined issue: a resource held at issue is busy
  // for the whole step.
  let MicroOpBufferSize = 0;
  // We do not model loads through the generic LLVM memory model.
  let LoadLatency = 1;
  // Only a subset of opcodes carries scheduling data during bring-up.
  let CompleteModel = 0;
}
