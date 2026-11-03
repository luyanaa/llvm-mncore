//===- MNCoreInstrFormats.td - MN-Core2 instruction formats -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction formats for MN-Core2.
//
// MN-Core2 has two machine-instruction families:
//
//   * PE instructions (PE commands): a 4-cycle "step" VLIW instruction that
//     simultaneously controls all memory units and functional units below
//     the L2B level. Multiple "instruction expressions" (ALU, MAU, L1BM,
//     L2BM, ...) are combined by the packetizer into a single PE
//     instruction line separated by ';'.
//
//   * MV instructions: asynchronous data-movement commands between the
//     upper memories (PDM, DRAM, L2BM).
//
// At this stage we only define the *textual* assembly-level structure
// (the .vsm language processed by assemble3). Binary encodings are not
// yet specified and every instruction carries a 64-bit placeholder
// encoding field.
//===----------------------------------------------------------------------===//

#ifndef MNCORE_INSTR_FORMATS_TD
#define MNCORE_INSTR_FORMATS_TD

class MNCoreInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<64> Inst;
  let Namespace = "MNCore";

  let AsmString = asmstr;
  let Pattern = pattern;
  let Size = 8;

  let OutOperandList = outs;
  let InOperandList = ins;

  // Default: no scheduling reservation. Concrete instruction classes
  // override this with the functional-unit group that the parallel
  // execution rules assign the expression to.
  let Itinerary = IIC_pseudo;
}

//===----------------------------------------------------------------------===//
// Base classes for the two instruction families.
//===----------------------------------------------------------------------===//

// An instruction expression occupying one functional-unit slot of a PE
// instruction step. PE expressions are packetized by ';' into a single
// PE instruction line. The packetizer marks them as bundled inside a
// BUNDLE instruction group via MachineInstr::setBundledWith* at codegen
// time; there is no TableGen bit for this.
class PEExpr<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : MNCoreInst<outs, ins, asmstr, pattern> {
  // TSFlags bit 0: "this is a PE instruction expression" — the
  // packetizer's isPEExpression tests this bit instead of enumerating
  // every PE-expression opcode (a list that grows with each milestone).
  let TSFlags{0} = 1;
}

// MAU expressions additionally tag their group and touched matrix face
// in TSFlags so the packetizer can enforce the section 3.6.4 rules
// without a generated-enum dependency:
//   [2:1]  mauGrp : 0=none 1=mau-calc 2=mau-mwrite 3=mau-mread
//   [4:3]  face   : 0=none 1=X 2=Y
//   [7:5]  prec   : 0=none 1=d 2=f 3=g 4=h
class MAUExpr<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              int grp = 0, int fc = 0, int pr = 0>
    : PEExpr<outs, ins, asmstr, pattern> {
  let TSFlags{2-1} = grp;
  let TSFlags{4-3} = fc;
  let TSFlags{7-5} = pr;
}

// L1BM / L2BM expressions tag their transfer direction in TSFlags so the
// packetizer can enforce two families of rules the itinerary cannot see:
//
//   * §3.6.4 co-issue — two L1BM expressions may share a step only when
//     one is the turnaround register's complementary-direction partner
//     (a PE->L1B write filling $lbi plus an L1B->PE read draining it).
//     Two same-direction L1BM ops conflict on the same encoding field.
//   * §3.6.3 hazards — the per-direction write->read latencies
//     (data-race / port-collision) the dependency builder consults.
//
// These bits are set by the L1BMC/L1BMCT and L2BM defs directly; the
// packetizer reads them via l1Direction/l2Direction.
//   [9:8]  l1dir : 0=none 1=L1B->PE (read) 2=PE->L1B (write)
//                  3=turnaround (uses $lbi; dir-of-the-read side)
//   [11:10] l2dir: 0=none 1=L1B->L2B 2=L2B->L1B 3=L1B-internal/DAR
//   [13]   vfma  : vfma/vmul — the section 3.6.4 sub-rule that pins the
//                  second input's source/shape to mwrite's input applies
//                  only to this group member (vadd/vpassa are excluded).
//   [14]   zmask : this def carries a zero-flush mask operand (3.6.2.2) as
//                  its second-to-last `ins` operand. The operand's own
//                  present bit says whether this particular instance uses it.
//   [15]   wmask : this def carries a write-mask operand (3.6.2.1) as its
//                  last `ins` operand.
// A masked def's operand list therefore ends with [.., $zmask, $wmask]; the
// PE-access decoder drops those trailing immediates before decoding.
//   [16]   isImm : this is the `imm` / `immu` command (3.6.12.3). Its first
//                  `ins` operand is the literal, not a PE operand, so the
//                  decoder drops it; and the command may not co-issue with
//                  any LM0 access (the LM0 address shares the immediate
//                  bit field).
//   [17]   mdst  : multi-destination form (`<dst_0> [<dst_1>..]`, 3.6.1).
//                  The PE-access decoder reads the write count from
//                  TSFlags[20:18] instead of assuming the last operand is
//                  the only destination.
//   [20:18] ndst : number of PE destination operands of a multi-dst form
//                  (only meaningful when [17] is set).
//   [21]   isMV  : an MV command line (section 3.5), not a PE expression.
//                  The hazard pass counts MV lines as steps and tracks
//                  their L2BM operand for the 3.6.3.1/.2 gaps.

// A complete MV instruction line.
class MVInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : MNCoreInst<outs, ins, asmstr, pattern> {
  let TSFlags{21} = 1;
}

// Pseudo instructions used only inside the compiler.
class PseudoMN<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : MNCoreInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
}

#endif // MNCORE_INSTR_FORMATS_TD
