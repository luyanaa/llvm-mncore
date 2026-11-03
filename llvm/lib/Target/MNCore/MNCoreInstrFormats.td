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

// A complete MV instruction line.
class MVInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : MNCoreInst<outs, ins, asmstr, pattern>;

// Pseudo instructions used only inside the compiler.
class PseudoMN<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : MNCoreInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
}

#endif // MNCORE_INSTR_FORMATS_TD
