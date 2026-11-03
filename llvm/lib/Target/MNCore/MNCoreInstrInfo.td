//===- MNCoreInstrInfo.td - MN-Core2 instruction info ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level MN-Core2 instruction description. This file defines a small
// M0/M1 instruction subset used by the initial direct-line kernel
// backend:
//
//   * control expressions: nop, noforward, wait
//   * scalar ALU expressions: ladd, lsub, land, lor, lxor, linc, ldec,
//     lnot, zero, imm
//   * one simple L1BM->PE broadcast expression (l1bmp)
//   * a pseudo RET marker used while lowering `ret` in a straight-line
//     kernel
//
// The instruction set is intentionally incomplete. Later milestones add
// the remaining ALU opcodes, MAU, L1BM/L2BM, and MV instruction forms.
//===----------------------------------------------------------------------===//

include "MNCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand types
//
// (i64imm comes from llvm/Target/Target.td.)
//===----------------------------------------------------------------------===//

// Signed immediates used by ALU pseudo expansion before .vsm emission.
def simm64 : Operand<i64>, ImmLeaf<i64, [{ return true; }]>;

// Constants that fit in a signed 32-bit single word — the only range
// the `i"..."` imm literal can express.
def simm32 : ImmLeaf<i64, [{
  return isInt<32>(Imm);
}]>;

//===----------------------------------------------------------------------===//
// Selection DAG node types
//===----------------------------------------------------------------------===//

def SDT_MNCoreCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i64>,
                                             SDTCisVT<1, i64>]>;
def SDT_MNCoreCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i64>,
                                           SDTCisVT<1, i64>]>;
def SDT_MNCoreRet          : SDTypeProfile<0, 0, []>;

def MNCoreRetFlag : SDNode<"MNCoreISD::RET", SDT_MNCoreRet,
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// Control expressions (PE instruction expressions)
//===----------------------------------------------------------------------===//

let Itinerary = IIC_nop in
def NOP : PEExpr<(outs), (ins), "nop">;

let Itinerary = IIC_noforward in
def NOFORWARD : PEExpr<(outs), (ins), "noforward">;

// wait <tag> - stall issue until the MV instruction tagged <tag>
// completes. The tag is a 2-hex-digit iXX literal; we store the numeric
// tag value as an immediate and print it in the iXX form.
let PrintMethod = "printTagOperand" in
  def tagimm : Operand<i64>;

let Itinerary = IIC_wait in
def WAIT : PEExpr<(outs), (ins tagimm:$tag), "wait $tag">;


//===----------------------------------------------------------------------===//
// ALU expressions (scalar core)
//
// These are defined in the .vsm syntax:
//
//    [u][precision]<opcode> <src_x> [<src_y>] <dst_0> [<dst_1>..]
//
// For M1 we support long-word integer ops (l-prefix) operating on the
// GRF register files. Operand order in .vsm is "inputs then outputs";
// LLVM's (outs, ins) convention is inverted, so the AsmString reorders
// accordingly.
//===----------------------------------------------------------------------===//

let Itinerary = IIC_alu in {
class ALU2<string opc>
    : PEExpr<(outs GRF:$dst), (ins GRF:$x, GRF:$y),
             !strconcat(opc, " $x $y $dst")>;

class ALU1<string opc>
    : PEExpr<(outs GRF:$dst), (ins GRF:$x),
             !strconcat(opc, " $x $dst")>;
}

def LADD : ALU2<"ladd">;
def LSUB : ALU2<"lsub">;
def LAND : ALU2<"land">;
def LOR  : ALU2<"lor">;
def LXOR : ALU2<"lxor">;
def LINC : ALU1<"linc">;
def LDEC : ALU1<"ldec">;
def LNOT : ALU1<"lnot">;
// lsl with register operands: the shift amount is the second input
// (3.6.12.17). Used by the wide-constant materialization in
// MNCoreDAGToDAGISel::Select; the assembly-level packed form lives in
// MNCoreInstrALU.td.
def LSLR : ALU2<"lsl">;

// zero <dst>: write all-zero to the destination PE memory.
let isReMaterializable = 1, Itinerary = IIC_const in
def ZERO : PEExpr<(outs GRF:$dst), (ins), "zero $dst">;

// imm <payload> <dst>: ALU immediate command. The .vsm literal is a
// typed single-word value, e.g. `i"42"` or `f"1.0"` — the ALU replicates
// it into a 2-longword per §3.6.12.3. We materialize the low 32 bits of
// an i64 constant as a signed word (`i"..."`); a float path and the
// unsigned/half forms are added when FP lowering lands.
let PrintMethod = "printImmLiteral" in
  def immlit : Operand<i64>;

// Typed immediate literal (section 3.2.2). Packs
//   [3:0]  type    : 0=i 1=s 2=ui 3=us 4=f 5=h
//   [35:4] payload : the 32-bit word the Alu places in every word slot.
//                    For s/us/f/h the word is the 16-bit value replicated
//                    twice (the half-precision numeric types).
// The `u` (xoxo) option belongs to the opcode (immu), not the operand.
let PrintMethod = "printImmTyped" in
  def immlitx : Operand<i64>;

// `imm` is the isel-facing materialization of a signed 32-bit constant
// into a GRF register; its literal prints as `i"<s32>"`. TSFlags[16]
// marks it as an imm command (see MNCoreInstrFormats.td).
let isReMaterializable = 1, Itinerary = IIC_const in
def IMM : PEExpr<(outs GRF:$dst), (ins immlit:$imm), "imm $imm $dst"> {
  let TSFlags{16} = 1;
}

// immu <payload> <dst>: the `u` (xoxo) option of 3.6.12.3 — the 32-bit
// word is placed in the odd word slots and the other two are zero, so
// the high half of the 64-bit result holds the word and the low half is
// zero. Used to materialize constants with a zero low word.
let isReMaterializable = 1, Itinerary = IIC_const in
def IMMU : PEExpr<(outs GRF:$dst), (ins immlit:$imm), "immu $imm $dst"> {
  let TSFlags{16} = 1;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// Call-boundary markers required by TargetInstrInfo's ctor arguments.
// MN-Core2 has no call stack; these expand to comments in the .vsm
// output and are elided by the asm printer.
def ADJCALLSTACKDOWN
    : PseudoMN<(outs), (ins i64imm:$amt, i64imm:$amt2),
               "# ADJCALLSTACKDOWN $amt, $amt2">;
def ADJCALLSTACKUP
    : PseudoMN<(outs), (ins i64imm:$amt1, i64imm:$amt2),
               "# ADJCALLSTACKUP $amt1">;

// Return marker for a straight-line kernel. Real MN-Core2 code has no
// call/return; this is emitted at the end of the function and printed
// as a comment (or `quit` when truncation is desired).  It carries the
// kernel's result values as variable operands purely to keep them live
// through isel.
let hasCtrlDep = 1 in
def RET : PseudoMN<(outs), (ins variable_ops), "# kernel return">;

// Copies / spill placeholders so the register allocator has a legal
// target for moves between GRF0 and GRF1 and for rematerialization.
def COPY_GRF : PseudoMN<(outs GRF:$dst), (ins GRF:$src),
                        "# copy $src -> $dst">;

//===----------------------------------------------------------------------===//
// Selection patterns (M1: minimal scalar lowering)
//===----------------------------------------------------------------------===//

// `ret` at the end of a kernel is selected to the RET pseudo in
// MNCoreDAGToDAGISel::Select (it carries variable operands that TableGen
// patterns cannot express).

def : Pat<(i64 (add GRF:$x, GRF:$y)), (LADD GRF:$x, GRF:$y)>;
def : Pat<(i64 (sub GRF:$x, GRF:$y)), (LSUB GRF:$x, GRF:$y)>;
def : Pat<(i64 (and GRF:$x, GRF:$y)), (LAND GRF:$x, GRF:$y)>;
def : Pat<(i64 (or  GRF:$x, GRF:$y)), (LOR  GRF:$x, GRF:$y)>;
def : Pat<(i64 (xor GRF:$x, GRF:$y)), (LXOR GRF:$x, GRF:$y)>;

// Materialize a 32-bit-range constant with the ALU `imm` command
// (`i"..."` literal). Wider constants need a multi-step sequence that
// is not yet implemented.
def : Pat<(i64 simm32:$imm), (IMM simm32:$imm)>;
