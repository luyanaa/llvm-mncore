//===- MNCoreInstrMV.td - MN-Core2 MV data-movement commands -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MV instructions (section 3.5) are asynchronous data-movement commands
// between the upper memories (PDM, DRAM, L2BM). Unlike PE expressions
// they are never ';'-joined: each is a single complete statement on its
// own line, issued at most once per 4-step unit.
//
// The machine opcode is split by '/' into a "basic mode" part (mvp, mvb,
// mvb2, mvb4, mvd, mvr<op>, mvr2<op>, mvr4<op>) and a "parameter" part
// carrying n<size> (transfer word count, mandatory), i<tag> (wait tag),
// nd<N> (DAR reuse count for DRAM-indirect) and p<prio> (priority 0-3).
//
// The parameter text appears inside the mnemonic, before the memory
// operands, so it cannot be a trailing operand. We therefore carry the
// four parameters packed in a single leading operand `mvparams:$params`
// and place it in the AsmString immediately after the opcode stem:
//   "mvp$params $src, $dst"  ->  "mvp/n64i01 $p0@0 $lc0@2.1"
//
//===----------------------------------------------------------------------===//

include "MNCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

// A whole MV memory operand ($p/$d/$di/$lc plus @group/.l2b qualifiers)
// packed into one i64 immediate. Field layout (little-endian bit index):
//   [1:0]   kind   : 0=PDM($p), 1=DRAM($d), 2=L2BM($lc), 3=DRAMi($di)
//   [35:4]  addr   : longword address within the memory
//   [40]    hasGrp : a @<group> qualifier is present
//   [42:41] group  : group number 0-3
//   [44]    hasL2b : a .<l2b> qualifier is present
//   [45]    l2b    : L2B number 0-1
let PrintMethod = "printMVOperand" in
  def mvoperand : Operand<i64>;

// The "/n<size>[i<tag>][nd<N>][p<prio>]" parameter block, packed into one
// i64. Field layout:
//   [31:0]  size   : transfer word count (n<size>), always present
//   [39:32] tag    : wait tag value, or -1 (0xff) if absent
//   [47:40] nd     : DAR reuse count N, or -1 (0xff) if absent
//   [55:48] prio   : priority 0-3, or -1 (0xff) if absent (default 0)
let PrintMethod = "printMVParams" in
  def mvparams : Operand<i64>;

//===----------------------------------------------------------------------===//
// MV instruction family.
//
// Operand order is always (params, src, dst). `src`/`dst` are packed
// mvoperand immediates whose kind bits encode which memory they are, so
// a single def covers both directions of a unit operation.
//===----------------------------------------------------------------------===//

// mvnop: no-op MV line, no parameters or operands.
let Itinerary = IIC_pseudo in
def MVNOP : MVInst<(outs), (ins), "mvnop">;

// A transfer command: "<stem><params> <src> <dst>". The two memory
// operands are space-separated (no comma) in .vsm.
let Itinerary = IIC_pseudo in
class MVCmd<string stem>
    : MVInst<(outs),
             (ins mvparams:$params, mvoperand:$src, mvoperand:$dst),
             !strconcat(stem, "$params $src $dst")>;

// Individual (point) transfers — mvp.
//   PDM<->DRAM, PDM<->L2BM, DRAM<->L2BM, PDM<->PDM, and the
//   group-parallel forms (an operand without @group addresses all
//   groups at once).
def MVP  : MVCmd<"mvp">;
// Broadcast to all 8 L2BMs — mvb (PDM->all-L2BM, DRAM->all-L2BM).
def MVB  : MVCmd<"mvb">;
// DRAM->both-L2BMs broadcast — mvb2.
def MVB2 : MVCmd<"mvb2">;
// DRAM->all-L2BMs inter-group distribute broadcast — mvb4.
def MVB4 : MVCmd<"mvb4">;
// Distribute — mvd (PDM->L2BM split 8-way, PDM->DRAM split 4-way) and
// the matching collect forms (L2BM->PDM, DRAM->PDM). Same operand kind
// bits; the direction is implicit in which side is PDM.
def MVD  : MVCmd<"mvd">;
// DRAM-indirect is not a separate opcode; it is any transfer whose DRAM
// operand is $di (kind=3) together with an nd<N> parameter.

// Reduce transfers — mvr<op> / mvr2<op> / mvr4<op>. The <op> is an
// rrn_opcode (section 3.5.5). We enumerate the ops assemble3 accepts and
// emit one def per (mode, op) pair.
let Itinerary = IIC_pseudo in
class MVR<string st> : MVCmd<st>;

// mvr<op>: L2BM -> PDM (all 8 L2BMs reduced), L2BM -> DRAM (all L2BMs
// reduced, 4-way split to DRAM).
def MVR_DFADD : MVR<"mvrdfadd">;
def MVR_FFADD : MVR<"mvrffadd">;
def MVR_HFADD : MVR<"mvrhfadd">;
def MVR_DMAX  : MVR<"mvrdmax">;
def MVR_FMAX  : MVR<"mvrfmax">;
def MVR_HMAX  : MVR<"mvrhmax">;
def MVR_DMIN  : MVR<"mvrdmin">;
def MVR_FMIN  : MVR<"mvrfmin">;
def MVR_HMIN  : MVR<"mvrhmin">;
def MVR_LIADD : MVR<"mvrliadd">;
def MVR_IIADD : MVR<"mvriiadd">;
def MVR_SIADD : MVR<"mvrsiadd">;
def MVR_LBAND : MVR<"mvrlband">;
def MVR_IBAND : MVR<"mvriband">;
def MVR_SBAND : MVR<"mvrsband">;
def MVR_LAND  : MVR<"mvrland">;
def MVR_IAND  : MVR<"mvriand">;
def MVR_SAND  : MVR<"mvrsand">;
def MVR_LBOR  : MVR<"mvrlbor">;
def MVR_IBOR  : MVR<"mvribor">;
def MVR_SBOR  : MVR<"mvrsbor">;
def MVR_LOR   : MVR<"mvrlor">;
def MVR_IOR   : MVR<"mvrior">;
def MVR_SOR   : MVR<"mvrsor">;

// mvr2<op>: both-L2BMs -> DRAM / PDM (within-group L2B-direction reduce).
def MVR2_DFADD : MVR<"mvr2dfadd">;
def MVR2_FFADD : MVR<"mvr2ffadd">;
def MVR2_HFADD : MVR<"mvr2hfadd">;
def MVR2_DMAX  : MVR<"mvr2dmax">;
def MVR2_FMAX  : MVR<"mvr2fmax">;
def MVR2_HMAX  : MVR<"mvr2hmax">;
def MVR2_DMIN  : MVR<"mvr2dmin">;
def MVR2_FMIN  : MVR<"mvr2fmin">;
def MVR2_HMIN  : MVR<"mvr2hmin">;
def MVR2_LIADD : MVR<"mvr2liadd">;
def MVR2_IIADD : MVR<"mvr2iiadd">;
def MVR2_SIADD : MVR<"mvr2siadd">;
def MVR2_LBAND : MVR<"mvr2lband">;
def MVR2_IBAND : MVR<"mvr2iband">;
def MVR2_SBAND : MVR<"mvr2sband">;
def MVR2_LAND  : MVR<"mvr2land">;
def MVR2_IAND  : MVR<"mvr2iand">;
def MVR2_SAND  : MVR<"mvr2sand">;
def MVR2_LBOR  : MVR<"mvr2lbor">;
def MVR2_IBOR  : MVR<"mvr2ibor">;
def MVR2_SBOR  : MVR<"mvr2sbor">;
def MVR2_LOR   : MVR<"mvr2lor">;
def MVR2_IOR   : MVR<"mvr2ior">;
def MVR2_SOR   : MVR<"mvr2sor">;

// mvr4<op>: all-L2BMs -> DRAM inter-group collect reduce.
def MVR4_DFADD : MVR<"mvr4dfadd">;
def MVR4_FFADD : MVR<"mvr4ffadd">;
def MVR4_HFADD : MVR<"mvr4hfadd">;
def MVR4_DMAX  : MVR<"mvr4dmax">;
def MVR4_FMAX  : MVR<"mvr4fmax">;
def MVR4_HMAX  : MVR<"mvr4hmax">;
def MVR4_DMIN  : MVR<"mvr4dmin">;
def MVR4_FMIN  : MVR<"mvr4fmin">;
def MVR4_HMIN  : MVR<"mvr4hmin">;
def MVR4_LIADD : MVR<"mvr4liadd">;
def MVR4_IIADD : MVR<"mvr4iiadd">;
def MVR4_SIADD : MVR<"mvr4siadd">;
def MVR4_LBAND : MVR<"mvr4lband">;
def MVR4_IBAND : MVR<"mvr4iband">;
def MVR4_SBAND : MVR<"mvr4sband">;
def MVR4_LAND  : MVR<"mvr4land">;
def MVR4_IAND  : MVR<"mvr4iand">;
def MVR4_SAND  : MVR<"mvr4sand">;
def MVR4_LBOR  : MVR<"mvr4lbor">;
def MVR4_IBOR  : MVR<"mvr4ibor">;
def MVR4_SBOR  : MVR<"mvr4sbor">;
def MVR4_LOR   : MVR<"mvr4lor">;
def MVR4_IOR   : MVR<"mvr4ior">;
def MVR4_SOR   : MVR<"mvr4sor">;
