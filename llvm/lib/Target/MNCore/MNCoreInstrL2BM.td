//===- MNCoreInstrL2BM.td - MN-Core2 L2BM PE expressions ----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// L2BM instruction expressions (section 3.6.7) move data between the L2
// buffer memory and the L1 buffer memories beneath it, plus the L1B
// internal multicast and the DAR write path. They are PE expressions —
// they share a step with other groups via ';' — occupying the "l2bm"
// parallel-execution group (l2bmdarw has its own "l2bmdarw" group and
// may co-issue with any other l2bm expression).
//
// Operands:
//   $lc<addr>            L2BM longword address (no @l1bset needed for
//                        most forms; l2bmdars adds "@.<l2b>")
//   $lb<addr>            L1BM longword address
//   $dar<addr>           DAR write-start entry (l2bmdars only)
//   @<l1bset>            optional L1B-subset qualifier on the opcode
//   <l1badr>             a single L1B index 0-7 (l2bm@ / l2bmi use it)
//===----------------------------------------------------------------------===//

include "MNCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

// L2BM operand. For l2bmdars the @.<l2b> qualifier selects which L2B's
// data goes to the DAR; for all other L2BM expressions the plain
// "$lc<addr>" form is used. Packed i64:
//   [35:4]  addr   : longword address (32Ki wrap)
//   [40]    hasL2b : "@.<l2b>" qualifier present (l2bmdars only)
//   [41]    l2b    : L2B number 0-1
let PrintMethod = "printLCOperand" in
  def lcoperand : Operand<i64>;

// L1BM operand, plain "$lb<addr>" (no @-qualifiers on L2BM expressions).
// Packed i64: [35:4] addr (8Ki wrap).
let PrintMethod = "printLBOperand" in
  def lboperand : Operand<i64>;

// DAR operand "$dar<addr>" — write-start entry, 1024 entries wrap.
let PrintMethod = "printDAROperand" in
  def daroperand : Operand<i64>;

// L1B-subset qualifier "@<l1bset>" carried on the opcode. The three
// textual forms (section 3.6.7.1) collapse to (l1badr, immode):
//   <l1badr>/<immode>  ->  mask form
//   <l1badr>           ->  immode=0 shorthand
//   [<l1blist>]        ->  a list, always resolvable to (l1badr,immode)
//                        per table 3.7
// We keep both an (l1badr,immode) pair and a raw-list bit so the printer
// can echo whichever form was written, though only the mask form is
// canonical. Packed i64:
//   [7]    present : qualifier was written at all
//   [6]    isList  : emit "[list]" instead of "adr/immode"
//   [3:0]  l1badr
//   [10:8] immode  (raw, for the mask form)
//   [31:16] listmask : 8-bit L1B mask, only when isList
let PrintMethod = "printL1BSetOperand" in
  def l1bsetoperand : Operand<i64>;

// A bare L1B index operand for l2bm@<l1badr> (0-7) and the l2bmi
// multicast selector (which reuses the <l1badr>/<immode> grammar via
// l1bsetoperand instead).
def l1badr : Operand<i64>;

//===----------------------------------------------------------------------===//
// L2BM instruction expressions.
//
// All occupy the "l2bm" group except l2bmdarw which has its own
// "l2bmdarw" group. Operand order is the .vsm "source then destination"
// order; the @<l1bset> qualifier is a leading operand on the opcode.
//
// Direction tags (TSFlags [11:10]) drive the §3.6.3 hazard latencies:
//   1 = L1B->L2B   2 = L2B->L1B   3 = L1B-internal multicast / DAR.
// They are not a co-issue constraint — l2bm is a single group already —
// they only tell the dependency builder which write->read distance a
// following transfer must respect.
//===----------------------------------------------------------------------===//
class L2BMExpr<dag outs, dag ins, string asmstr, int dir>
    : PEExpr<outs, ins, asmstr> {
  let TSFlags{11-10} = dir;
}

// Broadcast L2BM -> L1BM: l2bmb[@<l1bset>] $lc<addr> $lb<addr>
let Itinerary = IIC_l2bm in
def L2BMB : L2BMExpr<(outs),
                   (ins l1bsetoperand:$set, lcoperand:$src, lboperand:$dst),
                   "l2bmb$set $src $dst", 2>;

// Distribute-broadcast L2BM -> L1BM: l2bmb2[@<l1bset>] $lc $lb
let Itinerary = IIC_l2bm in
def L2BMB2 : L2BMExpr<(outs),
                    (ins l1bsetoperand:$set, lcoperand:$src, lboperand:$dst),
                    "l2bmb2$set $src $dst", 2>;

// Distribute L2BM -> L1BM: l2bmd[@<l1bset>] $lc $lb
// (l2bmd is overloaded — see L2BMD_L1B below for the L1BM->L2BM collect
// direction; the assembler distinguishes them by operand order.)
let Itinerary = IIC_l2bm in
def L2BMD_D : L2BMExpr<(outs),
                     (ins l1bsetoperand:$set, lcoperand:$src, lboperand:$dst),
                     "l2bmd$set $src $dst", 2>;

// Point transfer L1BM -> L2BM: l2bm@<l1badr> $lb<addr> $lc<addr>
let Itinerary = IIC_l2bm in
def L2BM_PT : L2BMExpr<(outs),
                     (ins l1badr:$l1b, lboperand:$src, lcoperand:$dst),
                     "l2bm@$l1b $src $dst", 1>;

// Collect L1BM -> L2BM: l2bmd $lb<addr> $lc<addr>
let Itinerary = IIC_l2bm in
def L2BMD_C : L2BMExpr<(outs),
                     (ins lboperand:$src, lcoperand:$dst),
                     "l2bmd $src $dst", 1>;

// Internal multicast L1BM -> L1BM: l2bmi@<l1bset> $lb<addr0> $lb<addr1>
let Itinerary = IIC_l2bm in
def L2BMI : L2BMExpr<(outs),
                   (ins l1bsetoperand:$set, lboperand:$src, lboperand:$dst),
                   "l2bmi$set $src $dst", 3>;

// DAR stage: l2bmdars $lc<addr>@.<l2b> $dar<addr>
let Itinerary = IIC_l2bm in
def L2BMDARS : L2BMExpr<(outs),
                      (ins lcoperand:$src, daroperand:$dst),
                      "l2bmdars $src $dst", 3>;

// DAR write drain: l2bmdarw — own group, co-issues with any l2bm.
let Itinerary = IIC_l2bmdarw in
def L2BMDARW : PEExpr<(outs), (ins), "l2bmdarw">;

//===----------------------------------------------------------------------===//
// Reduce forms l2bmr<rrn>[@<l1bset>] / l2bmr2<rrn>.
//
// l2bmr<rrn> reduces the L1B-set into L2BM; l2bmr2<rrn> does a
// collect-reduce across all L1Bs. Both take ($lb src, $lc dst).
//===----------------------------------------------------------------------===//

let Itinerary = IIC_l2bm in
class L2BMR<string op>
    : L2BMExpr<(outs),
             (ins l1bsetoperand:$set, lboperand:$src, lcoperand:$dst),
             !strconcat("l2bmr", op, "$set $src $dst"), 1>;

let Itinerary = IIC_l2bm in
class L2BMR2<string op>
    : L2BMExpr<(outs),
             (ins lboperand:$src, lcoperand:$dst),
             !strconcat("l2bmr2", op, " $src $dst"), 1>;

// rrn opcodes (section 3.5.5), same set as the MV reduce ops.
foreach p = [["dfadd","L2BMR_DFADD","L2BMR2_DFADD"],
             ["ffadd","L2BMR_FFADD","L2BMR2_FFADD"],
             ["hfadd","L2BMR_HFADD","L2BMR2_HFADD"],
             ["dmax","L2BMR_DMAX","L2BMR2_DMAX"],
             ["fmax","L2BMR_FMAX","L2BMR2_FMAX"],
             ["hmax","L2BMR_HMAX","L2BMR2_HMAX"],
             ["dmin","L2BMR_DMIN","L2BMR2_DMIN"],
             ["fmin","L2BMR_FMIN","L2BMR2_FMIN"],
             ["hmin","L2BMR_HMIN","L2BMR2_HMIN"],
             ["liadd","L2BMR_LIADD","L2BMR2_LIADD"],
             ["iiadd","L2BMR_IIADD","L2BMR2_IIADD"],
             ["siadd","L2BMR_SIADD","L2BMR2_SIADD"],
             ["lband","L2BMR_LBAND","L2BMR2_LBAND"],
             ["iband","L2BMR_IBAND","L2BMR2_IBAND"],
             ["sband","L2BMR_SBAND","L2BMR2_SBAND"],
             ["land","L2BMR_LAND","L2BMR2_LAND"],
             ["iand","L2BMR_IAND","L2BMR2_IAND"],
             ["sand","L2BMR_SAND","L2BMR2_SAND"],
             ["lbor","L2BMR_LBOR","L2BMR2_LBOR"],
             ["ibor","L2BMR_IBOR","L2BMR2_IBOR"],
             ["sbor","L2BMR_SBOR","L2BMR2_SBOR"],
             ["lor","L2BMR_LOR","L2BMR2_LOR"],
             ["ior","L2BMR_IOR","L2BMR2_IOR"],
             ["sor","L2BMR_SOR","L2BMR2_SOR"]] in {
  def p[1] : L2BMR<p[0]>;
  def p[2] : L2BMR2<p[0]>;
}
