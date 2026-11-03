//===- MNCoreInstrL1BM.td - MN-Core2 L1BM PE expressions ----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// L1BM instruction expressions (section 3.6.8, table 3.8) move data
// between an L1BM and the PEs beneath it. They are PE expressions and
// share a step with other groups via ';'. The parallel-execution groups
// are:
//
//   l1bm             any non-turnaround L1BM expression
//   l1bm-turnaround  an L1BM expression whose L1BM operand is $lbi /
//                    $llbi (the turnaround register)
//
// Whether a form is turnaround is a property of the packed lb2operand
// immediate, so each expression has two defs: the plain one reserving
// IIC_l1bm and a _T twin reserving IIC_l1bmt. They print identically.
//
// Operand summary (section 3.6.8):
//   l1bmp  $(l|ll)b(<addr>|i)  <pe_dst>...     PE broadcast 1/2-lw
//   l1bmm  $(l|ll)b(<addr>|i)  <pe_dst>...     16x1 MAB broadcast 1/2-lw
//   l1bmm@<mabadr> <pe_src> $(l|ll)b(<addr>|i) 16x1 individual 1/2-lw
//   l1bmr<rrn>[r]    <pe_src> $(l|ll)b(<addr>|i) 16x1 reduce
//   l1bmm4 $(l|ll)b(<addr>|i)  <pe_dst>...     4x4 MAB broadcast 1/2-lw
//   l1bmm4@<mabadr>  <pe_src> $(l|ll)b(<addr>|i) 4x4 individual
//   l1bmr4<rrn>[r]   <pe_src> $(l|ll)b(<addr>|i) 4x4 reduce
//   l1bmd[<mabdiff>] $(l|ll)b(<addr>|i) <pe_dst>...  distribute
//   l1bmd[<mabdiff>] <pe_src> $(l|ll)b(<addr>|i)     collect
//
// <mabdiff> is a mandatory-sign offset "+<0-15>" / "-<0-15>" inside the
// mnemonic; <mabadr> is 0-15 for l1bmm@ and 0-3 for l1bmm4@.
//
// Multiple PE destinations (<dst_0> [<dst_1>..]) are not modelled yet:
// each def carries exactly one PE-side operand. The variadic form lands
// with the operand-shapes work in M4.
//===----------------------------------------------------------------------===//

include "MNCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

// PE-side operand ("PE operand" in the manual) packed into one i64.
// Covers the memory/register names an instruction may take on its PE
// side (section 3.6.1) in BOTH stream modes:
//
//   autostride  $[(l|ll)](r|s|m|n)<addr>[v[<adri>]][j<madpe>]
//               $[(l|ll)]m t[<addr>[v[<adri>]]]      (T-indirect, LM0)
//   flat        $[(l|ll)](r|s|m|n)[<a0>,<a1>,<a2>,<a3>][j<madpe>]
//               $[(l|ll)]m t[<flat_addrs>]           (T-indirect flat)
//   $llt        T register (always 2-lw)
//   $mauf / $aluf / $lbf / $mreadf  forwarding latches
//   $nowrite    dummy output (forwarding sink)
//
// Autostride modifiers (bit 26 flat=0):
//   v[<adri>]  per-cycle stride; 'v' alone strides by one access unit,
//              v<adri> strides by <adri> words
//   j<madpe>   MAB-address-modify: PEs numbered <= madpe add one
//              access-width unit
//   t<...>     T-register indirect (LM0 only); the inner addr/v may be
//              omitted meaning all-zero base
//
// Packed i64:
//   [3:0]   kind   : 0=r 1=s 2=m 3=n 4=t 5=mauf 6=aluf 7=lbf 8=mreadf
//                    9=nowrite
//   [5:4]   width  : 0=word 1=lw 2=2lw (moved off the addr field so the
//                    flat layout below can reuse [19:4])
//   [6]     eflag  : 'e' precision-extend suffix
//   [7]     neg    : leading '-' sign inversion
//   [8]     rd     : trailing 'r' precision-reduce suffix
//   [9]     flat   : flat mode — the rest is a 4-address list
//   autostride (flat=0):
//     [25:10] addr   : word address (16b)
//     [26]    hasV   : 'v' marker present
//     [27]    hasAdri: explicit <adri> after v
//     [43:28] adri   : stride in words (16b)
//     [44]    hasJ   : j<madpe> present
//     [47:45] madpe  : MAB-address-modify PE bound 0-3
//     [48]    tInd   : T-register indirect (t prefix)
//   flat (flat=1):
//     [57:10] four 12-bit word addresses a0..a3 (a0=[21:10], a1=[33:22],
//             a2=[45:34], a3=[57:46])
//     [58]    hasJ / [61:59] madpe
//     [62]    tInd   : t[<flat>] form
let PrintMethod = "printPEOperand" in
  def peoperand : Operand<i64>;

// L1BM operand as used by L1BM expressions (section 3.6.1.5):
//   $(l|ll)b(<addr>|i)
// Packed i64:
//   [0]     turn : 'i' turnaround register instead of an address
//   [1]     ll   : 2-longword access ($llb)
//   [17:2]  addr : longword address (8Ki wrap, ignored when turn)
let PrintMethod = "printLB2Operand" in
  def lb2operand : Operand<i64>;

// "@<mabadr>" MAB qualifier carried on the opcode (l1bmm@ / l1bmm4@).
// Packed i64: [4] present, [3:0] mabadr.
let PrintMethod = "printMABSetOperand" in
  def mabsetoperand : Operand<i64>;

// "[<mabdiff>]" MAB-rotation qualifier inside the l1bmd mnemonic.
// Packed i64: [5] present, [4] minus, [3:0] value.
let PrintMethod = "printMABDiffOperand" in
  def mabdiffoperand : Operand<i64>;

// Mask-register write destination (3.6.1.13): "$omr<addr>". Packed as
// addr | (1<<32); bit 32 is a marker the PE-operand decoders use to skip it.
let PrintMethod = "printOMROperand" in
  def omroperand : Operand<i64>;

// Multi-line write-mask directive (3.6.2.1): "mask[l|ll][r][s][t][m][n][k] <addr>".
// Packed: [0] ll, [6:1] enables r,s,t,m,n,k, [13:7] entry address.
let PrintMethod = "printMaskDirective" in
  def maskdiroperand : Operand<i64>;

// The multi-line directive itself (3.6.2.1). It is a control statement,
// not a PE expression: it changes the mask applied to following writes
// until the next directive. Pseudo + non-PE, so it stays on its own line.
def MASK_DIR : PseudoMN<(outs), (ins maskdiroperand:$spec), "mask$spec">;

// Mask application (section 3.6.2), the shared "/(ll)?(<pattern>|$imr<adr>)(t|p)".
// Used two ways, distinguished by where the operand sits in the asm string:
//   * zero-flush:  <opcode>/(...)  — printed right after the mnemonic;
//   * write mask:  <dst>/(...)     — printed right after the destination.
// Both are always the LAST operand in the `ins` list (stable MI index); the
// class's asm string decides where it prints. Packed i64:
//   [0]     present
//   [1]     ll     : 2-longword mask word length
//   [2]     var    : variable entry ($imr) instead of a fixed 4-bit pattern
//   [15:3]  value  : fixed-entry 4-bit pattern, or variable-entry address
//   [17:16] tp     : 0 none, 1 't' (truncate), 2 'p' (pad)
//   [18]    zero   : 1 = zero-flush application (3.6.2.2), 0 = write mask
let PrintMethod = "printMaskOperand" in
  def maskoperand : Operand<i64>;

//===----------------------------------------------------------------------===//
// L1BM instruction expressions.
//
// Operand order is the .vsm "source then destination" order. Each form
// has a plain def (l1bm group) and a *_T def (l1bm-turnaround group) —
// identical text, different reserved unit.
//===----------------------------------------------------------------------===//

// Direction tags (TSFlags [9:8]) drive the §3.6.4 co-issue check and the
// §3.6.3 hazard latencies: 1 = L1B->PE (the expression READS L1BM,
// writes PE), 2 = PE->L1B (writes L1BM), 3 = turnaround ($lbi). For a
// turnaround form the tag records the direction of the $lbi access:
// a read-side turnaround (l1bmm $lbi -> PE) is L1B->PE, a write-side
// turnaround (PE -> $lbi) is PE->L1B. The plain/_T twin of one mnemonic
// shares the direction; only the reserved unit differs.
class L1BMC<dag ins, string asm, int dir>
    : PEExpr<(outs), ins, asm> {
  let Itinerary = IIC_l1bm;
  let TSFlags{9-8} = dir;
}
class L1BMCT<dag ins, string asm, int dir>
    : PEExpr<(outs), ins, asm> {
  let Itinerary = IIC_l1bmt;
  let TSFlags{9-8} = dir;
  // Bit 12 marks this as a turnaround ($lbi) form — the packetizer uses
  // it to pair a turnaround read with the PE->L1B write that feeds it.
  let TSFlags{12} = 1;
}

// ---- L1BM -> PE broadcast -------------------------------------------------

// l1bmp $(l|ll)b(<a>|i) <pe>: 1/2-longword PE broadcast (3.6.8.6/.7).
def L1BMP   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                    "l1bmp $src $dst", 1>;
def L1BMP_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmp $src $dst", 1>;
// Multi-destination broadcast (3.6.1/3.6.8.6): the same data is written
// to every listed PE operand. TSFlags[17] + [20:18] carry the write count.
def L1BMP2  : L1BMC<(ins lb2operand:$src, peoperand:$d0, peoperand:$d1),
                    "l1bmp $src $d0 $d1", 1> {
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}

// l1bmm $(l|ll)b(<a>|i) <pe>: 1/2-longword 16x1 MAB broadcast (.8/.9).
def L1BMM   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                    "l1bmm $src $dst", 1>;
def L1BMM2  : L1BMC<(ins lb2operand:$src, peoperand:$d0, peoperand:$d1),
                    "l1bmm $src $d0 $d1", 1> {
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
def L1BMM_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmm $src $dst", 1>;

// l1bmm4 $(l|ll)b(<a>|i) <pe>: 1/2-longword 4x4 MAB broadcast (.14/.15).
def L1BMM4   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmm4 $src $dst", 1>;
def L1BMM4_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                      "l1bmm4 $src $dst", 1>;

// l1bmd[<mabdiff>] $(l|ll)b(<a>|i) <pe>: distribute (3.6.8.20).
def L1BMD   : L1BMC<(ins mabdiffoperand:$diff, lb2operand:$src,
                     peoperand:$dst),
                    "l1bmd$diff $src $dst", 1>;
def L1BMD_T : L1BMCT<(ins mabdiffoperand:$diff, lb2operand:$src,
                      peoperand:$dst),
                     "l1bmd$diff $src $dst", 1>;

// ---- PE -> L1BM collect ----------------------------------------------------

// l1bmm@<mabadr> <pe> $(l|ll)b(<a>|i): 16x1 individual (.10/.11).
def L1BMM_AT   : L1BMC<(ins mabsetoperand:$at, peoperand:$src,
                        lb2operand:$dst),
                       "l1bmm$at $src $dst", 2>;
def L1BMM_AT_T : L1BMCT<(ins mabsetoperand:$at, peoperand:$src,
                         lb2operand:$dst),
                        "l1bmm$at $src $dst", 2>;

// l1bmm4@<mabadr> <pe> $(l|ll)b(<a>|i): 4x4 individual (.16/.17).
def L1BMM4_AT   : L1BMC<(ins mabsetoperand:$at, peoperand:$src,
                         lb2operand:$dst),
                        "l1bmm4$at $src $dst", 2>;
def L1BMM4_AT_T : L1BMCT<(ins mabsetoperand:$at, peoperand:$src,
                          lb2operand:$dst),
                         "l1bmm4$at $src $dst", 2>;

// l1bmd[<mabdiff>] <pe> $(l|ll)b(<a>|i): collect (3.6.8.21).
def L1BMD_C   : L1BMC<(ins mabdiffoperand:$diff, peoperand:$src,
                       lb2operand:$dst),
                      "l1bmd$diff $src $dst", 2>;
def L1BMD_C_T : L1BMCT<(ins mabdiffoperand:$diff, peoperand:$src,
                        lb2operand:$dst),
                       "l1bmd$diff $src $dst", 2>;

// ---- Reduce forms ----------------------------------------------------------
//
// l1bmr<rrn> / l1bmr4<rrn>: <pe_src> $(l|ll)b(<a>|i>. The rrn opcodes
// are the section 3.5.5 set; the trailing-r forms (ffaddr/fmaxr/fminr)
// request the single->half result-precision reduction of section
// 3.6.8.5 (l1bmrhfadd is an alias of l1bmrffaddr, so hfadd appears too).

class L1BMR<string op>
    : L1BMC<(ins peoperand:$src, lb2operand:$dst),
            !strconcat("l1bmr", op, " $src $dst"), 2>;
class L1BMRT<string op>
    : L1BMCT<(ins peoperand:$src, lb2operand:$dst),
             !strconcat("l1bmr", op, " $src $dst"), 2>;
class L1BMR4<string op>
    : L1BMC<(ins peoperand:$src, lb2operand:$dst),
            !strconcat("l1bmr4", op, " $src $dst"), 2>;
class L1BMR4T<string op>
    : L1BMCT<(ins peoperand:$src, lb2operand:$dst),
             !strconcat("l1bmr4", op, " $src $dst"), 2>;

foreach p = [["dfadd","DFADD"], ["ffadd","FFADD"], ["hfadd","HFADD"],
             ["dmax","DMAX"], ["fmax","FMAX"], ["hmax","HMAX"],
             ["dmin","DMIN"], ["fmin","FMIN"], ["hmin","HMIN"],
             ["liadd","LIADD"], ["iiadd","IIADD"], ["siadd","SIADD"],
             ["lband","LBAND"], ["iband","IBAND"], ["sband","SBAND"],
             ["land","LAND"], ["iand","IAND"], ["sand","SAND"],
             ["lbor","LBOR"], ["ibor","IBOR"], ["sbor","SBOR"],
             ["lor","LOR"], ["ior","IOR"], ["sor","SOR"],
             ["ffaddr","FFADDR"], ["fmaxr","FMAXR"], ["fminr","FMINR"]] in {
  def "L1BMR_"#p[1]     : L1BMR<p[0]>;
  def "L1BMR_"#p[1]#"_T"  : L1BMRT<p[0]>;
  def "L1BMR4_"#p[1]    : L1BMR4<p[0]>;
  def "L1BMR4_"#p[1]#"_T" : L1BMR4T<p[0]>;
}

//===----------------------------------------------------------------------===//
// Masked L1BM->PE broadcast forms (3.6.2). The transfer result is a
// maskable output: the zero-flush mask prints after the mnemonic, the
// write mask after the (single) destination. PE->L1B collect forms are
// not maskable outputs and have no masked twin.
//===----------------------------------------------------------------------===//
class L1BMC_M<string opc, dag extra>
    : PEExpr<(outs),
             !con(extra, (ins peoperand:$dst, maskoperand:$zmask,
                          maskoperand:$wmask)),
             !strconcat(opc, "$zmask $src $dst$wmask")> {
  let Itinerary = IIC_l1bm;
  let TSFlags{9-8} = 1;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}
def L1BMP_M : L1BMC_M<"l1bmp", (ins lb2operand:$src)>;
def L1BMM_M : L1BMC_M<"l1bmm", (ins lb2operand:$src)>;
def L1BMM4_M : L1BMC_M<"l1bmm4", (ins lb2operand:$src)>;
def L1BMD_M : L1BMC_M<"l1bmd", (ins mabdiffoperand:$diff, lb2operand:$src)>;

// Three-destination L1BM broadcasts (3.6.1).
def L1BMP3 : L1BMC<(ins lb2operand:$src, peoperand:$d0, peoperand:$d1,
                        peoperand:$d2),
                  "l1bmp $src $d0 $d1 $d2", 1> {
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
def L1BMM3 : L1BMC<(ins lb2operand:$src, peoperand:$d0, peoperand:$d1,
                        peoperand:$d2),
                  "l1bmm $src $d0 $d1 $d2", 1> {
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
