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

//===----------------------------------------------------------------------===//
// L1BM instruction expressions.
//
// Operand order is the .vsm "source then destination" order. Each form
// has a plain def (l1bm group) and a *_T def (l1bm-turnaround group) —
// identical text, different reserved unit.
//===----------------------------------------------------------------------===//

class L1BMC<dag ins, string asm> : PEExpr<(outs), ins, asm> {
  let Itinerary = IIC_l1bm;
}
class L1BMCT<dag ins, string asm> : PEExpr<(outs), ins, asm> {
  let Itinerary = IIC_l1bmt;
}

// ---- L1BM -> PE broadcast -------------------------------------------------

// l1bmp $(l|ll)b(<a>|i) <pe>: 1/2-longword PE broadcast (3.6.8.6/.7).
def L1BMP   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                    "l1bmp $src $dst">;
def L1BMP_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmp $src $dst">;

// l1bmm $(l|ll)b(<a>|i) <pe>: 1/2-longword 16x1 MAB broadcast (.8/.9).
def L1BMM   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                    "l1bmm $src $dst">;
def L1BMM_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmm $src $dst">;

// l1bmm4 $(l|ll)b(<a>|i) <pe>: 1/2-longword 4x4 MAB broadcast (.14/.15).
def L1BMM4   : L1BMC<(ins lb2operand:$src, peoperand:$dst),
                     "l1bmm4 $src $dst">;
def L1BMM4_T : L1BMCT<(ins lb2operand:$src, peoperand:$dst),
                      "l1bmm4 $src $dst">;

// l1bmd[<mabdiff>] $(l|ll)b(<a>|i) <pe>: distribute (3.6.8.20).
def L1BMD   : L1BMC<(ins mabdiffoperand:$diff, lb2operand:$src,
                     peoperand:$dst),
                    "l1bmd$diff $src $dst">;
def L1BMD_T : L1BMCT<(ins mabdiffoperand:$diff, lb2operand:$src,
                      peoperand:$dst),
                     "l1bmd$diff $src $dst">;

// ---- PE -> L1BM collect ----------------------------------------------------

// l1bmm@<mabadr> <pe> $(l|ll)b(<a>|i): 16x1 individual (.10/.11).
def L1BMM_AT   : L1BMC<(ins mabsetoperand:$at, peoperand:$src,
                        lb2operand:$dst),
                       "l1bmm$at $src $dst">;
def L1BMM_AT_T : L1BMCT<(ins mabsetoperand:$at, peoperand:$src,
                         lb2operand:$dst),
                        "l1bmm$at $src $dst">;

// l1bmm4@<mabadr> <pe> $(l|ll)b(<a>|i): 4x4 individual (.16/.17).
def L1BMM4_AT   : L1BMC<(ins mabsetoperand:$at, peoperand:$src,
                         lb2operand:$dst),
                        "l1bmm4$at $src $dst">;
def L1BMM4_AT_T : L1BMCT<(ins mabsetoperand:$at, peoperand:$src,
                          lb2operand:$dst),
                         "l1bmm4$at $src $dst">;

// l1bmd[<mabdiff>] <pe> $(l|ll)b(<a>|i): collect (3.6.8.21).
def L1BMD_C   : L1BMC<(ins mabdiffoperand:$diff, peoperand:$src,
                       lb2operand:$dst),
                      "l1bmd$diff $src $dst">;
def L1BMD_C_T : L1BMCT<(ins mabdiffoperand:$diff, peoperand:$src,
                        lb2operand:$dst),
                       "l1bmd$diff $src $dst">;

// ---- Reduce forms ----------------------------------------------------------
//
// l1bmr<rrn> / l1bmr4<rrn>: <pe_src> $(l|ll)b(<a>|i>. The rrn opcodes
// are the section 3.5.5 set; the trailing-r forms (ffaddr/fmaxr/fminr)
// request the single->half result-precision reduction of section
// 3.6.8.5 (l1bmrhfadd is an alias of l1bmrffaddr, so hfadd appears too).

class L1BMR<string op>
    : L1BMC<(ins peoperand:$src, lb2operand:$dst),
            !strconcat("l1bmr", op, " $src $dst")>;
class L1BMRT<string op>
    : L1BMCT<(ins peoperand:$src, lb2operand:$dst),
             !strconcat("l1bmr", op, " $src $dst")>;
class L1BMR4<string op>
    : L1BMC<(ins peoperand:$src, lb2operand:$dst),
            !strconcat("l1bmr4", op, " $src $dst")>;
class L1BMR4T<string op>
    : L1BMCT<(ins peoperand:$src, lb2operand:$dst),
             !strconcat("l1bmr4", op, " $src $dst")>;

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
