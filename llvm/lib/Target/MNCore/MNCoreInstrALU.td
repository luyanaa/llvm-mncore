//===- MNCoreInstrALU.td - MN-Core2 ALU opcode catalogue -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The rest of the ALU opcode set (section 3.6.12, table 3.9) beyond the
// eight scalar ops in MNCoreInstrInfo.td (ladd..lnot), which are the M1
// isel targets and use GRF register operands.
//
// Everything here is defined at the assembly level with packed peoperand
// immediates — the same operand the MAU / L1BM expressions use — so the
// hazard pass can decode the reads and writes. These are not yet isel
// targets; they exist so hand-written .mir lowers to assemble3-legal .vsm.
//
// Operand order is the .vsm input-then-output order: sources first, the
// destination last. The precision letter is part of the mnemonic. Which
// precisions an opcode accepts comes from table 3.10:
//   int   -> l, i, s       float -> d, f, h
//   both  -> d, f, h, l, i, s
//   bfn   -> d, f, g, h    bfe -> h      untyped -> none
// `u`-prefixed (unsigned) variants are separate defs, only where table 3.9
// marks the unsigned option as accepted.
//
// Included after MNCoreInstrL1BM.td, which defines peoperand.
//
//===----------------------------------------------------------------------===//

// One source, one destination.
class ALU1P<string opc>
    : PEExpr<(outs), (ins peoperand:$src, peoperand:$dst),
             !strconcat(opc, " $src $dst")> {
  let Itinerary = IIC_alupe;
}
// Two sources, one destination.
class ALU2P<string opc>
    : PEExpr<(outs), (ins peoperand:$sx, peoperand:$sy, peoperand:$dst),
             !strconcat(opc, " $sx $sy $dst")> {
  let Itinerary = IIC_alupe;
}

// imm / immu with a PE-operand destination (3.6.12.3, 3.2.2). This is the
// assembly-level spelling: the literal prints in its typed form and the
// destination as a peoperand (`$lr0v`). IMM in MNCoreInstrInfo.td is the
// isel-facing GRF-register variant. TSFlags[16] excludes the literal from
// the PE-operand decode and bars co-issue with any LM0 access.
class IMMPExpr<string opc>
    : PEExpr<(outs), (ins immlitx:$imm, peoperand:$dst),
             !strconcat(opc, " $imm $dst")> {
  let Itinerary = IIC_const;
  let TSFlags{16} = 1;
}
def IMM_P  : IMMPExpr<"imm">;
def IMMU_P : IMMPExpr<"immu">;

// Multi-destination ALU form (3.6.1: <src> <dst_0> [<dst_1>..]). TSFlags
// [20:18] holds the destination count so the PE-access decoder knows the
// trailing operands are writes.
class ALU1PMD<string opc, int ndst>
    : PEExpr<(outs),
             (ins peoperand:$src, peoperand:$d0, peoperand:$d1),
             !strconcat(opc, " $src $d0 $d1")> {
  let Itinerary = IIC_alupe;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = ndst;
}
def LPASSA2 : ALU1PMD<"lpassa", 2>;

// --------------------------------------------------------------------------
// untyped — no precision letter (table 3.10 "none").
// msl/msr: PE-to-PE circular shift (3.6.12.4).
// --------------------------------------------------------------------------
def MSL : ALU1P<"msl">;
def MSR : ALU1P<"msr">;

// --------------------------------------------------------------------------
// int (l/i/s), one input.
//
// `not` (3.6.12.7) is bitwise complement; `lnot` (3.6.12.8) is logical not.
// The long-word spellings collide: both want `lnot`, and the manual's own
// example (3.6.12.7) uses `lnot` for bitwise complement while 3.6.12.8 uses
// it for logical not. The logical-not long-word form is the M1 LNOT def, so
// bitwise-not is emitted for i/s only (`inot`/`snot`) and logical-not for
// i/s as `ilnot`/`slnot`.
// --------------------------------------------------------------------------
def INOT  : ALU1P<"inot">;
def SNOT  : ALU1P<"snot">;
def ILNOT : ALU1P<"ilnot">;
def SLNOT : ALU1P<"slnot">;

// inc/dec (3.6.12.6) for i/s plus the `u` forms. Long-word signed inc/dec
// are LINC/LDEC in MNCoreInstrInfo.td.
foreach p = ["i", "s"] in {
  def INC_#p : ALU1P<!strconcat(p, "inc")>;
  def DEC_#p : ALU1P<!strconcat(p, "dec")>;
}
foreach p = ["l", "i", "s"] in {
  def UINC_#p : ALU1P<!strconcat("u", p, "inc")>;
  def UDEC_#p : ALU1P<!strconcat("u", p, "dec")>;
}

// --------------------------------------------------------------------------
// float (d/f/h), one input. rsqrt (3.6.12.9), floor (3.6.12.10).
// --------------------------------------------------------------------------
foreach p = ["d", "f", "h"] in {
  def RSQRT_#p : ALU1P<!strconcat(p, "rsqrt")>;
  def FLOOR_#p : ALU1P<!strconcat(p, "floor")>;
}

// ftoi (3.6.12.11): float -> int, accepts the `u` option.
foreach p = ["d", "f", "h"] in {
  def FTOI_#p  : ALU1P<!strconcat(p, "ftoi")>;
  def UFTOI_#p : ALU1P<!strconcat("u", p, "ftoi")>;
}

// --------------------------------------------------------------------------
// Masked twins (section 3.6.2). An ALU expression is maskable: the write
// mask prints after the destination, the zero-flush mask after the
// mnemonic. Both mask operands are appended after the ordinary operands so
// the PE indices the decoder uses are unchanged; a clear present bit
// prints nothing. TSFlags 14/15 mark the trailing immediates as masks.
// --------------------------------------------------------------------------
class ALU2PM<string opc>
    : PEExpr<(outs),
             (ins peoperand:$sx, peoperand:$sy, peoperand:$dst,
                  maskoperand:$zmask, maskoperand:$wmask),
             !strconcat(opc, "$zmask $sx $sy $dst$wmask")> {
  let Itinerary = IIC_alupe;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}

def FMIN_M    : ALU2PM<"fmin">;
def FMAX_M    : ALU2PM<"fmax">;
def IMIN_M    : ALU2PM<"imin">;
def DADD_M    : ALU2PM<"dadd">;
def FMUL_M    : ALU2PM<"fmul">;

// --------------------------------------------------------------------------
// Block float (3.6.12.12). bfn: d/f/g (no suffix). The half-precision forms
// hbfn/<n> and hbfe/<n> take a mantissa-length suffix <n> in 6..9, printed
// inside the mnemonic.
// --------------------------------------------------------------------------
foreach p = ["d", "f", "g"] in
  def BFN_#p : ALU1P<!strconcat(p, "bfn")>;

let PrintMethod = "printBFMantOperand" in
  def bfmantsuffix : Operand<i64>;

class ALUBF<string opc>
    : PEExpr<(outs),
             (ins bfmantsuffix:$n, peoperand:$src, peoperand:$dst),
             !strconcat(opc, "$n $src $dst")> {
  let Itinerary = IIC_alupe;
}
def HBFN : ALUBF<"hbfn">;
def HBFE : ALUBF<"hbfe">;

// --------------------------------------------------------------------------
// passa — the copy (3.6.12.5). both -> d/f/h/l/i/s. 2-longword capable.
// --------------------------------------------------------------------------
foreach p = ["d", "f", "h", "l", "i", "s"] in
  def PASSA_#p : ALU1P<!strconcat(p, "passa")>;

// --------------------------------------------------------------------------
// both (d/f/h/l/i/s), two input. max/min take `u` for the integer
// precisions only (3.6.12.13). packbit takes no `u` (3.6.12.14).
// --------------------------------------------------------------------------
foreach p = ["d", "f", "h", "l", "i", "s"] in {
  def MAX_#p     : ALU2P<!strconcat(p, "max")>;
  def MIN_#p     : ALU2P<!strconcat(p, "min")>;
  def PACKBIT_#p : ALU2P<!strconcat(p, "packbit")>;
}
foreach p = ["l", "i", "s"] in {
  def UMAX_#p : ALU2P<!strconcat("u", p, "max")>;
  def UMIN_#p : ALU2P<!strconcat("u", p, "min")>;
}

// --------------------------------------------------------------------------
// int (l/i/s), two input.
// --------------------------------------------------------------------------

// bitwise and/or/xor (3.6.12.15). Long-word forms are LAND/LOR/LXOR.
foreach p = ["i", "s"] in {
  def AND_#p : ALU2P<!strconcat(p, "and")>;
  def OR_#p  : ALU2P<!strconcat(p, "or")>;
  def XOR_#p : ALU2P<!strconcat(p, "xor")>;
}

// add/sub (3.6.12.16), with the `u` option. Long-word signed forms are
// LADD/LSUB.
foreach p = ["i", "s"] in {
  def ADD_#p : ALU2P<!strconcat(p, "add")>;
  def SUB_#p : ALU2P<!strconcat(p, "sub")>;
}
foreach p = ["l", "i", "s"] in {
  def UADD_#p : ALU2P<!strconcat("u", p, "add")>;
  def USUB_#p : ALU2P<!strconcat("u", p, "sub")>;
}

// shifts (3.6.12.17). lsr takes `u`; lsl/bsl/bsr do not.
foreach p = ["l", "i", "s"] in {
  def LSL_#p : ALU2P<!strconcat(p, "lsl")>;
  def LSR_#p : ALU2P<!strconcat(p, "lsr")>;
  def BSL_#p : ALU2P<!strconcat(p, "bsl")>;
  def BSR_#p : ALU2P<!strconcat(p, "bsr")>;
  def ULSR_#p : ALU2P<!strconcat("u", p, "lsr")>;
}

// --------------------------------------------------------------------------
// ReLU family (3.6.12.18): float d/f/h, two inputs, no `u`.
// --------------------------------------------------------------------------
foreach p = ["d", "f", "h"] in {
  def RELU_#p    : ALU2P<!strconcat(p, "relu")>;
  def RELU0_#p   : ALU2P<!strconcat(p, "relu0")>;
  def RELU1_#p   : ALU2P<!strconcat(p, "relu1")>;
  def RELU2_#p   : ALU2P<!strconcat(p, "relu2")>;
  def RELU3_#p   : ALU2P<!strconcat(p, "relu3")>;
  def LRELUD_#p  : ALU2P<!strconcat(p, "lrelud")>;
  def LRELUO_#p  : ALU2P<!strconcat(p, "lreluo")>;
  def ILRELUD_#p : ALU2P<!strconcat(p, "ilrelud")>;
}

// --------------------------------------------------------------------------
// Masked twins for the whole ALU catalogue (3.6.2). Every ALU output is
// maskable; rather than hand-writing a twin per opcode, instantiate the
// generic multicast classes below for each mnemonic. The five defs above
// already have hand-written twins (FMIN_M, FMAX_M, IMIN_M, DADD_M, FMUL_M)
// and are not repeated here.
// Operand order: sources, destination, zero-flush mask, write mask.
// --------------------------------------------------------------------------
multiclass MALU1<string opc> {
  def _M : PEExpr<(outs),
                  (ins peoperand:$src, peoperand:$dst,
                       maskoperand:$zmask, maskoperand:$wmask),
                  !strconcat(opc, "$zmask $src $dst$wmask")> {
    let Itinerary = IIC_alupe;
    let TSFlags{14} = 1;
    let TSFlags{15} = 1;
  }
}
multiclass MALU2<string opc> {
  def _M : PEExpr<(outs),
                  (ins peoperand:$sx, peoperand:$sy, peoperand:$dst,
                       maskoperand:$zmask, maskoperand:$wmask),
                  !strconcat(opc, "$zmask $sx $sy $dst$wmask")> {
    let Itinerary = IIC_alupe;
    let TSFlags{14} = 1;
    let TSFlags{15} = 1;
  }
}

foreach p = ["d", "f", "g"] in
  defm BFN_#p : MALU1<!strconcat(p, "bfn")>;
foreach p = ["d", "f", "h", "l", "i", "s"] in
  defm PASSA_#p : MALU1<!strconcat(p, "passa")>;
defm MSL : MALU1<"msl">;
defm MSR : MALU1<"msr">;
defm INOT : MALU1<"inot">;
defm SNOT : MALU1<"snot">;
defm ILNOT : MALU1<"ilnot">;
defm SLNOT : MALU1<"slnot">;
foreach p = ["i", "s"] in {
  defm INC_#p : MALU1<!strconcat(p, "inc")>;
  defm DEC_#p : MALU1<!strconcat(p, "dec")>;
}
foreach p = ["l", "i", "s"] in {
  defm UINC_#p : MALU1<!strconcat("u", p, "inc")>;
  defm UDEC_#p : MALU1<!strconcat("u", p, "dec")>;
}
foreach p = ["d", "f", "h"] in {
  defm RSQRT_#p : MALU1<!strconcat(p, "rsqrt")>;
  defm FLOOR_#p : MALU1<!strconcat(p, "floor")>;
  defm FTOI_#p : MALU1<!strconcat(p, "ftoi")>;
  defm UFTOI_#p : MALU1<!strconcat("u", p, "ftoi")>;
}
foreach p = ["d", "f", "h", "l", "i", "s"] in {
  defm MAX_#p : MALU2<!strconcat(p, "max")>;
  defm MIN_#p : MALU2<!strconcat(p, "min")>;
  defm PACKBIT_#p : MALU2<!strconcat(p, "packbit")>;
}
foreach p = ["l", "i", "s"] in {
  defm UMAX_#p : MALU2<!strconcat("u", p, "max")>;
  defm UMIN_#p : MALU2<!strconcat("u", p, "min")>;
}
foreach p = ["i", "s"] in {
  defm AND_#p : MALU2<!strconcat(p, "and")>;
  defm OR_#p : MALU2<!strconcat(p, "or")>;
  defm XOR_#p : MALU2<!strconcat(p, "xor")>;
  defm ADD_#p : MALU2<!strconcat(p, "add")>;
  defm SUB_#p : MALU2<!strconcat(p, "sub")>;
}
foreach p = ["l", "i", "s"] in {
  defm UADD_#p : MALU2<!strconcat("u", p, "add")>;
  defm USUB_#p : MALU2<!strconcat("u", p, "sub")>;
  defm LSL_#p : MALU2<!strconcat(p, "lsl")>;
  defm LSR_#p : MALU2<!strconcat(p, "lsr")>;
  defm BSL_#p : MALU2<!strconcat(p, "bsl")>;
  defm BSR_#p : MALU2<!strconcat(p, "bsr")>;
  defm ULSR_#p : MALU2<!strconcat("u", p, "lsr")>;
}
foreach p = ["d", "f", "h"] in {
  defm RELU_#p : MALU2<!strconcat(p, "relu")>;
  defm RELU0_#p : MALU2<!strconcat(p, "relu0")>;
  defm RELU1_#p : MALU2<!strconcat(p, "relu1")>;
  defm RELU2_#p : MALU2<!strconcat(p, "relu2")>;
  defm RELU3_#p : MALU2<!strconcat(p, "relu3")>;
  defm LRELUD_#p : MALU2<!strconcat(p, "lrelud")>;
  defm LRELUO_#p : MALU2<!strconcat(p, "lreluo")>;
  defm ILRELUD_#p : MALU2<!strconcat(p, "ilrelud")>;
}

// Masked half-precision block-float forms (they carry the /<n> mantissa
// suffix operand as well).
class ALUBFM<string opc>
    : PEExpr<(outs),
             (ins bfmantsuffix:$n, peoperand:$src, peoperand:$dst,
                  maskoperand:$zmask, maskoperand:$wmask),
             !strconcat(opc, "$n$zmask $src $dst$wmask")> {
  let Itinerary = IIC_alupe;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}
def HBFN_M : ALUBFM<"hbfn">;
def HBFE_M : ALUBFM<"hbfe">;

// Three-destination ALU copy and the mask-register ($omr) output forms
// (3.6.1, 3.6.1.13). The OMR operand is separate from the ndst count: it
// is recognized by marker bit 32 and recorded as an MS_OMR write.
class ALU1PMD3<string opc>
    : PEExpr<(outs),
             (ins peoperand:$src, peoperand:$d0, peoperand:$d1,
                  peoperand:$d2),
             !strconcat(opc, " $src $d0 $d1 $d2")> {
  let Itinerary = IIC_alupe;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
def LPASSA3 : ALU1PMD3<"lpassa">;

class ALU1PMO<string opc>
    : PEExpr<(outs),
             (ins peoperand:$src, peoperand:$dst, omroperand:$omr),
             !strconcat(opc, " $src $dst $omr")> {
  let Itinerary = IIC_alupe;
}
def LPASSA_O : ALU1PMO<"lpassa">;

class ALU1PMD2O<string opc>
    : PEExpr<(outs),
             (ins peoperand:$src, peoperand:$d0, peoperand:$d1,
                  omroperand:$omr),
             !strconcat(opc, " $src $d0 $d1 $omr")> {
  let Itinerary = IIC_alupe;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
def LPASSA2O : ALU1PMD2O<"lpassa">;
