//===- MNCoreInstrMAU.td - MN-Core2 MAU PE expressions ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MAU instruction expressions (sections 3.6.9-3.6.11) occupy the three
// parallel-execution groups mau-calc / mau-mwrite / mau-mread. At most
// two of the three groups may issue in one step, and the two issued
// expressions must carry the same precision letter (enforced in the
// packetizer — see isLegalToPacketizeTogether and mauGroup/precision).
//
// Precision letters (opcode prefix):
//   d = double, f = single, g = pseudo-single, h = half
// Pseudo-single has NO vector forms (no gvfma/gvmul/gvadd/gvpassa);
// f and g differ only inside the MAU pipeline, and mwrite/mread have
// both spellings so co-issued expressions can carry matching letters.
//
// Operand modifiers carried by the packed peoperand extend the M3
// layout (sections 3.6.9.22-3.6.9.24):
//   [24]  neg : leading '-' sign inversion on a non-matrix input
//   [23]  e   : 'e' one-precision-up input extend
//   [25]  rd  : 'r' one-precision-down input reduce (vfma/vmul/mwrite
//               half-precision inputs read single-precision data)
// The 'r' output-precision-reduce of section 3.6.9.25 and the
// dmfma/dmmul/dvfma/dvmul (u|d) half-select are part of the opcode
// text, so each is its own def.
//
// Matrix-register operands: matoperand packs "$l(x|y)<addr>"
// (mwrite/mread row/column start), matside packs "$lx"/"$ly" (the
// whole-face operand of matvec expressions, grammar (2) of section
// 3.6.1.14). The touched face is a property of the operand value, so
// every face-touching form has an _X/_Y def pair reserving the
// matching MATX/MATY unit through its itinerary class.
//
// Multi-destination forms (<dst_0> [<dst_1>..]) are provided for the
// vector ops below (VFMA2/V2MD/V1MD). Matvec multi-dst and 3+ destinations
// are not modelled yet — TableGen's asm writer cannot print a variadic
// operand list, so each arity needs its own class.
//===----------------------------------------------------------------------===//

include "MNCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

// "$l(x|y)<addr>" — matrix register operand with a start row (write)
// or column (transposed read) address. Packed i64:
//   [0]    face  : 0=x 1=y
//   [8:1]  addr  : start row/column (wraps per precision)
//   [9]    ll    : $llx/$ll spelling (hmwrite/hmread 2-lw access)
let PrintMethod = "printMATOperand" in
  def matoperand : Operand<i64>;

// "$lx" / "$ly" — whole-face operand of the matvec expressions
// (grammar (2) of section 3.6.1.14). Packed i64: [0] face.
let PrintMethod = "printMATSideOperand" in
  def matside : Operand<i64>;

//===----------------------------------------------------------------------===//
// mau-calc expressions (section 3.6.9).
//
// Operand order is the .vsm input-then-output order. Each class fixes
// the group itinerary; face-touching classes take the itinerary as a
// parameter so _X/_Y defs differ only in their reservation.
//===----------------------------------------------------------------------===//

// matvec FMA / matvec product:
//   <p>mfma[r][u|d] $l(x|y) <src_x> <src_y> <dst>
//   <p>mmul[r][u|d] $l(x|y) <src_x> <dst>
// grp/face/prec feed the TSFlags tags the packetizer reads (see
// MAUExpr in MNCoreInstrFormats.td). itin already encodes group+face;
// the extra tag is the precision letter for the §3.6.4 same-precision
// rule.
class MATFMA<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins matside:$mat, peoperand:$sx, peoperand:$sy,
                   peoperand:$dst),
              !strconcat(op, " $mat $sx $sy $dst"), [], /*grp*/1, fc, pr> {
  let Itinerary = itin;
}
class MATMUL<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins matside:$mat, peoperand:$sx, peoperand:$dst),
              !strconcat(op, " $mat $sx $dst"), [], /*grp*/1, fc, pr> {
  let Itinerary = itin;
}

// vector forms (no matrix face; IIC_mau for all):
//   <p>vfma[r][u|d] <src_x> <src_y> <src_z> <dst>
//   <p>vmul[r][u|d] <src_x> <src_y> <dst>
//   <p>vadd[r]      <src_x> <src_y> <dst>
//   <p>vpassa[r]    <src_x> <dst>
class VFMA<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$sz,
                   peoperand:$dst),
              !strconcat(op, " $sx $sy $sz $dst"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = 1; // vfma/vmul family for the §3.6.4 shape rule
}
class V2<string op, int pr, int isVfma = 0>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$dst),
              !strconcat(op, " $sx $sy $dst"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = isVfma; // vmul carries the shape rule, vadd does not
}
class V1<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$dst),
              !strconcat(op, " $sx $dst"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
}

// Multi-destination vector forms (3.6.1: <dst_0> [<dst_1>..]). A MAU
// result may be written to several PE memories in one expression
// (3.7.3). TSFlags[17] flags the form and [20:18] carries the
// destination count; `isVfma` keeps the 3.6.4 shape rule applicable.
class VFMA2<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$sz,
                   peoperand:$d0, peoperand:$d1),
              !strconcat(op, " $sx $sy $sz $d0 $d1"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = 1;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
class V2MD<string op, int pr, int isVfma = 0>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$d0,
                   peoperand:$d1),
              !strconcat(op, " $sx $sy $d0 $d1"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = isVfma;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
class V1MD<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$d0, peoperand:$d1),
              !strconcat(op, " $sx $d0 $d1"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
def DVPASSA2 : V1MD<"dvpassa", 1>;
def FVPASSA2 : V1MD<"fvpassa", 2>;
def FVFMA2   : VFMA2<"fvfma", 2>;
def FVMUL2   : V2MD<"fvmul", 2, 1>;
def FVADD2   : V2MD<"fvadd", 2>;

// ---- matvec forms: _X/_Y def pairs per face ----
// Face: 1=X 2=Y. Precision: 1=d 2=f 3=g 4=h. multiclass gives properly
// typed params (no foreach string/int mixing).
multiclass matface<InstrItinClass itin, int fc> {
  // d-precision: mandatory (u|d) half-select — one def per half.
  def DMFMAU  : MATFMA<itin, "dmfmau",  fc, 1>;
  def DMFMAD  : MATFMA<itin, "dmfmad",  fc, 1>;
  def DMFMAUR : MATFMA<itin, "dmfmaur", fc, 1>;
  def DMFMADR : MATFMA<itin, "dmfmadr", fc, 1>;
  def DMMULU  : MATMUL<itin, "dmmulu",  fc, 1>;
  def DMMULD  : MATMUL<itin, "dmmuld",  fc, 1>;
  def DMMULUR : MATMUL<itin, "dmmulur", fc, 1>;
  def DMMULDR : MATMUL<itin, "dmmuldr", fc, 1>;
  // f/g/h: plain + output-reduce forms.
  def FMFMA   : MATFMA<itin, "fmfma",  fc, 2>;
  def FMFMAR  : MATFMA<itin, "fmfmar", fc, 2>;
  def FMMUL   : MATMUL<itin, "fmmul",  fc, 2>;
  def FMMULR  : MATMUL<itin, "fmmulr", fc, 2>;
  def GMFMA   : MATFMA<itin, "gmfma",  fc, 3>;
  def GMFMAR  : MATFMA<itin, "gmfmar", fc, 3>;
  def GMMUL   : MATMUL<itin, "gmmul",  fc, 3>;
  def GMMULR  : MATMUL<itin, "gmmulr", fc, 3>;
  def HMFMA   : MATFMA<itin, "hmfma",  fc, 4>;
  def HMFMAR  : MATFMA<itin, "hmfmar", fc, 4>;
  def HMMUL   : MATMUL<itin, "hmmul",  fc, 4>;
  def HMMULR  : MATMUL<itin, "hmmulr", fc, 4>;
}
defm X : matface<IIC_mauX, 1>;
defm Y : matface<IIC_mauY, 2>;

// ---- vector ops (d/f/h only; pseudo-single is matvec-only) ----
// Precision: 1=d 2=f 4=h. d-precision vector FMA/mul carry the
// mandatory (u|d) half-select.
def DVFMAU   : VFMA<"dvfmau", 1>;
def DVFMAD   : VFMA<"dvfmad", 1>;
def DVFMAUR  : VFMA<"dvfmaur", 1>;
def DVFMADR  : VFMA<"dvfmadr", 1>;
def DVMULU   : V2<"dvmulu", 1, 1>;
def DVMULD   : V2<"dvmuld", 1, 1>;
def DVMULUR  : V2<"dvmulur", 1, 1>;
def DVMULDR  : V2<"dvmuldr", 1, 1>;
def DVADD    : V2<"dvadd", 1>;
def DVADDR   : V2<"dvaddr", 1>;
def DVPASSA  : V1<"dvpassa", 1>;
def DVPASSAR : V1<"dvpassar", 1>;

def FVFMA    : VFMA<"fvfma", 2>;
def FVFMAR   : VFMA<"fvfmar", 2>;
def FVMUL    : V2<"fvmul", 2, 1>;
def FVMULR   : V2<"fvmulr", 2, 1>;
def FVADD    : V2<"fvadd", 2>;
def FVADDR   : V2<"fvaddr", 2>;
def FVPASSA  : V1<"fvpassa", 2>;
def FVPASSAR : V1<"fvpassar", 2>;

def HVFMA    : VFMA<"hvfma", 4>;
def HVFMAR   : VFMA<"hvfmar", 4>;
def HVMUL    : V2<"hvmul", 4, 1>;
def HVMULR   : V2<"hvmulr", 4, 1>;
def HVADD    : V2<"hvadd", 4>;
def HVADDR   : V2<"hvaddr", 4>;
def HVPASSA  : V1<"hvpassa", 4>;
def HVPASSAR : V1<"hvpassar", 4>;

//===----------------------------------------------------------------------===//
// mau-mwrite (section 3.6.10) and mau-mread (section 3.6.11).
//
//   <p>mwrite <src> $l(x|y)<addr>
//   <p>mread  $l(x|y)<addr> <dst>
//===----------------------------------------------------------------------===//

class MWRITE<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$src, matoperand:$mat),
              !strconcat(op, " $src $mat"), [], /*grp*/2, fc, pr> {
  let Itinerary = itin;
}
class MREAD<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins matoperand:$mat, peoperand:$dst),
              !strconcat(op, " $mat $dst"), [], /*grp*/3, fc, pr> {
  let Itinerary = itin;
}

// Face: 1=X 2=Y. Precision: 1=d 2=f 3=g 4=h.
multiclass matwr<InstrItinClass w, InstrItinClass r, int fc> {
  def DMWRITE : MWRITE<w, "dmwrite", fc, 1>;
  def FMWRITE : MWRITE<w, "fmwrite", fc, 2>;
  def GMWRITE : MWRITE<w, "gmwrite", fc, 3>;
  def HMWRITE : MWRITE<w, "hmwrite", fc, 4>;
  def DMREAD  : MREAD<r,  "dmread",  fc, 1>;
  def FMREAD  : MREAD<r,  "fmread",  fc, 2>;
  def GMREAD  : MREAD<r,  "gmread",  fc, 3>;
  def HMREAD  : MREAD<r,  "hmread",  fc, 4>;
}
defm W_X : matwr<IIC_mwrX, IIC_mrdX, 1>;
defm W_Y : matwr<IIC_mwrY, IIC_mrdY, 2>;

//===----------------------------------------------------------------------===//
// Masked twins (section 3.6.2). A masked expression may carry a zero-flush
// mask (3.6.2.2, printed after the mnemonic: "<opc>/(...) ...") and/or a
// write mask (3.6.2.1, printed after the destination: "... $dst/(...)").
// Both mask operands are appended after the ordinary operands (so the PE
// operand indices the decoder uses are unchanged); a clear present bit
// prints nothing. TSFlags bits 14/15 tell the decoder the trailing
// immediates are mask specifiers.
//
// The `_M` record suffix keeps the plain defs (and the isel paths that use
// them) untouched.
//===----------------------------------------------------------------------===//
class MVFMAM<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$sz,
                   peoperand:$dst, maskoperand:$zmask, maskoperand:$wmask),
              !strconcat(op, "$zmask $sx $sy $sz $dst$wmask"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = 1;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}
class MV2M<string op, int pr, int isVfma = 0>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$dst,
                   maskoperand:$zmask, maskoperand:$wmask),
              !strconcat(op, "$zmask $sx $sy $dst$wmask"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = isVfma;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}
class MV1M<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$dst,
                   maskoperand:$zmask, maskoperand:$wmask),
              !strconcat(op, "$zmask $sx $dst$wmask"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}
class MMWRITEM<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$src, matoperand:$mat,
                   maskoperand:$zmask, maskoperand:$wmask),
              !strconcat(op, "$zmask $src $mat$wmask"), [], /*grp*/2, fc, pr> {
  let Itinerary = itin;
  let TSFlags{14} = 1;
  let TSFlags{15} = 1;
}

// Masked vector defs. The mask operands are optional at the text level (the
// present bit is clear unless set), so these are the same .vsm spellings as
// the plain defs plus an optional "/(...)".
def FVFMA_M    : MVFMAM<"fvfma", 2>;
def FVFMAR_M   : MVFMAM<"fvfmar", 2>;
def HVFMA_M    : MVFMAM<"hvfma", 4>;
def HVFMAR_M   : MVFMAM<"hvfmar", 4>;
def DVFMAU_M   : MVFMAM<"dvfmau", 1>;
def DVFMAUR_M  : MVFMAM<"dvfmaur", 1>;
def DVMULU_M   : MV2M<"dvmulu", 1, 1>;
def DVMULD_M   : MV2M<"dvmuld", 1, 1>;
def FVMUL_M    : MV2M<"fvmul", 2, 1>;
def FVMULR_M   : MV2M<"fvmulr", 2, 1>;
def HVMUL_M    : MV2M<"hvmul", 4, 1>;
def HVMULR_M   : MV2M<"hvmulr", 4, 1>;
def FVADD_M    : MV2M<"fvadd", 2>;
def HVADD_M    : MV2M<"hvadd", 4>;
def FVPASSA_M  : MV1M<"fvpassa", 2>;
def HVPASSA_M  : MV1M<"hvpassa", 4>;

// Masked mwrite is deliberately NOT defined: assemble3 rejects both the
// zero-flush and write-mask spellings on mwrite ("Trailing letter after
// ... mwrite instruction"). mwrite's only destination is the matrix
// register, and section 3.6.2.1 lists the maskable destinations as
// GRF0/GRF1/T/LM0/LM1 — the matrix register is not among them. A masked
// mwrite is therefore not expressible, and none is emitted.

// Vector three-destination and mask-register-destination forms.
class V1MD3<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$d0, peoperand:$d1,
                   peoperand:$d2),
              !strconcat(op, " $sx $d0 $d1 $d2"), [], /*grp*/1, /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
class V2MD3<string op, int pr, int isVfma = 0>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$d0,
                   peoperand:$d1, peoperand:$d2),
              !strconcat(op, " $sx $sy $d0 $d1 $d2"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = isVfma;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
class VFMA3<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$sz,
                   peoperand:$d0, peoperand:$d1, peoperand:$d2),
              !strconcat(op, " $sx $sy $sz $d0 $d1 $d2"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = 1;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 3;
}
def DVPASSA3 : V1MD3<"dvpassa", 1>;
def FVFMA3   : VFMA3<"fvfma", 2>;
def FVMUL3   : V2MD3<"fvmul", 2, 1>;

// Mask-register destination (3.6.1.13): the mask flags are written to
// $omr<addr> while the ordinary result goes to the PE destination.
class V1MDO<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$dst, omroperand:$omr),
              !strconcat(op, " $sx $dst $omr"), [], /*grp*/1, /*fc*/0, pr> {
  let Itinerary = IIC_mau;
}
class V2MDO<string op, int pr, int isVfma = 0>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$dst,
                   omroperand:$omr),
              !strconcat(op, " $sx $sy $dst $omr"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = isVfma;
}
class VFMAO<string op, int pr>
    : MAUExpr<(outs),
              (ins peoperand:$sx, peoperand:$sy, peoperand:$sz,
                   peoperand:$dst, omroperand:$omr),
              !strconcat(op, " $sx $sy $sz $dst $omr"), [], /*grp*/1,
              /*fc*/0, pr> {
  let Itinerary = IIC_mau;
  let TSFlags{13} = 1;
}
def DVPASSA_O : V1MDO<"dvpassa", 1>;
def FVFMA_O   : VFMAO<"fvfma", 2>;
def FVMUL_O   : V2MDO<"fvmul", 2, 1>;

// Matvec multi-destination forms (the manual's <dst_0> [<dst_1>..] also
// applies to MAU matrix-vector ops; 3.7.3).
class MATFMA2<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins matside:$mat, peoperand:$sx, peoperand:$sy,
                   peoperand:$d0, peoperand:$d1),
              !strconcat(op, " $mat $sx $sy $d0 $d1"), [], /*grp*/1, fc, pr> {
  let Itinerary = itin;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
class MATMUL2<InstrItinClass itin, string op, int fc, int pr>
    : MAUExpr<(outs),
              (ins matside:$mat, peoperand:$sx, peoperand:$d0,
                   peoperand:$d1),
              !strconcat(op, " $mat $sx $d0 $d1"), [], /*grp*/1, fc, pr> {
  let Itinerary = itin;
  let TSFlags{17} = 1;
  let TSFlags{20-18} = 2;
}
multiclass matface2<InstrItinClass itin, int fc> {
  def DMFMAU2 : MATFMA2<itin, "dmfmau", fc, 1>;
  def DMFMAD2 : MATFMA2<itin, "dmfmad", fc, 1>;
  def DMFMAUR2 : MATFMA2<itin, "dmfmaur", fc, 1>;
  def DMFMADR2 : MATFMA2<itin, "dmfmadr", fc, 1>;
  def DMMULU2 : MATMUL2<itin, "dmmulu", fc, 1>;
  def DMMULD2 : MATMUL2<itin, "dmmuld", fc, 1>;
  def FMFMA2 : MATFMA2<itin, "fmfma", fc, 2>;
  def FMFMAR2 : MATFMA2<itin, "fmfmar", fc, 2>;
  def FMMUL2 : MATMUL2<itin, "fmmul", fc, 2>;
  def HMFMA2 : MATFMA2<itin, "hmfma", fc, 4>;
  def HMFMAR2 : MATFMA2<itin, "hmfmar", fc, 4>;
  def HMMUL2 : MATMUL2<itin, "hmmul", fc, 4>;
}
defm XMD : matface2<IIC_mauX, 1>;
defm YMD : matface2<IIC_mauY, 2>;
