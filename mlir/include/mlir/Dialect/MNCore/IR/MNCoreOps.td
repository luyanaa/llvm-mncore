//===- MNCoreOps.td - MN-Core 2 dialect operations ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operations of the `mncore` dialect. See MNCoreDialect.td for the design
// contract and llvm/lib/Target/MNCore/docs/mncore_intrinsics_design.md for the
// backend ABI these ops lower to.
//
// Enum values below are the frozen ABI values from
// llvm/include/llvm/Target/MNCore/MNCoreIntrinsicOps.h. They are append-only and must
// never be renumbered; `mncore-opcode-sync` (mlir/test/Dialect/MNCore) checks
// that the two sources agree.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MNCORE_IR_MNCOREOPS_TD
#define MLIR_DIALECT_MNCORE_IR_MNCOREOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/MNCore/IR/MNCoreDialect.td"

//===----------------------------------------------------------------------===//
// Opcode enums (frozen ABI)
//===----------------------------------------------------------------------===//

def MNCore_ALUOp_FMin   : I64EnumAttrCase<"FMin", 0>;
def MNCore_ALUOp_FMax   : I64EnumAttrCase<"FMax", 1>;
def MNCore_ALUOp_IAdd   : I64EnumAttrCase<"IAdd", 2>;
def MNCore_ALUOp_ISub   : I64EnumAttrCase<"ISub", 3>;
def MNCore_ALUOp_LPassa : I64EnumAttrCase<"LPassa", 4>;
def MNCore_ALUOp_DMax   : I64EnumAttrCase<"DMax", 10>;
def MNCore_ALUOp_DMin   : I64EnumAttrCase<"DMin", 11>;
def MNCore_ALUOp_HMax   : I64EnumAttrCase<"HMax", 12>;
def MNCore_ALUOp_HMin   : I64EnumAttrCase<"HMin", 13>;
// Block float (SDM 3.6.12.12), new range.
def MNCore_ALUOp_FBFN   : I64EnumAttrCase<"FBFN", 20>;

def MNCore_ALUOp : I64EnumAttr<"ALUOp", "MN-Core ALU opcode", [
  MNCore_ALUOp_FMin, MNCore_ALUOp_FMax, MNCore_ALUOp_IAdd,
  MNCore_ALUOp_ISub, MNCore_ALUOp_LPassa,
  MNCore_ALUOp_DMax, MNCore_ALUOp_DMin,
  MNCore_ALUOp_HMax, MNCore_ALUOp_HMin,
  MNCore_ALUOp_FBFN,
]> {
  let cppNamespace = "::mlir::mncore";
}

def MNCore_MAUOp_FVFMA    : I64EnumAttrCase<"FVFMA", 0>;
def MNCore_MAUOp_FVMUL    : I64EnumAttrCase<"FVMUL", 1>;
def MNCore_MAUOp_FVADD    : I64EnumAttrCase<"FVADD", 2>;
def MNCore_MAUOp_FVPASSA  : I64EnumAttrCase<"FVPASSA", 3>;
def MNCore_MAUOp_FMatVec  : I64EnumAttrCase<"FMatVec", 4>;
def MNCore_MAUOp_FMWrite  : I64EnumAttrCase<"FMWrite", 5>;
def MNCore_MAUOp_FMRead   : I64EnumAttrCase<"FMRead", 6>;
def MNCore_MAUOp_DVFMA    : I64EnumAttrCase<"DVFMA", 10>;
def MNCore_MAUOp_DVMUL    : I64EnumAttrCase<"DVMUL", 11>;
def MNCore_MAUOp_DVADD    : I64EnumAttrCase<"DVADD", 12>;
def MNCore_MAUOp_HVFMA    : I64EnumAttrCase<"HVFMA", 13>;
def MNCore_MAUOp_HVMUL    : I64EnumAttrCase<"HVMUL", 14>;
def MNCore_MAUOp_HVADD    : I64EnumAttrCase<"HVADD", 15>;
def MNCore_MAUOp_DMatVec  : I64EnumAttrCase<"DMatVec", 16>;
def MNCore_MAUOp_HMatVec  : I64EnumAttrCase<"HMatVec", 17>;
def MNCore_MAUOp_DMWrite  : I64EnumAttrCase<"DMWrite", 18>;
def MNCore_MAUOp_HMWrite  : I64EnumAttrCase<"HMWrite", 19>;
def MNCore_MAUOp_DMRead   : I64EnumAttrCase<"DMRead", 20>;
def MNCore_MAUOp_HMRead   : I64EnumAttrCase<"HMRead", 21>;

def MNCore_MAUOp : I64EnumAttr<"MAUOp", "MN-Core MAU opcode", [
  MNCore_MAUOp_FVFMA, MNCore_MAUOp_FVMUL, MNCore_MAUOp_FVADD,
  MNCore_MAUOp_FVPASSA, MNCore_MAUOp_FMatVec, MNCore_MAUOp_FMWrite,
  MNCore_MAUOp_FMRead,
  MNCore_MAUOp_DVFMA, MNCore_MAUOp_DVMUL, MNCore_MAUOp_DVADD,
  MNCore_MAUOp_HVFMA, MNCore_MAUOp_HVMUL, MNCore_MAUOp_HVADD,
  MNCore_MAUOp_DMatVec, MNCore_MAUOp_HMatVec,
  MNCore_MAUOp_DMWrite, MNCore_MAUOp_HMWrite,
  MNCore_MAUOp_DMRead, MNCore_MAUOp_HMRead,
]> {
  let cppNamespace = "::mlir::mncore";
}

def MNCore_L1BOp_L1BMP  : I64EnumAttrCase<"L1BMP", 0>;
def MNCore_L1BOp_L1BMM  : I64EnumAttrCase<"L1BMM", 1>;
def MNCore_L1BOp_L1BMM4 : I64EnumAttrCase<"L1BMM4", 2>;
def MNCore_L1BOp_L1BMD  : I64EnumAttrCase<"L1BMD", 3>;

def MNCore_L1BEnum : I64EnumAttr<"L1BKind", "MN-Core L1BM opcode", [
  MNCore_L1BOp_L1BMP, MNCore_L1BOp_L1BMM, MNCore_L1BOp_L1BMM4,
  MNCore_L1BOp_L1BMD,
]> {
  let cppNamespace = "::mlir::mncore";
}

def MNCore_L2BOp_L2BMB  : I64EnumAttrCase<"L2BMB", 0>;
def MNCore_L2BOp_L2BMDC : I64EnumAttrCase<"L2BMDC", 1>;
def MNCore_L2BOp_L2BMPT : I64EnumAttrCase<"L2BMPT", 2>;

def MNCore_L2BEnum : I64EnumAttr<"L2BKind", "MN-Core L2BM opcode", [
  MNCore_L2BOp_L2BMB, MNCore_L2BOp_L2BMDC, MNCore_L2BOp_L2BMPT,
]> {
  let cppNamespace = "::mlir::mncore";
}

def MNCore_MVOp_MVP : I64EnumAttrCase<"MVP", 0>;

def MNCore_MVOp : I64EnumAttr<"MVOp", "MN-Core MV opcode", [
  MNCore_MVOp_MVP,
]> {
  let cppNamespace = "::mlir::mncore";
}

// L1BM -> PE distribution mode (SDM 3.6.8.6/.8/.14/.20). The mode is chosen
// by the frontend from the source memdesc layout and the target distributed
// layout; the dialect does not default to any mode.
def MNCore_L1Mode_Broadcast   : I64EnumAttrCase<"Broadcast", 0>;
def MNCore_L1Mode_MAB         : I64EnumAttrCase<"MAB", 1>;
def MNCore_L1Mode_MAB4        : I64EnumAttrCase<"MAB4", 2>;
def MNCore_L1Mode_Distributed : I64EnumAttrCase<"Distributed", 3>;

def MNCore_L1Mode : I64EnumAttr<"L1Mode", "L1BM to PE distribution mode", [
  MNCore_L1Mode_Broadcast, MNCore_L1Mode_MAB, MNCore_L1Mode_MAB4,
  MNCore_L1Mode_Distributed,
]> {
  let cppNamespace = "::mlir::mncore";
}

//===----------------------------------------------------------------------===//
// Side-effect resources
//===----------------------------------------------------------------------===//

def MNCore_PEState   : Resource<"::mlir::mncore::PEState">;
def MNCore_DeviceMem : Resource<"::mlir::mncore::DeviceMem">;
def MNCore_MVChannel : Resource<"::mlir::mncore::MVChannel">;

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class MNCore_Op<string mnemonic, list<Trait> traits = []>
    : Op<MNCore_Dialect, mnemonic, traits>;

// One MN-Core "expression": it occupies one VLIW step and touches the PE
// register file (and possibly device memory). The operands are a fixed-arity
// list of `!mncore.token`; `mncore.none` marks an absent operand position.
class MNCore_InstOp<string mnemonic, list<Trait> traits = []>
    : MNCore_Op<mnemonic, !listconcat(traits, [
        MemoryEffects<[
          MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>,
          MemRead<MNCore_DeviceMem>, MemWrite<MNCore_DeviceMem>
        ]>
      ])> {
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layer A: token builders
//
// Every field is an attribute. This is not an implementation shortcut: the
// .vsm PE operand encoding has no general runtime-address form, so a token
// that cannot be folded at compile time is not representable (design doc
// section 11). Keeping the fields as attributes makes that a verifier error
// instead of a late assemble3 failure.
//===----------------------------------------------------------------------===//

def MNCore_NoneOp : MNCore_Op<"none", [Pure]> {
  let summary = "absent-operand sentinel";
  let description = [{
    Represents an absent optional operand. `convert-mncore-to-llvm` lowers it
    to the backend ABI sentinel `i64 -1`.
  }];
  let results = (outs MNCore_TokenType:$token);
}

def MNCore_PeOpOp : MNCore_Op<"peop", [Pure]> {
  let summary = "build a PE operand token";
  let description = [{
    `space`: 1=GRF0, 2=GRF1, 3=LM0, 4=LM1, 5=T.
    `width`: 0=word, 1=longword, 2=2-longword, 3=T (implicit 2-longword).
    `flags`: 1='-', 2='e', 4='r', 8='v', 16='j', 32=t-indirect,
             64=flat, 128=v<adri>, 256=LM base-address register.
    `aux`: `j<madpe>` or `v<adri>` payload, otherwise 0.
  }];
  let arguments = (ins I32Attr:$space, I64Attr:$addr, I32Attr:$width,
                       I32Attr:$flags, I64Attr:$aux);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

def MNCore_LbOpOp : MNCore_Op<"lbop", [Pure]> {
  let summary = "build an L1BM/L2BM operand token";
  let description = [{
    `mem`: 0=L1BM, 1=L2BM. `width`: 0=word, 1=longword, 2=2-longword.
    `flags`: 1=ll, 2=turnaround ($lbi).
  }];
  let arguments = (ins I32Attr:$mem, I64Attr:$addr, I32Attr:$width,
                       I32Attr:$flags);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

def MNCore_MatOpOp : MNCore_Op<"matop", [Pure]> {
  let summary = "build a matrix-register operand token";
  let description = [{
    `face`: 0=X, 1=Y. `row`: logical row (0..255). `width`: 0=word,
    1=longword, 2=2-longword. `flags` is currently required to be 0.
  }];
  let arguments = (ins I32Attr:$face, I64Attr:$row, I32Attr:$width,
                       I32Attr:$flags);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

def MNCore_MaskOpOp : MNCore_Op<"maskop", [Pure]> {
  let summary = "build a mask specifier token";
  let description = [{
    `kind`: 0=fixed 4-bit pattern, 1=$imr register.
    `value`: pattern (0..15) or $imr address (0..15).
    `flags`: 1=ll, 2=zero-flush, 4='t', 8='p'.
  }];
  let arguments = (ins I32Attr:$kind, I32Attr:$value, I32Attr:$flags);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

def MNCore_MvOpOp : MNCore_Op<"mvop", [Pure]> {
  let summary = "build an MV memory operand token";
  let description = [{
    `kind`: 0=PDM ($p), 1=DRAM ($d), 2=L2BM ($lc), 3=DRAM-indirect ($di).
    `addr` is a 32-bit static address. `flags`: 1=hasGrp, 2..3=grp,
    4=hasL2b, 8=l2b.
  }];
  let arguments = (ins I32Attr:$kind, I64Attr:$addr, I32Attr:$flags);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

def MNCore_MvParamsOp : MNCore_Op<"mvparams", [Pure]> {
  let summary = "build MV parameters";
  let description = [{
    `size` [31:0], `tag` [39:32], `nd` [47:40] (DAR reuse count,
    DRAM-indirect), `prio` [55:48]. 0xff marks an absent optional field.
  }];
  let arguments = (ins I64Attr:$size, I32Attr:$tag, I32Attr:$nd,
                       I32Attr:$prio);
  let results = (outs MNCore_TokenType:$token);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layer A: instruction expressions
//
// Operand arity (including `mncore.none` placeholders) is fixed and checked by
// the verifier:
//   alu:  sx, sy, d0, d1, d2, omr, zmask, wmask            (8)
//   mau:  mat, sx, sy, sz, d0, d1, d2, omr, zmask, wmask  (10)
//   l1b:  src, dst, set, zmask, wmask                      (5)
//   l2b:  set, a, b                                        (3)
//   mv:   params, src, dst                                 (3)
//===----------------------------------------------------------------------===//

def MNCore_AluOp : MNCore_InstOp<"alu"> {
  let summary = "ALU expression";
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens,
                       MNCore_ALUOp:$opcode, I32Attr:$prec, I32Attr:$opt);
}

def MNCore_MauOp : MNCore_InstOp<"mau"> {
  let summary = "MAU expression (vector, matvec, mwrite, mread)";
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens,
                       MNCore_MAUOp:$opcode, I32Attr:$prec);
}

def MNCore_L1BOp : MNCore_InstOp<"l1b"> {
  let summary = "L1BM expression";
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens,
                       MNCore_L1BEnum:$opcode, I32Attr:$dir);
}

def MNCore_L2BOp : MNCore_InstOp<"l2b"> {
  let summary = "L2BM expression";
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens,
                       MNCore_L2BEnum:$opcode);
}

def MNCore_MvOp : MNCore_InstOp<"mv"> {
  let summary = "MV command";
  let arguments = (ins Variadic<MNCore_TokenType>:$tokens,
                       MNCore_MVOp:$opcode);
}

// DRAM/PDM -> L2BM -> L1BM copy (one MV command plus one L2BM->L1B move).
// Addresses are longword units.
//
// `src_arg >= 0` makes the source address relocatable (kernel ABI §13): the
// emitted MV operand is a placeholder 0 and `src_addr` becomes the static
// offset within the buffer, recorded as `addend_units` in the sidecar.
// `src_arg = -1` keeps the absolute static address.
def MNCore_DmaOp : MNCore_Op<"dma", [
    MemoryEffects<[
      MemRead<MNCore_DeviceMem>, MemWrite<MNCore_DeviceMem>,
      MemRead<MNCore_MVChannel>, MemWrite<MNCore_MVChannel>
    ]>]> {
  let summary = "DRAM/PDM -> L2BM -> L1BM copy";
  let arguments = (ins I32Attr:$src_space, I64Attr:$src_addr,
                       I64Attr:$l2_addr, I64Attr:$l1_addr,
                       I64Attr:$size, I32Attr:$tag,
                       DefaultValuedAttr<I32Attr, "0">:$mv_flags,
                       DefaultValuedAttr<I32Attr, "-1">:$src_arg);
  let hasVerifier = 1;
}

// L1BM -> L2BM -> DRAM copy. Addresses are longword units. The caller is
// responsible for the matching wait on `tag` before reusing the L1BM/L2BM
// region.
//
// The transfer is `l2bm@<l1b>`, the single-L1B point form (SDM 3.6.7.5), not
// `l2bmd` (3.6.7.8). A Triton "distributed" tile is replicated: every L1B
// holds the same plane, so `l2bmd` would interleave eight identical
// 8-longword chunks into L2BM and destroy the plane. `l2bmd` is only correct
// when the L1Bs hold disjoint data, which this ABI does not produce yet.
// `dr_arg >= 0` makes the DRAM destination relocatable (kernel ABI §13): the
// emitted MV operand is a placeholder 0 and `dr_addr` becomes the static
// offset within the buffer, recorded as `addend_units` in the sidecar.
def MNCore_DmaStoreOp : MNCore_Op<"dma.store", [
    MemoryEffects<[
      MemRead<MNCore_DeviceMem>, MemWrite<MNCore_DeviceMem>,
      MemRead<MNCore_MVChannel>, MemWrite<MNCore_MVChannel>
    ]>]> {
  let summary = "L1BM -> L2BM -> DRAM copy";
  let arguments = (ins I64Attr:$l1_addr, I64Attr:$l2_addr, I64Attr:$dr_addr,
                       I64Attr:$size, I32Attr:$tag,
                       DefaultValuedAttr<I32Attr, "0">:$l1b,
                       DefaultValuedAttr<I32Attr, "-1">:$dr_arg);
  let hasVerifier = 1;
}

def MNCore_WaitOp : MNCore_Op<"wait", [
    MemoryEffects<[MemRead<MNCore_MVChannel>, MemWrite<MNCore_MVChannel>]>]> {
  let summary = "wait on an MV tag";
  let arguments = (ins I32Attr:$tag);
  let hasVerifier = 1;
}

def MNCore_FenceOp : MNCore_Op<"fence", [
    MemoryEffects<[
      MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>,
      MemRead<MNCore_DeviceMem>, MemWrite<MNCore_DeviceMem>,
      MemRead<MNCore_MVChannel>, MemWrite<MNCore_MVChannel>
    ]>]> {
  let summary = "ordering barrier between instruction expressions";
}

def MNCore_MaskDirOp : MNCore_Op<"maskdir", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>]> {
  let summary = "multi-line write-mask directive";
  let arguments = (ins I32Attr:$spec);
}

//===----------------------------------------------------------------------===//
// Layer B: typed vector and matrix operations
//
// These lower to the Layer B intrinsics (llvm.mncore.fma.v4f32, ...), so the
// PE slot allocator in MNCoreLowerTypedIntrinsics.cpp remains the single
// implementation. Masks are compile-time constants; -1 means "no mask".
//===----------------------------------------------------------------------===//

class MNCore_VecOp<string mnemonic, list<Trait> traits = []>
    : MNCore_Op<mnemonic, !listconcat(traits, [
        SameOperandsAndResultType,
        MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>
      ])> {
  let results = (outs AnyFixedVectorOfNonZeroRank:$result);
  let hasVerifier = 1;
}

def MNCore_VecFmaOp : MNCore_VecOp<"vec.fma"> {
  let summary = "d = a * b + c";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$a, AnyFixedVectorOfNonZeroRank:$b,
                       AnyFixedVectorOfNonZeroRank:$c,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
}

def MNCore_VecMulOp : MNCore_VecOp<"vec.mul"> {
  let summary = "elementwise multiply";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$lhs, AnyFixedVectorOfNonZeroRank:$rhs,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
}

def MNCore_VecAddOp : MNCore_VecOp<"vec.add"> {
  let summary = "elementwise add";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$lhs, AnyFixedVectorOfNonZeroRank:$rhs,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
}

def MNCore_VecMaxOp : MNCore_VecOp<"vec.max"> {
  let summary = "elementwise maximum";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$lhs, AnyFixedVectorOfNonZeroRank:$rhs,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
}

def MNCore_VecMinOp : MNCore_VecOp<"vec.min"> {
  let summary = "elementwise minimum";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$lhs, AnyFixedVectorOfNonZeroRank:$rhs,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
}

// `mat.vec` is deliberately NOT a `MNCore_VecOp`: the three operand roles have
// different hardware widths (SDM 4.4, vendor header). For f32, x is one 32-bit
// word per PE per cycle (4 f32 = 2 longwords over the 4-cycle expression),
// while y and the result are one longword per PE per cycle (8 f32 = 4
// longwords). `fma`/`mul`/... keep the uniform `<4 x f32>` form.
def MNCore_MatVecOp : MNCore_Op<"mat.vec", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>]> {
  let summary = "dst = A*x + y";
  let arguments = (ins I64Attr:$face, AnyFixedVectorOfNonZeroRank:$x,
                       AnyFixedVectorOfNonZeroRank:$y);
  let results = (outs AnyFixedVectorOfNonZeroRank:$result);
  let hasVerifier = 1;
}

def MNCore_MatWriteOp : MNCore_Op<"mat.write", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>]> {
  let summary = "write four rows of the matrix register";
  let description = [{
    One `fmwrite` covers four rows: cycle c writes row `(row + c) mod 8`, with
    PE q's longword LOW half landing on column 2q and the HIGH half on 2q+1.
    For f32 the operand is therefore 4 longwords per PE (`<8 x float>`, four
    rows x two columns), not a single row. `src` must already be in the
    block-float format the matrix face stores: this op does not convert.
  }];
  let arguments = (ins I64Attr:$face, I64Attr:$row,
                       AnyFixedVectorOfNonZeroRank:$src);
  let hasVerifier = 1;
}

def MNCore_MatReadOp : MNCore_Op<"mat.read", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>]> {
  let summary = "transposed read of four columns of the matrix register";
  let description = [{
    Cycle c reads column `(col + c)`, transposed: PE q receives row 2q in the
    HIGH half and row 2q+1 in the LOW half. For f32 the result is 4 longwords
    per PE (`<8 x float>`, four columns x two rows). The read is a bit copy;
    no floating-point normalisation happens (block-float payloads come back
    raw).
  }];
  let arguments = (ins I64Attr:$face, I64Attr:$row);
  let results = (outs AnyFixedVectorOfNonZeroRank:$result);
  let hasVerifier = 1;
}

// One 8x4x4 GEMM tile: C = A(8x4) * B(4x4) + C0, as a single MAU matrix
// expression plus the two writes that fill the face.
//
//   a_lo, a_hi  block-float slabs for matrix rows 0-3 and 4-7. Each PE holds
//               two columns of four rows; A's four K values sit in even
//               columns 0,2,4,6 (the only columns a single-precision MAU
//               matvec reads). 4 longwords per PE each.
//   b           the K values for the output column owned by this PE
//               (4 f32 = 2 longwords per PE).
//   c0, result  the IEEE f32 accumulator, 8 f32 = 4 longwords per PE, one
//               output row per element, result[m] = sum_k A[m][k]*b[k].
//
// The block-float conversion and the L1BM -> PE distribution are NOT part of
// this op. Numeric validation is open: see design doc 22.5 and 22.13.
def MNCore_GemmTileOp : MNCore_Op<"gemm.tile", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_PEState>]>]> {
  let summary = "C = A(8x4) * B(4x4) + C0 on one matrix face";
  let arguments = (ins I64Attr:$face, AnyFixedVectorOfNonZeroRank:$a_lo,
                       AnyFixedVectorOfNonZeroRank:$a_hi,
                       AnyFixedVectorOfNonZeroRank:$b,
                       AnyFixedVectorOfNonZeroRank:$c0);
  let results = (outs AnyFixedVectorOfNonZeroRank:$result);
  let hasVerifier = 1;
}

// Layer B memory access: the source is an L1BM address token and the result
// occupies a PE slot assigned by llc's Layer B pass. This is the mapping for
// Triton `ttg.local_load` / TileLang shared-memory reads.
def MNCore_VecLoadOp : MNCore_Op<"vec.load", [
    MemoryEffects<[MemRead<MNCore_DeviceMem>, MemWrite<MNCore_PEState>]>]> {
  let summary = "load a vector from L1BM into a PE slot with an explicit "
                "distribution mode";
  let arguments = (ins MNCore_L1Mode:$mode,
                       MNCore_TokenType:$source,
                       MNCore_TokenType:$source_hi,
                       DefaultValuedAttr<I64Attr, "-1">:$zmask,
                       DefaultValuedAttr<I64Attr, "-1">:$wmask);
  let results = (outs AnyFixedVectorOfNonZeroRank:$result);
  let hasVerifier = 1;
}

// PE -> L1BM writeback. The collect forms are l1bmm@ (MAB) and l1bmm4@
// (MAB4); a broadcast store is undefined, and l1bmd collect still needs the
// mabdiff operand.
def MNCore_VecStoreOp : MNCore_Op<"vec.store", [
    MemoryEffects<[MemRead<MNCore_PEState>, MemWrite<MNCore_DeviceMem>]>]> {
  let summary = "store a PE vector into L1BM with an explicit distribution "
                "mode";
  let arguments = (ins AnyFixedVectorOfNonZeroRank:$value,
                       MNCore_L1Mode:$mode,
                       MNCore_TokenType:$dest,
                       MNCore_TokenType:$dest_hi,
                       I64Attr:$set);
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_MNCORE_IR_MNCOREOPS_TD
