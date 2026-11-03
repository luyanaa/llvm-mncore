//===- MNCoreDialect.td - MN-Core 2 dialect definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the `mncore` dialect: the device-semantics entry point for MN-Core 2
// frontends (Triton, TileLang, ...).
//
// The dialect is deliberately a *thin, verifiable* layer over the frozen
// `llvm.mncore.*` intrinsic ABI documented in
// llvm/lib/Target/MNCore/docs/mncore_intrinsics_design.md:
//
//   * token builders mirror llvm.mncore.peop/lbop/matop/maskop/mvop/mvparams;
//   * instruction ops mirror one PE instruction expression each;
//   * typed ops mirror the Layer B (typed vector/matrix) intrinsics and are
//     lowered back to Layer B so that the single PE slot allocator in
//     MNCoreLowerTypedIntrinsics.cpp remains the only implementation.
//
// Lowering to LLVM IR is `convert-mncore-to-llvm`, which emits
// `llvm.call_intrinsic` calls.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MNCORE_IR_MNCOREDIALECT_TD
#define MLIR_DIALECT_MNCORE_IR_MNCOREDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// MN-Core 2 dialect
//===----------------------------------------------------------------------===//

def MNCore_Dialect : Dialect {
  let name = "mncore";
  let cppNamespace = "::mlir::mncore";

  let summary = "MN-Core 2 device dialect";
  let description = [{
    `mncore` expresses MN-Core 2 device semantics at two levels.

    **Layer A (tokens and instruction expressions).** A `!mncore.token` is a
    packed PE operand. Tokens are produced only by the token builders
    (`mncore.peop`, `mncore.lbop`, `mncore.matop`, `mncore.maskop`,
    `mncore.mvop`, `mncore.mvparams`) or by `mncore.none`, the explicit
    "operand absent" sentinel (the backend ABI encodes absence as `i64 -1`).
    All token fields are compile-time constants: the .vsm PE operand encoding
    has no general "address from a register" form (see design doc section 11).
    Instruction ops (`mncore.alu`, `mncore.mau`, `mncore.l1b`,
    `mncore.l2b`, `mncore.mv`, `mncore.wait`, `mncore.fence`,
    `mncore.maskdir`) take a fixed number of tokens and correspond one-to-one
    with `llvm.mncore.*` instruction intrinsics.

    **Layer B (typed values).** `mncore.vec.*` and `mncore.mat.*` operate on
    `<2 x f64>`, `<4 x f32>`, or `<8 x f16>` values. They lower to the Layer B
    `llvm.mncore.*` intrinsics, so PE slot allocation is performed by the
    existing `mncore-lower-typed-intrinsics` pass inside llc, not here.

    The dialect is a *contract*: it does not change the backend ABI. Numeric
    opcode values are frozen in llvm/include/llvm/Target/MNCore/MNCoreIntrinsicOps.h.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
  ];

  // Types are printed as !mncore.token.
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class MNCore_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MNCore_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MNCore_TokenType : MNCore_Type<"Token", "token"> {
  let summary = "packed MN-Core PE operand token";
  let description = [{
    An opaque, compile-time-packed PE operand. The backend ABI represents the
    same value as `i64`; `convert-mncore-to-llvm` maps this type to `i64`.

    Tokens are produced only by the token-builder ops and consumed only by
    instruction ops. They are not arithmetic values: do not add, select, or
    otherwise manipulate them.
  }];
}

#endif // MLIR_DIALECT_MNCORE_IR_MNCOREDIALECT_TD
