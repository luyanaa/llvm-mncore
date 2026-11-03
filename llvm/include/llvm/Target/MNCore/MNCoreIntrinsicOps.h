//===- MNCoreIntrinsicOps.h - MN-Core2 Layer A opcode IDs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable opcode IDs for the Layer A instruction intrinsics
// (llvm.mncore.alu / .mau / .l1b / .l2b / .mv). The numeric values are the
// ABI: they are passed as the first i64 argument of the instruction
// intrinsic and must never be renumbered. New opcodes are appended in a new
// range; an ID that is not listed here is unsupported and rejected by the
// selector.
//
// This header is the public single source of truth shared by SelectionDAG
// isel, the Layer B lowering pass, and the MLIR mncore dialect
// (mlir/lib/Dialect/MNCore/IR/MNCoreOps.cpp). The MLIR dialect mirrors these
// values in MNCoreOps.td; static_asserts in MNCoreOps.cpp reject any drift.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MNCORE_MNCOREINTRINSIC_OPS_H
#define LLVM_TARGET_MNCORE_MNCOREINTRINSIC_OPS_H

#include <cstdint>

namespace llvm {
namespace mncore {

// llvm.mncore.alu(opcode, prec, opt, sx, sy, d0, d1, d2, omr, zmask, wmask)
enum class ALUOp : int64_t {
  FMin = 0,
  FMax = 1,
  IAdd = 2,
  ISub = 3,
  LPassa = 4, // one input, one destination
  DMax = 10,
  DMin = 11,
  HMax = 12,
  HMin = 13,
};

// llvm.mncore.mau(opcode, prec, mat, sx, sy, sz, d0, d1, d2, omr,
//                 zmask, wmask)
enum class MAUOp : int64_t {
  FVFMA = 0,
  FVMUL = 1,
  FVADD = 2,
  FVPASSA = 3,
  FMatVec = 4,
  FMWrite = 5,
  FMRead = 6,
  DVFMA = 10,
  DVMUL = 11,
  DVADD = 12,
  HVFMA = 13,
  HVMUL = 14,
  HVADD = 15,
  DMatVec = 16,
  HMatVec = 17,
  DMWrite = 18,
  HMWrite = 19,
  DMRead = 20,
  HMRead = 21,
};

// llvm.mncore.l1b(opcode, dir, src, dst, set, zmask, wmask)
enum class L1BOp : int64_t {
  L1BMP = 0,
  L1BMM = 1,
  L1BMM4 = 2,
  L1BMD = 3, // one longword per PE; no double-longword form
};

// llvm.mncore.l2b(opcode, set, a, b)
enum class L2BOp : int64_t {
  L2BMB = 0,
};

// llvm.mncore.mv(opcode, params, src, dst)
enum class MVOp : int64_t {
  MVP = 0,
};

} // namespace mncore
} // namespace llvm

#endif // LLVM_TARGET_MNCORE_MNCOREINTRINSIC_OPS_H
