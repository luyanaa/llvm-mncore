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
  L2BMB = 0,  // L2BM -> L1BM broadcast;            set = @<l1bset>
  L2BMDC = 1, // L1BM -> L2BM collect (all L1Bs);   set = unused
  // L1BM -> L2BM point transfer from one L1B: `l2bm@<l1badr>` (SDM 3.6.7.5).
  // Unlike the collect it copies one L1B's 16-longword-per-cycle stream
  // verbatim, so a replicated (SPMD) L1BM plane lands contiguously in L2BM
  // instead of being interleaved in 8-longword chunks across the 8 L1Bs.
  L2BMPT = 2, // set = <l1badr> (0-7)
};

// llvm.mncore.mv(opcode, params, src, dst)
enum class MVOp : int64_t {
  MVP = 0,
};

inline const char *mvOpcodeName(unsigned Op) {
  switch ((MVOp)Op) {
  case MVOp::MVP:
    return "mvp";
  }
  return "mv?";
}

//===----------------------------------------------------------------------===//
// Relocation records (mncore_kernel_abi_v1.md §13).
//
// A memory operand whose base comes from a kernel buffer argument cannot be an
// immediate: the `.vsm` address field is static. Such an operand is emitted
// with a placeholder address (0) and a *relocation record* that says which
// argument supplies the base and what the static offset is. The record rides
// on the `llvm.mncore.mv.reloc` intrinsic (the frozen `llvm.mncore.mv` ABI is
// unchanged) and becomes, in the asm printer, a `# @mncore.fixup <site>` line
// comment plus an entry in the YAML sidecar block.
//
// Units follow §13.1: the record's `addend_units` is in device address units
// (longword = 8 B for DRAM/PDM/L2BM, word = 4 B for GRF/LM); the host side
// converts bytes to units and must reject a non-integral conversion.
//===----------------------------------------------------------------------===//

// `space` field of a fixup: where the relocated address lives.
enum class RelocSpace : int64_t {
  DRAM = 0,
  PDM = 1,
  L2BM = 2,
};

// Address unit of the relocated field.
enum class RelocUnit : int64_t {
  Longword = 0, // 8 bytes
  Word = 1,     // 4 bytes
};

// Which MV operand carries the relocated address (`operand` field, §13.2).
enum class RelocOperand : int64_t {
  Params = 0,
  Src = 1,
  Dst = 2,
};

constexpr unsigned relocUnitBytes(RelocUnit U) {
  return U == RelocUnit::Word ? 4u : 8u;
}

struct RelocRecord {
  unsigned Arg = 0;      // kernel argument index
  RelocSpace Space = RelocSpace::DRAM;
  RelocUnit Unit = RelocUnit::Longword;
  unsigned WidthBits = 0; // ISA address subfield width, for the bound check
  unsigned Operand = 0;   // RelocOperand
  unsigned Opcode = 0;    // MVOp
  int64_t AddendUnits = 0;
};

// Packed into one i64 intrinsic argument:
//   [7:0]   arg
//   [10:8]  space
//   [11]    unit
//   [19:12] width_bits   (0 is not a valid width: it marks "no fixup")
//   [23:20] operand
//   [31:24] opcode
//   [63:32] addend_units (signed, in `unit`)
constexpr uint64_t packReloc(const RelocRecord &R) {
  return ((uint64_t)(R.Arg & 0xffu)) | ((uint64_t)((unsigned)R.Space & 0x7u) << 8) |
         ((uint64_t)((unsigned)R.Unit & 0x1u) << 11) |
         ((uint64_t)(R.WidthBits & 0xffu) << 12) |
         ((uint64_t)(R.Operand & 0xfu) << 20) |
         ((uint64_t)(R.Opcode & 0xffu) << 24) |
         ((uint64_t)(uint32_t)R.AddendUnits << 32);
}

constexpr RelocRecord unpackReloc(uint64_t V) {
  RelocRecord R;
  R.Arg = (unsigned)(V & 0xffu);
  R.Space = (RelocSpace)((V >> 8) & 0x7u);
  R.Unit = (RelocUnit)((V >> 11) & 0x1u);
  R.WidthBits = (unsigned)((V >> 12) & 0xffu);
  R.Operand = (unsigned)((V >> 20) & 0xfu);
  R.Opcode = (unsigned)((V >> 24) & 0xffu);
  R.AddendUnits = (int64_t)(int32_t)(V >> 32);
  return R;
}

// True when the packed record denotes an actual fixup. `width_bits == 0` is
// the "no fixup" marker: every real address field is at least one bit wide.
constexpr bool hasReloc(uint64_t V) { return ((V >> 12) & 0xffu) != 0; }

} // namespace mncore
} // namespace llvm

#endif // LLVM_TARGET_MNCORE_MNCOREINTRINSIC_OPS_H
