//===- MNCoreOps.cpp - MN-Core 2 dialect operation verifiers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifiers enforce the backend contract at MLIR level so that illegal
// operand combinations are rejected here instead of by assemble3.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MNCore/IR/MNCoreOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Target/MNCore/MNCoreIntrinsicOps.h"

using namespace mlir;
using namespace mlir::mncore;

//===----------------------------------------------------------------------===//
// Opcode ABI sync
//
// MNCoreOps.td mirrors the append-only values in MNCoreIntrinsicOps.h. These
// asserts make any drift a build error in both directions (a renamed value or
// a changed number).
//===----------------------------------------------------------------------===//

namespace {
template <typename T> constexpr int64_t asI64(T value) {
  return static_cast<int64_t>(value);
}

#define MNCORE_ENUM_SYNC(MLIR_ENUM, LLVM_ENUM, NAME)                        \
  static_assert(asI64(::mlir::mncore::MLIR_ENUM::NAME) ==                   \
                    asI64(::llvm::mncore::LLVM_ENUM::NAME),                 \
                #LLVM_ENUM "::" #NAME " is out of sync with MNCoreOps.td")

MNCORE_ENUM_SYNC(ALUOp, ALUOp, FMin);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, FMax);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, IAdd);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, ISub);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, LPassa);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, DMax);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, DMin);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, HMax);
MNCORE_ENUM_SYNC(ALUOp, ALUOp, HMin);

MNCORE_ENUM_SYNC(MAUOp, MAUOp, FVFMA);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FVMUL);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FVADD);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FVPASSA);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FMatVec);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FMWrite);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, FMRead);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DVFMA);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DVMUL);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DVADD);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HVFMA);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HVMUL);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HVADD);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DMatVec);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HMatVec);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DMWrite);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HMWrite);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, DMRead);
MNCORE_ENUM_SYNC(MAUOp, MAUOp, HMRead);

MNCORE_ENUM_SYNC(L1BKind, L1BOp, L1BMP);
MNCORE_ENUM_SYNC(L1BKind, L1BOp, L1BMM);
MNCORE_ENUM_SYNC(L1BKind, L1BOp, L1BMM4);
MNCORE_ENUM_SYNC(L1BKind, L1BOp, L1BMD);

MNCORE_ENUM_SYNC(L2BKind, L2BOp, L2BMB);
MNCORE_ENUM_SYNC(L2BKind, L2BOp, L2BMDC);

MNCORE_ENUM_SYNC(MVOp, MVOp, MVP);

#undef MNCORE_ENUM_SYNC
} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isNoneToken(Value v) { return v.getDefiningOp<NoneOp>() != nullptr; }

static bool isMaskToken(Value v) {
  return isNoneToken(v) || v.getDefiningOp<MaskOpOp>() != nullptr;
}

static std::optional<std::pair<int64_t, int64_t>> peSpaceAddr(Value v) {
  if (auto pe = v.getDefiningOp<PeOpOp>())
    return std::make_pair(static_cast<int64_t>(pe.getSpace()),
                          static_cast<int64_t>(pe.getAddr()));
  return std::nullopt;
}

/// One expression has a single address field per PE space: two *input*
/// operands in the same space must therefore share one address.
static LogicalResult checkSameSpaceSameAddr(Operation *op, ValueRange tokens,
                                            ArrayRef<unsigned> positions) {
  for (unsigned i = 0; i < positions.size(); ++i) {
    for (unsigned j = i + 1; j < positions.size(); ++j) {
      auto a = peSpaceAddr(tokens[positions[i]]);
      auto b = peSpaceAddr(tokens[positions[j]]);
      if (a && b && a->first == b->first && a->second != b->second)
        return op->emitOpError()
               << "inputs address space " << a->first
               << " at different addresses (" << a->second << " vs "
               << b->second
               << "); an expression has one address field per space";
    }
  }
  return success();
}

static bool isSupportedVectorType(Type type) {
  auto vec = dyn_cast<VectorType>(type);
  if (!vec || vec.getRank() != 1)
    return false;
  int64_t n = vec.getNumElements();
  Type e = vec.getElementType();
  if (e.isF64())
    return n == 2;
  if (e.isF32())
    return n == 4;
  if (e.isF16())
    return n == 8;
  return false;
}

static LogicalResult verifyVectorType(Operation *op, Type type) {
  if (isSupportedVectorType(type))
    return success();
  return op->emitOpError()
         << "expects <2 x f64>, <4 x f32>, or <8 x f16>, got " << type;
}

//===----------------------------------------------------------------------===//
// Token builders
//===----------------------------------------------------------------------===//

LogicalResult PeOpOp::verify() {
  uint32_t space = getSpace();
  if (space < 1 || space > 5)
    return emitOpError() << "space must be 1..5 (1=GRF0 2=GRF1 3=LM0 4=LM1 "
                            "5=T), got " << space;

  uint32_t width = getWidth();
  if (width > 3)
    return emitOpError() << "width must be 0..3, got " << width;

  uint32_t flags = getFlags();
  if (flags & ~uint32_t{0x1ff})
    return emitOpError() << "unknown flag bits in " << flags;

  if (flags & 64)
    return emitOpError()
           << "flat (4-address) operands are not modelled by the backend";

  if ((flags & 256) && (space != 3 && space != 4))
    return emitOpError()
           << "the LM base-address-register flag requires space LM0/LM1";

  if ((flags & 256) && getAddr() != 0)
    return emitOpError()
           << "a base-address-register operand has no address field";

  if (flags & 32) {
    if (space != 3)
      return emitOpError() << "T-indirect requires space LM0 (3)";
    if (width != 1)
      return emitOpError() << "T-indirect requires longword width (1)";
  }

  uint64_t addr = getAddr();
  if (addr > 0xffff)
    return emitOpError() << "address must fit the 16-bit operand field, got "
                         << addr;

  uint64_t aux = getAux();
  if ((flags & 16) && aux > 7)
    return emitOpError() << "j<madpe> payload must be 0..7, got " << aux;
  if ((flags & 128) && aux > 0xffff)
    return emitOpError() << "v<adri> payload must fit 16 bits, got " << aux;

  return success();
}

LogicalResult LbOpOp::verify() {
  if (getMem() != 0 && getMem() != 1)
    return emitOpError() << "mem must be 0 (L1BM) or 1 (L2BM), got "
                         << getMem();
  if (getWidth() > 2)
    return emitOpError() << "width must be 0..2, got " << getWidth();
  if (getAddr() > 0xffff)
    return emitOpError() << "address must fit 16 bits, got " << getAddr();
  if (getFlags() & ~uint32_t{0x3})
    return emitOpError() << "unknown flag bits in " << getFlags();
  return success();
}

LogicalResult MatOpOp::verify() {
  if (getFace() != 0 && getFace() != 1)
    return emitOpError() << "face must be 0 (X) or 1 (Y), got " << getFace();
  if (getRow() > 0xff)
    return emitOpError() << "row must be 0..255, got " << getRow();
  if (getWidth() > 2)
    return emitOpError() << "width must be 0..2, got " << getWidth();
  if (getFlags() != 0)
    return emitOpError() << "matrix operands have no flag bits, got "
                         << getFlags();
  return success();
}

LogicalResult MaskOpOp::verify() {
  if (getKind() != 0 && getKind() != 1)
    return emitOpError() << "kind must be 0 (fixed pattern) or 1 ($imr), got "
                         << getKind();
  if (getValue() > 0xf)
    return emitOpError() << "value must be 0..15, got " << getValue();
  if (getFlags() & ~uint32_t{0xf})
    return emitOpError() << "unknown flag bits in " << getFlags();
  return success();
}

LogicalResult MvOpOp::verify() {
  if (getKind() > 3)
    return emitOpError() << "kind must be 0..3, got " << getKind();
  if (getAddr() > 0xffffffffULL)
    return emitOpError() << "address must fit 32 bits, got " << getAddr();
  if (getFlags() & ~uint32_t{0xf})
    return emitOpError() << "unknown flag bits in " << getFlags();
  return success();
}

LogicalResult MvParamsOp::verify() {
  if (getSize() > 0xffffffffULL)
    return emitOpError() << "size must fit 32 bits, got " << getSize();
  for (auto [name, value] : {std::pair{"tag", getTag()},
                             std::pair{"nd", getNd()},
                             std::pair{"prio", getPrio()}}) {
    if (value > 0xff)
      return emitOpError() << name << " must be 0..255, got " << value;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Instruction expressions
//===----------------------------------------------------------------------===//

LogicalResult AluOp::verify() {
  auto tokens = getTokens();
  if (tokens.size() != 8)
    return emitOpError() << "expects 8 token operands (sx, sy, d0, d1, d2, "
                            "omr, zmask, wmask), got " << tokens.size();

  if (!isMaskToken(tokens[6]) || !isMaskToken(tokens[7]))
    return emitOpError()
           << "zmask/wmask must be mncore.maskop or mncore.none";

  for (unsigned i : {0u, 1u}) {
    if (isNoneToken(tokens[i]))
      continue;
    if (!tokens[i].getDefiningOp<PeOpOp>())
      return emitOpError() << "input " << i << " must be a PE operand token";
  }

  return checkSameSpaceSameAddr(getOperation(), tokens, {0, 1});
}

LogicalResult MauOp::verify() {
  auto tokens = getTokens();
  if (tokens.size() != 10)
    return emitOpError() << "expects 10 token operands (mat, sx, sy, sz, d0, "
                            "d1, d2, omr, zmask, wmask), got "
                         << tokens.size();

  if (!isNoneToken(tokens[0]) && !tokens[0].getDefiningOp<MatOpOp>())
    return emitOpError() << "mat must be a matrix operand token or "
                            "mncore.none";

  if (!isMaskToken(tokens[8]) || !isMaskToken(tokens[9]))
    return emitOpError()
           << "zmask/wmask must be mncore.maskop or mncore.none";

  for (unsigned i : {1u, 2u, 3u}) {
    if (isNoneToken(tokens[i]))
      continue;
    if (!tokens[i].getDefiningOp<PeOpOp>())
      return emitOpError() << "input " << i << " must be a PE operand token";
  }

  return checkSameSpaceSameAddr(getOperation(), tokens, {1, 2, 3});
}

LogicalResult L1BOp::verify() {
  auto tokens = getTokens();
  if (tokens.size() != 5)
    return emitOpError() << "expects 5 token operands (src, dst, set, zmask, "
                            "wmask), got " << tokens.size();
  if (!isMaskToken(tokens[3]) || !isMaskToken(tokens[4]))
    return emitOpError()
           << "zmask/wmask must be mncore.maskop or mncore.none";
  if (getDir() != 1 && getDir() != 2)
    return emitOpError() << "dir must be 1 (L1B->PE) or 2 (PE->L1B), got "
                         << getDir();
  return success();
}

LogicalResult L2BOp::verify() {
  auto tokens = getTokens();
  if (tokens.size() != 3)
    return emitOpError() << "expects 3 token operands (set, a, b), got "
                         << tokens.size();
  return success();
}

LogicalResult MvOp::verify() {
  auto tokens = getTokens();
  if (tokens.size() != 3)
    return emitOpError() << "expects 3 token operands (params, src, dst), got "
                         << tokens.size();
  return success();
}

LogicalResult DmaOp::verify() {
  if (getSrcSpace() != 0 && getSrcSpace() != 1)
    return emitOpError() << "src_space must be 0 (PDM) or 1 (DRAM), got "
                         << getSrcSpace();
  if (getSize() == 0)
    return emitOpError() << "size must be nonzero";
  if (getSize() > 0xffffffffULL)
    return emitOpError() << "size must fit 32 bits, got " << getSize();
  if (getTag() == 0 || getTag() > 0xff)
    return emitOpError()
           << "tag must be 1..255 (tag 0 is not a valid MV completion tag)";
  for (auto [name, addr] : {std::pair{"src_addr", getSrcAddr()},
                            std::pair{"l2_addr", getL2Addr()},
                            std::pair{"l1_addr", getL1Addr()}}) {
    if (addr > 0xffffffffULL)
      return emitOpError() << name << " must fit 32 bits, got " << addr;
  }
  if (getMvFlags() & ~uint32_t{0xf})
    return emitOpError() << "unknown MV flag bits in " << getMvFlags();
  return success();
}

LogicalResult DmaStoreOp::verify() {
  if (getSize() == 0)
    return emitOpError() << "size must be nonzero";
  if (getSize() > 0xffffffffULL)
    return emitOpError() << "size must fit 32 bits, got " << getSize();
  if (getTag() == 0 || getTag() > 0xff)
    return emitOpError() << "tag must be 1..255";
  for (auto [name, addr] : {std::pair{"l1_addr", getL1Addr()},
                            std::pair{"l2_addr", getL2Addr()},
                            std::pair{"dr_addr", getDrAddr()}}) {
    if (addr > 0xffffffffULL)
      return emitOpError() << name << " must fit 32 bits, got " << addr;
  }
  return success();
}

LogicalResult WaitOp::verify() {
  if (getTag() == 0 || getTag() > 0xff)
    return emitOpError() << "tag must be 1..255 (tag 0 is not a valid MV "
                            "completion tag), got " << getTag();
  return success();
}

//===----------------------------------------------------------------------===//
// Typed vector/matrix operations
//===----------------------------------------------------------------------===//

LogicalResult VecFmaOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecMulOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecAddOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecMaxOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecMinOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult MatVecOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecStoreOp::verify() {
  if (failed(verifyVectorType(getOperation(), getValue().getType())))
    return failure();
  auto lb = getDest().getDefiningOp<LbOpOp>();
  if (!lb)
    return emitOpError() << "dest must be an mncore.lbop token";
  if (lb.getMem() != 0)
    return emitOpError() << "vec.store writes L1BM; lbop.mem must be 0 (L1B)";
  if (getMode() == L1Mode::Broadcast)
    return emitOpError()
           << "a broadcast store is not defined; choose MAB (l1bmm@), "
              "MAB4 (l1bmm4@), or Distributed";
  if (getMode() == L1Mode::Distributed) {
    auto hi = getDestHi().getDefiningOp<LbOpOp>();
    if (!hi)
      return emitOpError()
             << "distributed stores require two longword-plane destinations";
    if (lb.getWidth() != 1 || hi.getWidth() != 1)
      return emitOpError()
             << "distributed store destinations must be 1-longword operands "
                "(l1bmd has no double-longword form, SDM 3.6.8.21)";
    if (lb.getAddr() % 64 != 0)
      return emitOpError() << "the low plane must be 64-longword aligned";
    if (hi.getAddr() != lb.getAddr() + 64)
      return emitOpError()
             << "the high plane must start 64 longwords after the low plane";
    return success();
  }
  if (!getDestHi().getDefiningOp<NoneOp>())
    return emitOpError()
           << "dest_hi must be mncore.none unless the mode is Distributed";
  uint64_t alignment =
      (getMode() == L1Mode::MAB) ? (lb.getWidth() == 2 ? 8 : 4)
                                 : (lb.getWidth() == 2 ? 32 : 16);
  if (lb.getAddr() % alignment != 0)
    return emitOpError() << "address " << lb.getAddr()
                         << " (longwords) is not " << alignment
                         << "-longword aligned for distribution mode "
                         << stringifyL1Mode(getMode());
  if (getSet() > 0xf)
    return emitOpError() << "set must be 0..15, got " << getSet();
  return success();
}

LogicalResult MatWriteOp::verify() {
  return verifyVectorType(getOperation(), getSrc().getType());
}
LogicalResult MatReadOp::verify() {
  return verifyVectorType(getOperation(), getResult().getType());
}
LogicalResult VecLoadOp::verify() {
  if (failed(verifyVectorType(getOperation(), getResult().getType())))
    return failure();
  auto lb = getSource().getDefiningOp<LbOpOp>();
  if (!lb)
    return emitOpError() << "source must be an mncore.lbop token";
  if (lb.getMem() != 0)
    return emitOpError() << "vec.load reads L1BM; lbop.mem must be 0 (L1B)";

  // SDM 3.6.8. Distributed loads are the two-longword plane split: l1bmd has
  // no double-longword form, so the low and high planes are separate
  // 64-longword-aligned L1BM regions.
  if (getMode() == L1Mode::Distributed) {
    auto hi = getSourceHi().getDefiningOp<LbOpOp>();
    if (!hi)
      return emitOpError()
             << "distributed loads require two longword-plane sources "
                "(source and source_hi)";
    if (lb.getWidth() != 1 || hi.getWidth() != 1)
      return emitOpError()
             << "distributed load sources must be 1-longword operands "
                "(l1bmd has no double-longword form, SDM 3.6.8.20)";
    if (lb.getAddr() % 64 != 0)
      return emitOpError() << "the low plane must be 64-longword aligned";
    if (hi.getAddr() != lb.getAddr() + 64)
      return emitOpError()
             << "the high plane must start 64 longwords after the low plane";
    return success();
  }
  if (!getSourceHi().getDefiningOp<NoneOp>())
    return emitOpError()
           << "source_hi must be mncore.none unless the mode is Distributed";

  uint32_t width = lb.getWidth();
  uint64_t addr = lb.getAddr();
  uint64_t alignment = 0;
  switch (getMode()) {
  case L1Mode::Broadcast:
    // l1bmp: no extra alignment. Its double-longword form has non-contiguous
    // reads and range limits that cannot be checked statically here.
    break;
  case L1Mode::MAB:
    alignment = width == 2 ? 8 : 4;
    break;
  case L1Mode::MAB4:
    alignment = width == 2 ? 32 : 16;
    break;
  case L1Mode::Distributed:
    llvm_unreachable("handled above");
  }
  if (alignment && addr % alignment != 0)
    return emitOpError() << "address " << addr << " (longwords) is not "
                         << alignment << "-longword aligned for distribution "
                         << "mode " << stringifyL1Mode(getMode());
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MNCore/IR/MNCoreOps.cpp.inc"
