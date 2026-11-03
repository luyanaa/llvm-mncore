//===- MNCoreToLLVM.cpp - MN-Core dialect to LLVM conversion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers `mncore` to `llvm.call_intrinsic` calls:
//
//   * token builders -> llvm.mncore.peop/lbop/matop/maskop/mvop/mvparams,
//     `mncore.none` -> i64 -1;
//   * instruction ops -> the matching llvm.mncore.* instruction intrinsic;
//   * typed ops -> the Layer B intrinsic (llvm.mncore.fma.v4f32, ...), so the
//     PE slot allocator in llc stays the only implementation.
//
// The pass also converts func/arith to LLVM so that the result can be fed to
// `mlir-translate --mlir-to-llvmir` and then `llc -mtriple=mncore2`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MNCoreToLLVM/MNCoreToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MNCore/IR/MNCoreOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMNCORETOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value i64Const(ConversionPatternRewriter &rewriter, Location loc,
                      int64_t value) {
  return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                           rewriter.getI64IntegerAttr(value));
}

static std::string vectorSuffix(Type type) {
  auto vec = cast<VectorType>(type);
  Type element = vec.getElementType();
  if (element.isF64())
    return "v2f64";
  if (element.isF32())
    return "v4f32";
  return "v8f16";
}

//===----------------------------------------------------------------------===//
// Token builders
//===----------------------------------------------------------------------===//

struct NoneOpLowering : OpConversionPattern<mncore::NoneOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::NoneOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, i64Const(rewriter, op.getLoc(), -1));
    return success();
  }
};

struct PeOpLowering : OpConversionPattern<mncore::PeOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::PeOpOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 5> args = {
        i64Const(rewriter, loc, op.getSpace()),
        i64Const(rewriter, loc, op.getAddr()),
        i64Const(rewriter, loc, op.getWidth()),
        i64Const(rewriter, loc, op.getFlags()),
        i64Const(rewriter, loc, op.getAux()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(), rewriter.getStringAttr("llvm.mncore.peop"),
        args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LbOpLowering : OpConversionPattern<mncore::LbOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::LbOpOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 4> args = {
        i64Const(rewriter, loc, op.getMem()),
        i64Const(rewriter, loc, op.getAddr()),
        i64Const(rewriter, loc, op.getWidth()),
        i64Const(rewriter, loc, op.getFlags()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(), rewriter.getStringAttr("llvm.mncore.lbop"),
        args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct MatOpLowering : OpConversionPattern<mncore::MatOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MatOpOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 4> args = {
        i64Const(rewriter, loc, op.getFace()),
        i64Const(rewriter, loc, op.getRow()),
        i64Const(rewriter, loc, op.getWidth()),
        i64Const(rewriter, loc, op.getFlags()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(), rewriter.getStringAttr("llvm.mncore.matop"),
        args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct MaskOpLowering : OpConversionPattern<mncore::MaskOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MaskOpOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 3> args = {
        i64Const(rewriter, loc, op.getKind()),
        i64Const(rewriter, loc, op.getValue()),
        i64Const(rewriter, loc, op.getFlags()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(), rewriter.getStringAttr("llvm.mncore.maskop"),
        args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct MvOpLowering : OpConversionPattern<mncore::MvOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MvOpOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 3> args = {
        i64Const(rewriter, loc, op.getKind()),
        i64Const(rewriter, loc, op.getAddr()),
        i64Const(rewriter, loc, op.getFlags()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(), rewriter.getStringAttr("llvm.mncore.mvop"),
        args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct MvParamsLowering : OpConversionPattern<mncore::MvParamsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MvParamsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 4> args = {
        i64Const(rewriter, loc, op.getSize()),
        i64Const(rewriter, loc, op.getTag()),
        i64Const(rewriter, loc, op.getNd()),
        i64Const(rewriter, loc, op.getPrio()),
    };
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getI64Type(),
        rewriter.getStringAttr("llvm.mncore.mvparams"), args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Instruction expressions
//===----------------------------------------------------------------------===//

struct AluLowering : OpConversionPattern<mncore::AluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::AluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getOpcode())),
        i64Const(rewriter, loc, op.getPrec()),
        i64Const(rewriter, loc, op.getOpt()),
    };
    llvm::append_range(args, adaptor.getTokens());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr("llvm.mncore.alu"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MauLowering : OpConversionPattern<mncore::MauOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MauOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getOpcode())),
        i64Const(rewriter, loc, op.getPrec()),
    };
    llvm::append_range(args, adaptor.getTokens());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr("llvm.mncore.mau"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct L1BLowering : OpConversionPattern<mncore::L1BOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::L1BOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getOpcode())),
        i64Const(rewriter, loc, op.getDir()),
    };
    llvm::append_range(args, adaptor.getTokens());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr("llvm.mncore.l1b"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct L2BLowering : OpConversionPattern<mncore::L2BOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::L2BOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getOpcode())),
    };
    llvm::append_range(args, adaptor.getTokens());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr("llvm.mncore.l2b"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MvLowering : OpConversionPattern<mncore::MvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getOpcode())),
    };
    llvm::append_range(args, adaptor.getTokens());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr("llvm.mncore.mv"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaLowering : OpConversionPattern<mncore::DmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::DmaOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i64 = rewriter.getI64Type();
    auto intr = [&](StringRef n) { return rewriter.getStringAttr(n); };
    auto c = [&](int64_t v) { return i64Const(rewriter, loc, v); };
    auto callI64 = [&](StringRef n, ValueRange args) -> Value {
      return rewriter
          .create<LLVM::CallIntrinsicOp>(loc, i64, intr(n), args)
          .getResult(0);
    };
    auto callVoid = [&](StringRef n, ValueRange args) {
      rewriter.create<LLVM::CallIntrinsicOp>(loc, intr(n), args);
    };

    Value flags = c(op.getMvFlags());
    Value src = callI64(
        "llvm.mncore.mvop",
        {c(op.getSrcSpace()), c(op.getSrcAddr()), flags});
    Value dst = callI64("llvm.mncore.mvop",
                        {c(2), c(op.getL2Addr()), flags});
    Value params = callI64("llvm.mncore.mvparams",
                           {c(op.getSize()), c(op.getTag()), c(-1), c(-1)});
    callVoid("llvm.mncore.mv", {c(0), params, src, dst});

    Value l2 = callI64("llvm.mncore.lbop",
                       {c(1), c(op.getL2Addr()), c(1), c(0)});
    Value l1 = callI64("llvm.mncore.lbop",
                       {c(0), c(op.getL1Addr()), c(1), c(0)});
    callVoid("llvm.mncore.l2b", {c(0), c(0), l2, l1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitLowering : OpConversionPattern<mncore::WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::WaitOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 1> args = {
        i64Const(rewriter, op.getLoc(), op.getTag())};
    rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), rewriter.getStringAttr("llvm.mncore.wait"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct FenceLowering : OpConversionPattern<mncore::FenceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::FenceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), rewriter.getStringAttr("llvm.mncore.fence"), ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MaskDirLowering : OpConversionPattern<mncore::MaskDirOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MaskDirOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 1> args = {
        i64Const(rewriter, op.getLoc(), op.getSpec())};
    rewriter.create<LLVM::CallIntrinsicOp>(
        op.getLoc(), rewriter.getStringAttr("llvm.mncore.maskdir"), args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Typed vector/matrix operations
//===----------------------------------------------------------------------===//

template <typename OpT>
struct TypedOpLowering : OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = this->getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");

    SmallVector<Value> args;
    if (failed(getOperands(op, adaptor, rewriter, args)))
      return failure();

    std::string name = getBaseName().str() + vectorSuffix(op.getType());
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, resultType, rewriter.getStringAttr(name), args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }

protected:
  virtual StringRef getBaseName() const = 0;
  virtual LogicalResult getOperands(OpT op, typename OpT::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    SmallVectorImpl<Value> &args) const = 0;
};

struct VecFmaLowering : TypedOpLowering<mncore::VecFmaOp> {
  using Base = TypedOpLowering<mncore::VecFmaOp>;
  using Base::Base;

protected:
  StringRef getBaseName() const override { return "llvm.mncore.fma."; }
  LogicalResult getOperands(mncore::VecFmaOp op,
                            OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &args) const override {
    args.clear();
    args.push_back(adaptor.getA());
    args.push_back(adaptor.getB());
    args.push_back(adaptor.getC());
    args.push_back(i64Const(rewriter, op.getLoc(), op.getZmask()));
    args.push_back(i64Const(rewriter, op.getLoc(), op.getWmask()));
    return success();
  }
};

template <typename OpT>
struct VecBinaryLowering : TypedOpLowering<OpT> {
  using TypedOpLowering<OpT>::TypedOpLowering;

protected:
  LogicalResult getOperands(OpT op, typename OpT::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &args) const override {
    args.clear();
    args.push_back(adaptor.getLhs());
    args.push_back(adaptor.getRhs());
    args.push_back(i64Const(rewriter, op.getLoc(), op.getZmask()));
    args.push_back(i64Const(rewriter, op.getLoc(), op.getWmask()));
    return success();
  }
};

struct VecMulLowering : VecBinaryLowering<mncore::VecMulOp> {
  using Base = VecBinaryLowering<mncore::VecMulOp>;
  using Base::Base;
  StringRef getBaseName() const override { return "llvm.mncore.mul."; }
};
struct VecAddLowering : VecBinaryLowering<mncore::VecAddOp> {
  using Base = VecBinaryLowering<mncore::VecAddOp>;
  using Base::Base;
  StringRef getBaseName() const override { return "llvm.mncore.add."; }
};
struct VecMaxLowering : VecBinaryLowering<mncore::VecMaxOp> {
  using Base = VecBinaryLowering<mncore::VecMaxOp>;
  using Base::Base;
  StringRef getBaseName() const override { return "llvm.mncore.max."; }
};
struct VecMinLowering : VecBinaryLowering<mncore::VecMinOp> {
  using Base = VecBinaryLowering<mncore::VecMinOp>;
  using Base::Base;
  StringRef getBaseName() const override { return "llvm.mncore.min."; }
};

struct MatVecLowering : TypedOpLowering<mncore::MatVecOp> {
  using Base = TypedOpLowering<mncore::MatVecOp>;
  using Base::Base;

protected:
  StringRef getBaseName() const override { return "llvm.mncore.matvec."; }
  LogicalResult getOperands(mncore::MatVecOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &args) const override {
    args.clear();
    args.push_back(i64Const(rewriter, op.getLoc(), op.getFace()));
    args.push_back(adaptor.getX());
    args.push_back(adaptor.getY());
    return success();
  }
};

struct VecLoadLowering : OpConversionPattern<mncore::VecLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::VecLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");
    SmallVector<Value> args = {
        i64Const(rewriter, loc, static_cast<int64_t>(op.getMode())),
        adaptor.getSource(),
        i64Const(rewriter, loc, op.getZmask()),
        i64Const(rewriter, loc, op.getWmask()),
    };
    std::string name = "llvm.mncore.l1load." + vectorSuffix(op.getType());
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, resultType, rewriter.getStringAttr(name), args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct MatWriteLowering : OpConversionPattern<mncore::MatWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MatWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> args = {
        i64Const(rewriter, loc, op.getFace()),
        i64Const(rewriter, loc, op.getRow()),
        adaptor.getSrc(),
    };
    std::string name =
        "llvm.mncore.matwrite." + vectorSuffix(op.getSrc().getType());
    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, rewriter.getStringAttr(name), args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MatReadLowering : OpConversionPattern<mncore::MatReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mncore::MatReadOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result type is not convertible");
    SmallVector<Value> args = {
        i64Const(rewriter, loc, op.getFace()),
        i64Const(rewriter, loc, op.getRow()),
    };
    std::string name =
        "llvm.mncore.matread." + vectorSuffix(op.getResult().getType());
    auto call = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, resultType, rewriter.getStringAttr(name), args);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct MNCoreToLLVMPass
    : public impl::ConvertMNCoreToLLVMPassBase<MNCoreToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    LLVMTypeConverter converter(ctx);
    converter.addConversion([ctx](mncore::TokenType) -> Type {
      return IntegerType::get(ctx, 64);
    });

    RewritePatternSet patterns(ctx);
    patterns
        .add<NoneOpLowering, PeOpLowering, LbOpLowering, MatOpLowering,
             MaskOpLowering, MvOpLowering, MvParamsLowering,
             AluLowering, MauLowering, L1BLowering, L2BLowering, MvLowering,
             WaitLowering, FenceLowering, MaskDirLowering, DmaLowering,
             VecFmaLowering, VecMulLowering, VecAddLowering, VecMaxLowering,
             VecMinLowering, MatVecLowering, MatWriteLowering,
             MatReadLowering, VecLoadLowering>(converter, ctx);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<mncore::MNCoreDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // llc reads the triple from the module; make the device target explicit so
    // `mlir-translate` output can be piped straight into `llc`.
    if (!module->hasAttr(LLVM::LLVMDialect::getTargetTripleAttrName()))
      module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                      StringAttr::get(ctx, "mncore2"));
  }
};

} // namespace

