//===- MNCoreLowerTypedIntrinsics.cpp - Layer B -> Layer A ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Layer B of the mncore intrinsic ABI (see
// llvm/lib/Target/MNCore/docs/mncore_intrinsics_design.md). This pass
// performs PE slot allocation and rewrites typed vector/matrix intrinsics
// into Layer A tokens plus instruction intrinsic calls.
//
// Coverage:
//   vector: fma / mul / add / max / min for <2 x double>, <4 x float>,
//           <8 x half>;
//   matrix: matvec / matwrite / matread for the same three element types.
//
// Slot allocation (v2, still deliberately simple but constraint-aware):
//   * 2-longword slots. The accessed region of a 2-longword operand is 16
//     words (4 cycles x 4 words), so slots are 16 words apart — a smaller
//     step makes neighbouring slots overlap and assemble3 reports a data
//     race. Slots rotate over GRF0/GRF1/LM0/LM1 because one address field
//     per space exists: two distinct same-space inputs in one expression
//     would be a field conflict;
//   * a value's slot is released after its last use and reused (linear-scan
//     liveness, no cross-block analysis);
//   * a destination takes a space not read by the same expression;
//   * vector function arguments take the first slots in argument order.
//
// Masks on the typed vector ops must be the constant 0 (unwired).
//
//===----------------------------------------------------------------------===//

#include "MNCore.h"
#include "llvm/Target/MNCore/MNCoreIntrinsicOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

enum class Op { FMA, MUL, ADD, MAX, MIN, MATVEC, MATWRITE, MATREAD,
                L1LOAD };
enum class Prec { D, F, H };

class MNCoreLowerTypedIntrinsics : public FunctionPass {
public:
  static char ID;
  MNCoreLowerTypedIntrinsics() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;

private:
  static bool parseName(StringRef Name, Op &O, Prec &P);
  static bool classify(Type *VT, Prec &P);
};

} // end anonymous namespace

bool MNCoreLowerTypedIntrinsics::classify(Type *VT, Prec &P) {
  auto *Vec = dyn_cast<FixedVectorType>(VT);
  if (!Vec)
    return false;
  unsigned N = Vec->getNumElements();
  Type *E = Vec->getElementType();
  if (E->isDoubleTy() && N == 2) { P = Prec::D; return true; }
  if (E->isFloatTy() && N == 4) { P = Prec::F; return true; }
  if (E->isHalfTy() && N == 8) { P = Prec::H; return true; }
  return false;
}

bool MNCoreLowerTypedIntrinsics::parseName(StringRef Name, Op &O, Prec &P) {
  // llvm.mncore.<op>.<type>
  if (!Name.starts_with("llvm.mncore."))
    return false;
  Name = Name.drop_front(12);
  StringRef OpName = Name.take_front(Name.find('.'));
  StringRef Ty = Name.drop_front(OpName.size() + 1);
  if (OpName == "fma") O = Op::FMA;
  else if (OpName == "mul") O = Op::MUL;
  else if (OpName == "add") O = Op::ADD;
  else if (OpName == "max") O = Op::MAX;
  else if (OpName == "min") O = Op::MIN;
  else if (OpName == "matvec") O = Op::MATVEC;
  else if (OpName == "matwrite") O = Op::MATWRITE;
  else if (OpName == "matread") O = Op::MATREAD;
  else if (OpName == "l1load") O = Op::L1LOAD;
  else return false;
  if (Ty == "v2f64") { P = Prec::D; return true; }
  if (Ty == "v4f32") { P = Prec::F; return true; }
  if (Ty == "v8f16") { P = Prec::H; return true; }
  return false;
}

bool MNCoreLowerTypedIntrinsics::runOnFunction(Function &F) {
  Module &M = *F.getParent();
  LLVMContext &C = F.getContext();
  Type *I64 = Type::getInt64Ty(C);

  struct CallInfo {
    CallInst *CI;
    Op O;
    Prec P;
  };
  SmallVector<CallInfo, 16> Calls;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      Op O;
      Prec P;
      if (parseName(Callee->getName(), O, P))
        Calls.push_back({CI, O, P});
    }
  if (Calls.empty())
    return false;

  FunctionType *PeopFT =
      FunctionType::get(I64, {I64, I64, I64, I64, I64}, false);
  FunctionCallee Peop = M.getOrInsertFunction("llvm.mncore.peop", PeopFT);
  FunctionType *MauFT = FunctionType::get(
      Type::getVoidTy(C),
      {I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64}, false);
  FunctionCallee Mau = M.getOrInsertFunction("llvm.mncore.mau", MauFT);
  FunctionType *AluFT = FunctionType::get(
      Type::getVoidTy(C),
      {I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64}, false);
  FunctionCallee Alu = M.getOrInsertFunction("llvm.mncore.alu", AluFT);
  FunctionType *L1bFT = FunctionType::get(
      Type::getVoidTy(C), {I64, I64, I64, I64, I64, I64, I64}, false);
  FunctionCallee L1b = M.getOrInsertFunction("llvm.mncore.l1b", L1bFT);

  struct Slot {
    unsigned Space;
    uint64_t Addr;
  };
  DenseMap<Value *, Slot> Slots;
  unsigned NextIdx = 0;
  unsigned ResultIdx = 0;
  uint64_t Bump[5] = {0, 0, 0, 0, 0};
  SmallVector<uint64_t, 16> Free[5];
  auto Alloc = [&](unsigned Wanted) -> Slot {
    Slot S;
    if (Wanted >= 1 && Wanted <= 4 && !Free[Wanted].empty()) {
      S = {Wanted, Free[Wanted].pop_back_val()};
      return S;
    }
    unsigned Space = Wanted ? Wanted : 1 + NextIdx % 4;
    if (!Wanted)
      ++NextIdx;
    S = {Space, Bump[Space]};
    Bump[Space] += 16; // one 2-longword region

    if (Bump[Space] > 256)
      report_fatal_error("MN-Core2 Layer B: exhausted PE slots");
    return S;
  };
  auto Release = [&](Value *V) {
    auto It = Slots.find(V);
    if (It != Slots.end())
      Free[It->second.Space].push_back(It->second.Addr);
  };
  auto SlotOf = [&](Value *V) -> Slot {
    auto It = Slots.find(V);
    if (It != Slots.end())
      return It->second;
    if (!isa<Argument>(V))
      report_fatal_error(
          "MN-Core2 Layer B: operand is not a typed value or kernel argument");
    Slot S = Alloc(0);
    Slots[V] = S;
    return S;
  };

  for (Argument &A : F.args())
    if (A.getType()->isVectorTy())
      SlotOf(&A);

  // Contract: a typed result may only be consumed by other typed calls.
  // Anything else (return value, store, plain arithmetic) would be
  // silently dropped when the call is erased.
  for (CallInfo &Info : Calls) {
    CallInst *CI = Info.CI;
    if (!CI->getType()->isVectorTy())
      continue;
    for (User *U : CI->users()) {
      auto *UC = dyn_cast<CallInst>(U);
      Op O;
      Prec P;
      if (!UC || !UC->getCalledFunction() ||
          !parseName(UC->getCalledFunction()->getName(), O, P))
        report_fatal_error(
            "MN-Core2 Layer B: a typed value has a non-typed use "
            "(unsupported ABI contract)");
    }
  }

  // Last-use analysis over the typed call sequence (linear, same block or
  // across blocks in program order).
  DenseMap<Value *, int> LastUse;
  for (int I = 0; I < (int)Calls.size(); ++I)
    for (Value *Op : Calls[I].CI->args())
      if (Op->getType()->isVectorTy())
        LastUse[Op] = I;

  // Assign result slots in order (destination space is chosen at rewrite
  // time from the spaces not read by the call).
  for (CallInfo &CI : Calls)
    if (CI.CI->getType()->isVectorTy())
      ; // allocated during rewriting

  auto VecArgs = [](CallInst *CI) {
    SmallVector<Value *, 3> V;
    for (Value *Op : CI->args())
      if (Op->getType()->isVectorTy())
        V.push_back(Op);
    return V;
  };
  auto SpaceOf = [&](Value *V) { return Slots[V].Space; };

  for (int I = 0; I < (int)Calls.size(); ++I) {
    CallInst *CI = Calls[I].CI;
    Op O = Calls[I].O;
    Prec P = Calls[I].P;
    SmallVector<Value *, 3> VA = VecArgs(CI);
    unsigned NA = VA.size(); // mask operands follow the vector ones
    // Which vector operands are read, and how many are sources.
    unsigned NSrc = 0;
    if (O == Op::FMA) NSrc = 3;
    else if (O == Op::MUL || O == Op::ADD || O == Op::MAX || O == Op::MIN)
      NSrc = 2;
    else if (O == Op::MATVEC) NSrc = 2;
    else if (O == Op::MATWRITE) NSrc = 1;
    else NSrc = 0;
    // l1load takes (lbop, zmask, wmask): the masks follow the address token,
    // not a vector operand.
    unsigned MaskBase = (O == Op::L1LOAD) ? 1 : NA;
    SmallVector<Value *, 3> Src(VA.begin(), VA.begin() + NSrc);

    // Pick a destination space. A GRF write may share a space with a read
    // (separate fields), but an LM write must not overlap a read of that
    // LM space in the same expression (3.6.4 same-region rule). Rotate
    // result spaces so that values later read together do not collide.
    bool HasRes = CI->getType()->isVectorTy();
    Slot DS{0, 0};
    if (HasRes) {
      unsigned Preferred = 1 + (ResultIdx++ % 4);
      auto ReadIn = [&](unsigned S) {
        for (Value *V : Src)
          if (SpaceOf(V) == S)
            return true;
        return false;
      };
      unsigned DstSpace = 0;
      for (unsigned I2 = 0; I2 < 4; ++I2) {
        unsigned C = 1 + ((Preferred - 1 + I2) % 4);
        bool IsLM = (C == 3 || C == 4);
        if (IsLM && ReadIn(C))
          continue;
        DstSpace = C;
        break;
      }
      if (!DstSpace)
        report_fatal_error(
            "MN-Core2 Layer B: no free PE space for the destination");
      DS = Alloc(DstSpace);
    }
    // Same-space reads must be the same slot (one address field per space).
    for (unsigned A = 0; A < Src.size(); ++A)
      for (unsigned B = A + 1; B < Src.size(); ++B)
        if (SpaceOf(Src[A]) == SpaceOf(Src[B]) &&
            Slots[Src[A]].Addr != Slots[Src[B]].Addr)
          report_fatal_error(
              "MN-Core2 Layer B: two inputs share a PE space at different "
              "addresses");
    if (HasRes)
      Slots[CI] = DS;

    IRBuilder<> B(CI);
    auto Tok = [&](Value *V) -> Value * {
      const Slot &S = Slots[V];
      Value *Args[5] = {B.getInt64(S.Space), B.getInt64(S.Addr),
                        B.getInt64(2), B.getInt64(8), B.getInt64(0)};
      return B.CreateCall(Peop, Args);
    };
    auto TokSlot = [&](const Slot &S) -> Value * {
      Value *Args[5] = {B.getInt64(S.Space), B.getInt64(S.Addr),
                        B.getInt64(2), B.getInt64(8), B.getInt64(0)};
      return B.CreateCall(Peop, Args);
    };
    auto ConstArg = [&](unsigned I) -> uint64_t {
      auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(I));
      if (!C)
        report_fatal_error("MN-Core2 Layer B: face/row must be constant");
      return C->getZExtValue();
    };
    // Typed vector ops take (zmask, wmask) as their last two operands;
    // the constant 0 means "unmasked" and is mapped to Layer A's -1
    // sentinel. Non-constant mask tokens are forwarded as-is (isel folds
    // them).
    auto MaskVal = [&](unsigned I) -> Value * {
      Value *V = CI->getArgOperand(I);
      if (auto *UC = dyn_cast<CallInst>(V)) {
        Function *MF = UC->getCalledFunction();
        if (!MF || MF->getName() != "llvm.mncore.maskop")
          report_fatal_error(
              "MN-Core2 Layer B: mask must be a constant or maskop token");
      }
      if (auto *C = dyn_cast<ConstantInt>(V)) {
        if (C->isZero())
          return B.getInt64(-1);
        return B.getInt64(C->getSExtValue());
      }
      return V;
    };

    uint64_t MauId = 0, AluId = 0;
    bool UseAlu = false;
    switch (O) {
    case Op::FMA:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DVFMA : P == Prec::F ? mncore::MAUOp::FVFMA : mncore::MAUOp::HVFMA);
      break;
    case Op::MUL:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DVMUL : P == Prec::F ? mncore::MAUOp::FVMUL : mncore::MAUOp::HVMUL);
      break;
    case Op::ADD:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DVADD : P == Prec::F ? mncore::MAUOp::FVADD : mncore::MAUOp::HVADD);
      break;
    case Op::MAX: // Layer A alu ids: 1=fmax, 10=dmax, 12=hmax
      UseAlu = true;
      AluId = (uint64_t)(P == Prec::D ? mncore::ALUOp::DMax : P == Prec::F ? mncore::ALUOp::FMax : mncore::ALUOp::HMax);
      break;
    case Op::MIN: // Layer A alu ids: 0=fmin, 11=dmin, 13=hmin
      UseAlu = true;
      AluId = (uint64_t)(P == Prec::D ? mncore::ALUOp::DMin : P == Prec::F ? mncore::ALUOp::FMin : mncore::ALUOp::HMin);
      break;
    case Op::MATVEC:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DMatVec : P == Prec::F ? mncore::MAUOp::FMatVec : mncore::MAUOp::HMatVec);
      break;
    case Op::MATWRITE:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DMWrite : P == Prec::F ? mncore::MAUOp::FMWrite : mncore::MAUOp::HMWrite);
      break;
    case Op::MATREAD:
      MauId = (uint64_t)(P == Prec::D ? mncore::MAUOp::DMRead : P == Prec::F ? mncore::MAUOp::FMRead : mncore::MAUOp::HMRead);
      break;
    case Op::L1LOAD:
      break;
    }

    if (UseAlu) {
      Value *SX = Tok(Src[0]);
      Value *SY = Tok(Src[1]);
      Value *D = TokSlot(DS);
      Value *Args[11] = {B.getInt64(AluId), B.getInt64(0), B.getInt64(0),
                         SX, SY, D, B.getInt64(-1), B.getInt64(-1),
                         B.getInt64(-1), MaskVal(MaskBase), MaskVal(MaskBase + 1)};
      B.CreateCall(Alu, Args);
    } else if (O == Op::MATWRITE || O == Op::MATREAD) {
      Value *Mat = B.getInt64((ConstArg(0) & 1) |
                              ((ConstArg(1) & 0xff) << 1));
      Value *Args[12] = {B.getInt64(MauId), B.getInt64(0), Mat,
                         O == Op::MATWRITE ? Tok(Src[0]) : B.getInt64(-1),
                         B.getInt64(-1), B.getInt64(-1),
                         O == Op::MATREAD ? TokSlot(DS) : B.getInt64(-1),
                         B.getInt64(-1), B.getInt64(-1), B.getInt64(-1),
                         B.getInt64(-1), B.getInt64(-1)};
      B.CreateCall(Mau, Args);
    } else if (O == Op::MATVEC) {
      Value *Mat = B.getInt64(ConstArg(0) & 1);
      Value *Args[12] = {B.getInt64(MauId), B.getInt64(0), Mat,
                         Tok(Src[0]), Tok(Src[1]), B.getInt64(-1),
                         TokSlot(DS), B.getInt64(-1), B.getInt64(-1),
                         B.getInt64(-1), B.getInt64(-1), B.getInt64(-1)};
      B.CreateCall(Mau, Args);
    } else if (O == Op::L1LOAD) {
      // (opcode, dir, src, dst, set, zmask, wmask)
      Value *Args[7] = {B.getInt64((uint64_t)mncore::L1BOp::L1BMP),
                        B.getInt64(1), CI->getArgOperand(0), TokSlot(DS),
                        B.getInt64(-1), MaskVal(MaskBase),
                        MaskVal(MaskBase + 1)};
      B.CreateCall(L1b, Args);
    } else {
      // fma / mul / add (MAU vector forms)
      Value *SX = Tok(Src[0]);
      Value *SY = Tok(Src[1]);
      Value *SZ = NSrc == 3 ? Tok(Src[2]) : B.getInt64(-1);
      Value *D = TokSlot(DS);
      Value *Args[12] = {B.getInt64(MauId), B.getInt64(0), B.getInt64(-1),
                         SX, SY, SZ, D, B.getInt64(-1), B.getInt64(-1),
                         B.getInt64(-1), MaskVal(MaskBase), MaskVal(MaskBase + 1)};
      B.CreateCall(Mau, Args);
    }

    // Release operands whose last use is this call. Across multiple basic
    // blocks the linear call order is not a valid liveness order, so slots
    // are conservatively kept (no reuse) in that case.
    if (F.size() == 1) {
      SmallPtrSet<Value *, 4> Released;
      for (Value *V : Src)
        if (LastUse.lookup(V) == I && Released.insert(V).second)
          Release(V);
    }
  }

  for (int I = (int)Calls.size() - 1; I >= 0; --I)
    Calls[I].CI->eraseFromParent();
  return true;
}

char MNCoreLowerTypedIntrinsics::ID = 0;
INITIALIZE_PASS(MNCoreLowerTypedIntrinsics, "mncore-lower-typed-intrinsics",
                "MN-Core2 Layer B -> Layer A", false, false)

FunctionPass *llvm::createMNCoreLowerTypedIntrinsics() {
  return new MNCoreLowerTypedIntrinsics();
}
