//===- MNCoreISelLowering.h - MN-Core2 DAG Lowering Interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MN-Core2 target lowering. The device executes a single straight-line
// instruction stream per board with no branching and no calls; the
// lowering therefore accepts only a small scalar subset of IR and
// rejects everything else explicitly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREISELLOWERING_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREISELLOWERING_H

#include "MNCore.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class MNCoreSubtarget;
class MNCoreTargetMachine;

class MNCoreTargetLowering : public TargetLowering {
public:
  explicit MNCoreTargetLowering(const TargetMachine &TM,
                                const MNCoreSubtarget &Subtarget);

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  MVT getRegisterTypeForCallingConv(LLVMContext &Context,
                                    CallingConv::ID CC,
                                    EVT VT) const override;
  unsigned getNumRegistersForCallingConv(
      LLVMContext &Context, CallingConv::ID CC, EVT VT) const override;

  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override {
    // A kernel's `ret` always lowers to the RET pseudo marker.
    return true;
  }

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      const SDLoc &dl, SelectionDAG &DAG) const override;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override {
    llvm_unreachable("MN-Core2 has no calls");
  }

private:
  const MNCoreSubtarget &Subtarget;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREISELLOWERING_H
