//===- MNCoreISelLowering.cpp - MN-Core2 DAG Lowering Implementation - C++ -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreISelLowering.h"
#include "MNCoreMachineFunctionInfo.h"
#include "MNCoreSelectionDAGInfo.h"
#include "MNCoreSubtarget.h"
#include "MNCoreTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mncore-lower"

MNCoreTargetLowering::MNCoreTargetLowering(const TargetMachine &TM,
                                           const MNCoreSubtarget &STI)
    : TargetLowering(TM, STI), Subtarget(STI) {

  // The only legal scalar integer type is the 64-bit "long word".
  addRegisterClass(MVT::i64, &MNCore::GRFRegClass);

  // Compute the derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  // Everything not listed here must be promoted/custom lowered; the
  // M0/M1 subset keeps scalar integer ops only.
  setOperationAction(ISD::ADD, MVT::i64, Legal);
  setOperationAction(ISD::SUB, MVT::i64, Legal);
  setOperationAction(ISD::AND, MVT::i64, Legal);
  setOperationAction(ISD::OR, MVT::i64, Legal);
  setOperationAction(ISD::XOR, MVT::i64, Legal);

  // No hardware stack: no argument/return value stack slots, no
  // call/return sequences, no varargs.
  setStackPointerRegisterToSaveRestore(Register());

  // Scheduling (and therefore the VLIW packetizer) works on SUnits that
  // carry the per-cycle slot resources described in MNCoreSchedule.td.
  setSchedulingPreference(Sched::VLIW);
}

//===----------------------------------------------------------------------===//
// Register type helpers
//===----------------------------------------------------------------------===//

MVT MNCoreTargetLowering::getRegisterTypeForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT) const {
  // All value traffic is carried in 64-bit long words.
  return MVT::i64;
}

unsigned MNCoreTargetLowering::getNumRegistersForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// Argument / return lowering
//
// MN-Core2 has no call ABI: a kernel is launched once per L1B group and
// parameters are pre-loaded into the GRFs by the host. To keep isel
// simple we model each scalar i64 parameter as if it arrived in a
// dedicated GRF0 slot (grf0_0, grf0_1, ...) and return no value.
//
// This is a placeholder contract for M0/M1; the real argument-passing
// convention is defined by the kernel launch protocol, not by hardware.
//===----------------------------------------------------------------------===//

SDValue MNCoreTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Longword PE operands need an even GRF address (3.6.1.6), so the
  // preloaded parameter slots are 2 apart.
  if (Ins.size() > 128)
    report_fatal_error(
        "MN-Core2 kernel takes too many scalar parameters (>128 even slots)");

  // Map each incoming i64 parameter to a live-in GRF0 slot.
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    EVT ArgVT = Ins[i].ArgVT;
    if (!ArgVT.isSimple() || ArgVT.getSimpleVT().SimpleTy != MVT::i64)
      report_fatal_error(
          "MN-Core2 only accepts i64 kernel parameters");

    MCRegister PhysReg = MNCore::GRF0_0 + 2 * i;
    Register VReg = MRI.createVirtualRegister(&MNCore::GRFRegClass);
    MRI.addLiveIn(PhysReg, VReg);
    InVals.push_back(DAG.getCopyFromReg(Chain, dl, VReg, MVT::i64));
  }

  return Chain;
}

SDValue MNCoreTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
    SelectionDAG &DAG) const {
  // A kernel produces its results by leaving them in the register file;
  // there is no architectural return-value register.  Keep the produced
  // values live by gluing them onto the RET node so they are not
  // dead-code eliminated before the straight-line stream is emitted.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  for (const SDValue &V : OutVals) {
    if (!V)
      continue;
    EVT VT = V.getValueType();
    if (!VT.isSimple() || VT.getSimpleVT().SimpleTy != MVT::i64)
      report_fatal_error("MN-Core2 can only return i64 values");
    Ops.push_back(V);
  }
  return DAG.getNode(MNCoreISD::RET, dl, MVT::Other, Ops);
}

//===----------------------------------------------------------------------===//
// Operation lowering
//===----------------------------------------------------------------------===//

SDValue MNCoreTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unsupported operation for MN-Core2 lowering");
  }
}
