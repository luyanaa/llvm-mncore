//===- MNCoreFrameLowering.cpp - MN-Core2 frame lowering --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreFrameLowering.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

void MNCoreFrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  // MN-Core2 kernels have no stack: the prologue is empty.
}

void MNCoreFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  // MN-Core2 kernels have no stack: the epilogue is empty.
}
