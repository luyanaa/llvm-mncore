//===- MNCoreRegisterInfo.cpp - MN-Core2 Register Information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreRegisterInfo.h"
#include "MNCore.h"
#include "MNCoreSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "MNCoreGenRegisterInfo.inc"

MNCoreRegisterInfo::MNCoreRegisterInfo(const MNCoreSubtarget &sti)
    : MNCoreGenRegisterInfo(/*RA=*/0), STI(sti) {}

BitVector MNCoreRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // Forwarding latches, turnaround register and matrix faces are only
  // touched through implicit operands of the producing instruction
  // expressions; they must never be allocated as virtual registers.
  Reserved.set(MNCore::MAUF);
  Reserved.set(MNCore::ALUF);
  Reserved.set(MNCore::LBF);
  Reserved.set(MNCore::MREADF);
  Reserved.set(MNCore::LBI);
  Reserved.set(MNCore::MATX);
  Reserved.set(MNCore::MATY);

  // Mask register entries 0 and 16-31 are fixed hardware constants.
  Reserved.set(MNCore::OMR_0);
  for (unsigned I = 16; I <= 31; ++I)
    Reserved.set(MNCore::OMR_0 + I);

  return Reserved;
}

Register MNCoreRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // No frame register exists; returning the null register is safe
  // because generic passes treat Register()==0 as "no frame register".
  return Register();
}

bool MNCoreRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  llvm_unreachable("MN-Core2 has no stack frame; frame indices are invalid");
}
