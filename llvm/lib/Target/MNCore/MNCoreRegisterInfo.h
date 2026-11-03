//===- MNCoreRegisterInfo.h - MN-Core2 Register Information Impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MN-Core2 implementation of the
// TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREREGISTERINFO_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "MNCoreGenRegisterInfo.inc"

namespace llvm {

class MNCoreSubtarget;

struct MNCoreRegisterInfo : public MNCoreGenRegisterInfo {
  const MNCoreSubtarget &STI;

public:
  explicit MNCoreRegisterInfo(const MNCoreSubtarget &sti);

  /// Return the set of registers that the register allocator cannot
  /// use: fixed forwarding latches, the turnaround register, matrix
  /// faces, and the read-only mask entries 0/16-31.
  BitVector getReservedRegs(const MachineFunction &MF) const override;

  // MN-Core2 has no call/return ABI, hence no callee-saved registers.
  // The generic code still expects a null-terminated array, so return
  // an empty list rather than nullptr.
  const MCPhysReg *getCalleeSavedRegs(
      const MachineFunction *MF) const override {
    static const MCPhysReg Empty[] = {0};
    return Empty;
  }

  // Prefer 2-aligned GRF addresses: longword PE operands require an even
  // address (3.6.1.6/.10). The allocator otherwise has no notion of the
  // packed VSM address, and an odd choice is rejected at emission.
  bool getRegAllocationHints(Register VirtReg, ArrayRef<MCPhysReg> Order,
                             SmallSetVector<MCPhysReg, 16> &Hints,
                             const MachineFunction &MF, const VirtRegMap *VRM,
                             const LiveRegMatrix *Matrix) const override;

  // MN-Core2 has no stack frames or calls, so the frame/call related
  // hooks are inert or hard-error if the generic pipeline asks for
  // them.
  Register getFrameRegister(const MachineFunction &MF) const override;
  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return false;
  }
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return false;
  }
  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const override {
    return true;
  }
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREREGISTERINFO_H
