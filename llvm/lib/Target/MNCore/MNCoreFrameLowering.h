//===- MNCoreFrameLowering.h - MN-Core2 frame lowering ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MN-Core2 kernels have no call stack and no addressable local storage
// above the register files, so there is no frame to lay out. This class
// exists only because the generic codegen pipeline dereferences
// TargetSubtargetInfo::getFrameLowering() unconditionally; every entry
// point either does nothing or unreachable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREFRAMELOWERING_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class MNCoreSubtarget;

class MNCoreFrameLowering : public TargetFrameLowering {
public:
  explicit MNCoreFrameLowering(const MNCoreSubtarget &STI)
      : TargetFrameLowering(StackDirection::StackGrowsDown, Align(8),
                            /*LocalAreaOffset=*/0) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREFRAMELOWERING_H
