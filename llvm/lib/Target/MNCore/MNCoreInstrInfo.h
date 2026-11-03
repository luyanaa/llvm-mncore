//===- MNCoreInstrInfo.h - MN-Core2 Instruction Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREINSTRINFO_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREINSTRINFO_H

#include "MNCoreRegisterInfo.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MNCoreGenInstrInfo.inc"

namespace llvm {

class MNCoreSubtarget;

class MNCoreInstrInfo : public MNCoreGenInstrInfo {
  const MNCoreRegisterInfo RI;
  virtual void anchor();

public:
  explicit MNCoreInstrInfo(const MNCoreSubtarget &STI);

  const MNCoreRegisterInfo &getRegisterInfo() const { return RI; }

  // MN-Core2 has no branches/calls in its hardware instruction stream;
  // every control-flow query is answered conservatively until the
  // straight-line pipeline enforces it at a higher level.
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  // VLIW resource tracker used by the selection-DAG packet scheduler.
  // The generated DFAPacketizer encodes the per-group single-slot
  // reservations of the parallel-execution rules.
  DFAPacketizer *
  CreateTargetScheduleState(const TargetSubtargetInfo &STI) const override;

  // A kernel is a single straight-line block; the generic branch
  // analysis/insertion helpers are never used.
  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override {
    return true;
  }
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override {
    llvm_unreachable("MN-Core2 has no branch instructions");
  }
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override {
    llvm_unreachable("MN-Core2 has no branch instructions");
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREINSTRINFO_H
