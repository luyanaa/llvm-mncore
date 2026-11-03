//===- MNCoreSubtarget.h - Define Subtarget for the MN-Core2 -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MN-Core2 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCORESUBTARGET_H
#define LLVM_LIB_TARGET_MNCORE_MNCORESUBTARGET_H

#include "MNCoreFrameLowering.h"
#include "MNCoreISelLowering.h"
#include "MNCoreInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "MNCoreGenSubtargetInfo.inc"

namespace llvm {

class StringRef;
class TargetMachine;

class MNCoreSubtarget : public MNCoreGenSubtargetInfo {
  virtual void anchor();
  MNCoreInstrInfo InstrInfo;
  MNCoreTargetLowering TLInfo;
  MNCoreFrameLowering FrameLowering;
  std::unique_ptr<const SelectionDAGTargetInfo> TSInfo;
  InstrItineraryData InstrItins;

  // Packing mode features.
  bool HasFlatMode = true;
  bool HasAutoStride = false;

public:
  MNCoreSubtarget(const Triple &TT, const std::string &CPU,
                  const std::string &FS, const TargetMachine &TM);
  ~MNCoreSubtarget() override;

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  const MNCoreInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const MNCoreTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const MNCoreRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override;

  // MN-Core2 has no hardware-managed stack frame; the frame lowering
  // object is a no-op stub that only exists because generic codegen
  // passes dereference it unconditionally.
  const MNCoreFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  // The PE step is a 12-slot VLIW bundle described by the itinerary and
  // DFA generated from MNCoreSchedule.td.
  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  bool hasFlatMode() const { return HasFlatMode; }
  bool hasAutoStride() const { return HasAutoStride; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCORESUBTARGET_H
