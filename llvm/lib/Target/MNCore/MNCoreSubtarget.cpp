//===- MNCoreSubtarget.cpp - MN-Core2 Subtarget Information -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreSubtarget.h"
#include "MNCore.h"
#include "MNCoreSelectionDAGInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "mncore-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MNCoreGenSubtargetInfo.inc"

void MNCoreSubtarget::anchor() {}

MNCoreSubtarget::MNCoreSubtarget(const Triple &TT, const std::string &CPU,
                                 const std::string &FS,
                                 const TargetMachine &TM)
    : MNCoreGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS), InstrInfo(*this),
      TLInfo(TM, *this), FrameLowering(*this) {
  TSInfo = std::make_unique<MNCoreSelectionDAGInfo>();
  // Populate the itinerary/sched-model data generated from
  // MNCoreSchedule.td; the default InstrItineraryData is empty and the
  // VLIW scheduler requires a real model.
  initInstrItins(InstrItins);
}

MNCoreSubtarget::~MNCoreSubtarget() = default;

const SelectionDAGTargetInfo *MNCoreSubtarget::getSelectionDAGInfo() const {
  return TSInfo.get();
}
