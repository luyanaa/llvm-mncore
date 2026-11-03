//===- MNCoreInstrInfo.cpp - MN-Core2 Instruction Information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreInstrInfo.h"
#include "MNCore.h"
#include "MNCoreSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "MNCoreGenInstrInfo.inc"

// Generated DFA automaton + resource tables for the VLIW packetizer.
#include "MNCoreGenDFAPacketizer.inc"

void MNCoreInstrInfo::anchor() {}

DFAPacketizer *MNCoreInstrInfo::CreateTargetScheduleState(
    const TargetSubtargetInfo &STI) const {
  const InstrItineraryData *II = STI.getInstrItineraryData();
  return static_cast<const MNCoreSubtarget &>(STI).createDFAPacketizer(II);
}

MNCoreInstrInfo::MNCoreInstrInfo(const MNCoreSubtarget &STI)
    : MNCoreGenInstrInfo(STI, RI, MNCore::ADJCALLSTACKDOWN,
                         MNCore::ADJCALLSTACKUP),
      RI(STI) {}

void MNCoreInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  const DebugLoc &DL, Register DestReg,
                                  Register SrcReg, bool KillSrc,
                                  bool RenamableDest,
                                  bool RenamableSrc) const {
  // Physical copies inside the GRF files lower to the generic COPY_GRF
  // pseudo; the asm printer emits the `lpassa`-style long-word copy.
  BuildMI(MBB, MI, DL, get(MNCore::COPY_GRF), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}
