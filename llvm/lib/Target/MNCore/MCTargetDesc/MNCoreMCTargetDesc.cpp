//===- MNCoreMCTargetDesc.cpp - MN-Core2 Target Descriptions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreMCTargetDesc.h"
#include "MNCoreInstPrinter.h"
#include "MNCoreMCAsmInfo.h"
#include "TargetInfo/MNCoreTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "MNCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MNCoreGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MNCoreGenRegisterInfo.inc"

static MCInstrInfo *createMNCoreMCInstrInfo() {
  auto *X = new MCInstrInfo();
  InitMNCoreMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMNCoreMCRegisterInfo(const Triple &TT) {
  auto *X = new MCRegisterInfo();
  InitMNCoreMCRegisterInfo(X, /*RA=*/0);
  return X;
}

static MCSubtargetInfo *createMNCoreMCSubtargetInfo(const Triple &TT,
                                                    StringRef CPU,
                                                    StringRef FS) {
  return createMNCoreMCSubtargetInfoImpl(TT, CPU, /*TuneCPU=*/CPU, FS);
}

static MCAsmInfo *createMNCoreMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  return new MNCoreMCAsmInfo(TT, Options);
}

static MCInstPrinter *createMNCoreMCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new MNCoreInstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMNCoreTargetMC() {
  Target &TheMNCoreTarget = getTheMNCoreTarget();

  RegisterMCAsmInfoFn X(TheMNCoreTarget, createMNCoreMCAsmInfo);
  TargetRegistry::RegisterMCInstrInfo(TheMNCoreTarget,
                                      createMNCoreMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(TheMNCoreTarget,
                                    createMNCoreMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheMNCoreTarget,
                                          createMNCoreMCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(TheMNCoreTarget,
                                        createMNCoreMCInstPrinter);
}
