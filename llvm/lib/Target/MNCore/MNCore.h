//===- MNCore.h - Top-level interface for MN-Core2 backend -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the
// LLVM MN-Core2 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCORE_H
#define LLVM_LIB_TARGET_MNCORE_MNCORE_H

#include "MCTargetDesc/MNCoreMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class MNCoreTargetMachine;
class FunctionPass;
class PassRegistry;

FunctionPass *createMNCoreISelDag(MNCoreTargetMachine &TM,
                                  CodeGenOptLevel OptLevel);
FunctionPass *createMNCorePacketizer();
FunctionPass *createMNCoreHazard();
void initializeMNCoreAsmPrinterPass(PassRegistry &);
void initializeMNCoreDAGToDAGISelLegacyPass(PassRegistry &);
void initializeMNCorePacketizerPass(PassRegistry &);
void initializeMNCoreHazardPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCORE_H
