//===- MNCoreTargetInfo.cpp - MN-Core2 Target Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/MNCoreTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheMNCoreTarget() {
  static Target TheMNCoreTarget;
  return TheMNCoreTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMNCoreTargetInfo() {
  RegisterTarget<Triple::mncore2> X(getTheMNCoreTarget(), "mncore2",
                                    "MN-Core2", "MNCore");
}
