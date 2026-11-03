//===- MNCoreMachineFunctionInfo.h - MN-Core2 machine function info - C++ -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MN-Core2 has no call frames or stack; this class is kept minimal and
// only exists to satisfy the TargetMachine interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class MNCoreFunctionInfo : public MachineFunctionInfo {
public:
  explicit MNCoreFunctionInfo(const Function &F,
                              const TargetSubtargetInfo *STI) {}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREMACHINEFUNCTIONINFO_H
