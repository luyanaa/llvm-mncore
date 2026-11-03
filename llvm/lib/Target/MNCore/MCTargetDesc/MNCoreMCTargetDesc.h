//===- MNCoreMCTargetDesc.h - MN-Core2 Target Descriptions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCTARGETDESC_H
#define LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class Target;

} // end namespace llvm

// Defines symbolic names for MN-Core2 registers.
#define GET_REGINFO_ENUM
#include "MNCoreGenRegisterInfo.inc"

// Defines symbolic names for the MN-Core2 instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "MNCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MNCoreGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCTARGETDESC_H
