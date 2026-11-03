//===- MNCoreMCAsmInfo.h - MN-Core2 asm properties -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCASMINFO_H
#define LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {

class Triple;

class MNCoreMCAsmInfo : public MCAsmInfo {
public:
  explicit MNCoreMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MCTARGETDESC_MNCOREMCASMINFO_H
