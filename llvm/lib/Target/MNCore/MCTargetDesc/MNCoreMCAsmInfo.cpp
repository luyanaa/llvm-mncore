//===- MNCoreMCAsmInfo.cpp - MN-Core2 asm properties -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreMCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

MNCoreMCAsmInfo::MNCoreMCAsmInfo(const Triple &TT,
                                 const MCTargetOptions &Options)
    : MCAsmInfo(Options) {
  // .vsm line-oriented assembly; '#' comments, one statement per line.
  CommentString = "#";
  HasFunctionAlignment = false;
  SupportsDebugInformation = false;

  // Statements are terminated by end-of-line, not ';'.
  SeparatorString = "\n";
}
