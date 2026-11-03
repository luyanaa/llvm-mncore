//===- MNCoreSelectionDAGInfo.cpp - MN-Core2 SelectionDAG Info ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreSelectionDAGInfo.h"

using namespace llvm;

#define GET_SDNODE_DESC
#include "MNCoreGenSDNodeInfo.inc"

MNCoreSelectionDAGInfo::MNCoreSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(MNCoreGenSDNodeInfo) {}

MNCoreSelectionDAGInfo::~MNCoreSelectionDAGInfo() = default;
