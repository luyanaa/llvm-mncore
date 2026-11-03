//===- MNCoreSelectionDAGInfo.h - MN-Core2 SelectionDAG Info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCORESELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_MNCORE_MNCORESELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "MNCoreGenSDNodeInfo.inc"

namespace llvm {

class MNCoreSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  MNCoreSelectionDAGInfo();
  ~MNCoreSelectionDAGInfo() override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCORESELECTIONDAGINFO_H
