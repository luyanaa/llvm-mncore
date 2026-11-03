//===- MNCoreDialect.cpp - MN-Core 2 dialect implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MNCore/IR/MNCoreOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/MNCore/IR/MNCoreOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::mncore;

#include "mlir/Dialect/MNCore/IR/MNCoreOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MNCoreDialect
//===----------------------------------------------------------------------===//

void MNCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MNCore/IR/MNCoreOps.cpp.inc"
      >();
  registerTypes();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MNCore/IR/MNCoreOpsTypes.cpp.inc"

void MNCoreDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MNCore/IR/MNCoreOpsTypes.cpp.inc"
      >();
}
