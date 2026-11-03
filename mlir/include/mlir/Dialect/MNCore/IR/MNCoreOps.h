//===- MNCoreOps.h - MN-Core 2 dialect operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MNCORE_IR_MNCOREOPS_H
#define MLIR_DIALECT_MNCORE_IR_MNCOREOPS_H

#include "mlir/Dialect/MNCore/IR/MNCoreTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::mncore {

/// Side-effect resources. Instruction ops touch the PE register file and
/// device memory; MV commands additionally hand data to the transfer engine.
struct PEState : public SideEffects::Resource::Base<PEState> {
  StringRef getName() const final { return "mncore.pe"; }
};
struct DeviceMem : public SideEffects::Resource::Base<DeviceMem> {
  StringRef getName() const final { return "mncore.device_mem"; }
};
struct MVChannel : public SideEffects::Resource::Base<MVChannel> {
  StringRef getName() const final { return "mncore.mv_channel"; }
};

} // namespace mlir::mncore

#include "mlir/Dialect/MNCore/IR/MNCoreOpsEnums.h.inc"
#include "mlir/Dialect/MNCore/IR/MNCoreOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MNCore/IR/MNCoreOps.h.inc"

#endif // MLIR_DIALECT_MNCORE_IR_MNCOREOPS_H
