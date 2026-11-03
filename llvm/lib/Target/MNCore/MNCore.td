//===- MNCore.td - Describe the MN-Core2 Target Machine -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// MN-Core2 subtarget features
//===----------------------------------------------------------------------===//

// Instruction-stream packing modes. The default skeleton emits flat-mode
// assembly (2 PE instructions + 1 MV instruction per stream unit) which
// does not restrict the per-cycle addressing of LM0/LM1/GRF0/GRF1.
// Auto-stride mode (3 PE instructions + 1 MV instruction per unit with
// auto-incremented addresses) is left as a later optimization.
def FeatureFlatMode
    : SubtargetFeature<"flat", "HasFlatMode", "true",
                       "Emit flat-mode instruction stream (default)">;
def FeatureAutoStride
    : SubtargetFeature<"autostride", "HasAutoStride", "false",
                       "Emit auto-stride-mode instruction stream">;

//===----------------------------------------------------------------------===//
// Registers, instruction descriptions, scheduling model
//===----------------------------------------------------------------------===//

include "MNCoreRegisterInfo.td"
include "MNCoreSchedule.td"
include "MNCoreInstrInfo.td"
include "MNCoreInstrMV.td"
include "MNCoreInstrL2BM.td"
include "MNCoreInstrL1BM.td"
include "MNCoreInstrMAU.td"

def MNCoreInstrInfo : InstrInfo;

// MN-Core2 has no addressable memory above the register files in the
// scalar model, so the generic `ptr_rc` used by the standard pseudo
// instructions (LOAD_STACK_GUARD, PREALLOCATED_ARG, ...) maps onto the
// general register class. These pseudos never survive isel for a
// straight-line kernel.
defm : RemapAllTargetPseudoPointerOperands<GRF>;

class Proc<string Name, list<SubtargetFeature> Features>
    : ProcessorModel<Name, MNCoreModel, Features>;

def : Proc<"mncore2", [FeatureFlatMode]>;

def MNCore : Target {
  let InstructionSet = MNCoreInstrInfo;
}
