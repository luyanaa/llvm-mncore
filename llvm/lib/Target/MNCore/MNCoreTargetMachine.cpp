//===- MNCoreTargetMachine.cpp - Define TargetMachine for MN-Core2 - C++ -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MNCoreTargetMachine.h"
#include "MNCore.h"
#include "MNCoreMachineFunctionInfo.h"
#include "TargetInfo/MNCoreTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include <optional>

using namespace llvm;

static Reloc::Model getRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

MNCoreTargetMachine::MNCoreTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(
          T, TT, CPU, FS, Options, getRelocModel(RM),
          getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      // There is a single processor; default to it so that the
      // scheduling model (and its DFA) is always selected, not the
      // empty NoSchedModel fallback.
      Subtarget(TT, CPU.empty() ? "mncore2" : std::string(CPU),
                std::string(FS), *this) {
  initAsmInfo();
}

MNCoreTargetMachine::~MNCoreTargetMachine() = default;

namespace {

class MNCorePassConfig : public TargetPassConfig {
public:
  MNCorePassConfig(MNCoreTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  MNCoreTargetMachine &getMNCoreTargetMachine() const {
    return getTM<MNCoreTargetMachine>();
  }

  bool addInstSelector() override;
  void addPreEmitPass() override;
};

} // end anonymous namespace

TargetPassConfig *MNCoreTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MNCorePassConfig(*this, PM);
}

bool MNCorePassConfig::addInstSelector() {
  addPass(createMNCoreISelDag(getMNCoreTargetMachine(), getOptLevel()));
  return false;
}

void MNCorePassConfig::addPreEmitPass() {
  // Bundle the PE instruction expressions scheduled into the same
  // 4-cycle step so the AsmPrinter can emit them as one ';'-separated
  // .vsm line.
  addPass(createMNCorePacketizer());
}

MachineFunctionInfo *MNCoreTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return MNCoreFunctionInfo::create<MNCoreFunctionInfo>(Allocator, F, STI);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMNCoreTarget() {
  RegisterTargetMachine<MNCoreTargetMachine> X(getTheMNCoreTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeMNCoreAsmPrinterPass(PR);
  initializeMNCoreDAGToDAGISelLegacyPass(PR);
  initializeMNCorePacketizerPass(PR);
}
