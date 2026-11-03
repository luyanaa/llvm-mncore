//===- MNCoreMCInstLower.h - Lower MachineInstr to MCInst -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MNCORE_MNCOREMCINSTLOWER_H
#define LLVM_LIB_TARGET_MNCORE_MNCOREMCINSTLOWER_H

#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCInst.h"

namespace llvm {

class AsmPrinter;
class MCContext;
class MachineInstr;
class Mangler;

class MNCoreMCInstLower {
  MCContext &Ctx;
  AsmPrinter &Printer;

public:
  MNCoreMCInstLower(MCContext &Ctx, AsmPrinter &Printer)
      : Ctx(Ctx), Printer(Printer) {}

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;
  MCOperand LowerOperand(const MachineOperand &MO) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MNCORE_MNCOREMCINSTLOWER_H
