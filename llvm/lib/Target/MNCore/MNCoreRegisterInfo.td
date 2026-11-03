//===- MNCoreRegisterInfo.td - MN-Core2 register defs ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MN-Core2 per-PE storage model (software view):
//
//   * GRF0 / GRF1 : 256 longwords (64b) each, general register file.
//   * TREG        : 8 longwords (2 longwords x 4 cycles), temporary register.
//   * OMR         : 32 entries x 16b output mask register (flags).
//   * MATX/MATY   : two matrix-register faces per MAB (16 physical rows).
//   * Forwarding  : $mauf / $aluf / $lbf / $mreadf and the L1BM
//                   turnaround register $lbi are modelled as physical
//                   registers that are only implicitly defined by the
//                   producing PE instruction expressions.
//
// LM0/LM1 (2Ki longword each) and the upper memories (L1BM, L2BM, PDM,
// DRAM) are *not* registers; they are accessed through explicit
// memory-command instruction expressions (see MNCoreInstrL*.td).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the MN-Core2 register file
//===----------------------------------------------------------------------===//

class MNCoreReg<string n> : Register<n> {
  field bits<16> HwEncoding;
  let Namespace = "MNCore";
}

// General register file entries (one per longword, indexed by address).
class GReg<int num, string n> : MNCoreReg<n> {
  let HWEncoding = num;
}

// Temporary register entries.
class TReg<int num, string n> : MNCoreReg<n> {
  let HWEncoding = num;
}

// Output mask register entries.
class OMReg<int num, string n> : MNCoreReg<n> {
  let HWEncoding = num;
}

// Special singleton registers.
class SReg<int num, string n> : MNCoreReg<n> {
  let HWEncoding = num;
}

//===----------------------------------------------------------------------===//
// GRF0 / GRF1: 256 longwords each.
//===----------------------------------------------------------------------===//

foreach i = 0 - 255 in {
  def GRF0_#i : GReg<i, "grf0_"#i>;
  def GRF1_#i : GReg<!add(256, i), "grf1_"#i>;
}

//===----------------------------------------------------------------------===//
// T register: 8 longwords.
//===----------------------------------------------------------------------===//

foreach i = 0 - 7 in
  def TREG_#i : TReg<!add(512, i), "t_"#i>;

//===----------------------------------------------------------------------===//
// OMR (mask register): 32 entries of 16 bits.
// Entries 1-15 are writable; 0 and 16-31 are fixed constants.
//===----------------------------------------------------------------------===//

foreach i = 0 - 31 in
  def OMR_#i : OMReg<!add(520, i), "omr_"#i>;

//===----------------------------------------------------------------------===//
// Special registers: forwarding latches, turnaround register, matrix
// register faces. These are never register-allocated; they only appear
// as implicit operands or as the destination of dedicated instructions.
//===----------------------------------------------------------------------===//

def MAUF    : SReg<552, "mauf">;    // MAU result forwarding latch
def ALUF    : SReg<553, "aluf">;    // ALU result forwarding latch
def LBF     : SReg<554, "lbf">;     // L1BM->PE forwarding latch
def MREADF  : SReg<555, "mreadf">;  // matrix transpose-read forwarding latch
def LBI     : SReg<556, "lbi">;     // L1BM turnaround register

def MATX    : SReg<560, "matx">;    // matrix register face X
def MATY    : SReg<561, "maty">;    // matrix register face Y

//===----------------------------------------------------------------------===//
// Register classes
//===----------------------------------------------------------------------===//

// 64-bit general register files. i64 is the only legal value type for
// general-purpose data; narrower/larger accesses are handled by the
// instruction-level operand width specifiers (word / long-word /
// double-long-word).
def GRF0 : RegisterClass<"MNCore", [i64], 64,
                         (add (sequence "GRF0_%u", 0, 255))>;

def GRF1 : RegisterClass<"MNCore", [i64], 64,
                         (add (sequence "GRF1_%u", 0, 255))>;

// Union class: either general register file. Most ALU commands can use
// either file for their PE memory operands, so a common allocatable
// class keeps the allocator simple for now.
def GRF : RegisterClass<"MNCore", [i64], 64,
                        (add GRF0, GRF1)>;

// T register class.
def TREG : RegisterClass<"MNCore", [i64], 64,
                         (add (sequence "TREG_%u", 0, 7))>;

// Mask register class (writable entries 1-15 plus the read-only
// fixed entries for completeness).
def OMR : RegisterClass<"MNCore", [i32], 32,
                        (add (sequence "OMR_%u", 0, 31))>;

// Matrix register faces (2 per MAB, not addressable by number).
def MATREG : RegisterClass<"MNCore", [untyped], 0, (add MATX, MATY)>;
