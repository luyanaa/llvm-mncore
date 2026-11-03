// RUN: not mlir-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

// Negative test: the 64-PE-distinct distribution (l1bmd) has no
// double-longword form, so a 2-longword vector cannot use it. This is the
// guard against silently emitting l1bmp when PEs must hold different values.

// ERR: l1bmd has no double-longword form
func.func @distributed_2lw() {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %v = "mncore.vec.load"(%lb) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// ERR: 8-longword aligned
func.func @unaligned_mab() {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 4 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %v = "mncore.vec.load"(%lb) <{mode = 1 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}
