// RUN: not mlir-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

// Negative test: the 64-PE-distinct distribution (l1bmd) has no
// double-longword form, so a 2-longword vector cannot use it. This is the
// guard against silently emitting l1bmp when PEs must hold different values.

// ERR: distributed load sources must be 1-longword operands
func.func @distributed_2lw() {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %v = "mncore.vec.load"(%lb, %hi) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// ERR: 8-longword aligned
func.func @unaligned_mab() {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 4 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hn = "mncore.none"() : () -> !mncore.token
  %v = "mncore.vec.load"(%lb, %hn) <{mode = 1 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// ERR: a broadcast store is not defined
func.func @broadcast_store(%v: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hn = "mncore.none"() : () -> !mncore.token
  "mncore.vec.store"(%v, %lb, %hn) <{mode = 0 : i64, set = 0 : i64}> : (vector<4xf32>, !mncore.token, !mncore.token) -> ()
  return
}

// ERR: distributed store destinations must be 1-longword operands
func.func @distributed_store(%v: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  "mncore.vec.store"(%v, %lb, %hi) <{mode = 3 : i64, set = 0 : i64}> : (vector<4xf32>, !mncore.token, !mncore.token) -> ()
  return
}
