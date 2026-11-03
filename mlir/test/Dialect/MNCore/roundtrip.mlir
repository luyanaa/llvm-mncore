// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Round-trip of every token builder and instruction expression. The generic
// assembly form is used for now; pretty syntax is a follow-up.
//
// Inherent attributes are printed inside <{...}>.

// CHECK-LABEL: func.func @tokens
func.func @tokens() -> !mncore.token {
  // CHECK: "mncore.peop"() <{addr = 0 : i64, aux = 0 : i64, flags = 8 : i32, space = 1 : i32, width = 2 : i32}>
  %pe = "mncore.peop"() <{space = 1 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  // CHECK: "mncore.lbop"() <{addr = 4 : i64, flags = 0 : i32, mem = 1 : i32, width = 2 : i32}>
  %lb = "mncore.lbop"() <{mem = 1 : i32, addr = 4 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: "mncore.matop"() <{face = 1 : i32, flags = 0 : i32, row = 3 : i64, width = 0 : i32}>
  %mat = "mncore.matop"() <{face = 1 : i32, row = 3 : i64, width = 0 : i32, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: "mncore.maskop"() <{flags = 1 : i32, kind = 0 : i32, value = 5 : i32}>
  %mask = "mncore.maskop"() <{kind = 0 : i32, value = 5 : i32, flags = 1 : i32}> : () -> !mncore.token
  // CHECK: "mncore.mvop"() <{addr = 0 : i64, flags = 0 : i32, kind = 1 : i32}>
  %mv = "mncore.mvop"() <{kind = 1 : i32, addr = 0 : i64, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: "mncore.mvparams"() <{nd = 0 : i32, prio = 0 : i32, size = 64 : i64, tag = 1 : i32}>
  %mp = "mncore.mvparams"() <{size = 64 : i64, tag = 1 : i32, nd = 0 : i32, prio = 0 : i32}> : () -> !mncore.token
  // CHECK: "mncore.none"() : () -> !mncore.token
  %n = "mncore.none"() : () -> !mncore.token
  return %pe : !mncore.token
}

// CHECK-LABEL: func.func @inst
func.func @inst() {
  %sx = "mncore.peop"() <{space = 1 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %sy = "mncore.peop"() <{space = 2 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %d = "mncore.peop"() <{space = 3 : i32, addr = 16 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %n = "mncore.none"() : () -> !mncore.token
  // CHECK: "mncore.alu"({{.*}}) <{opcode = 0 : i64, opt = 0 : i32, prec = 0 : i32}>
  "mncore.alu"(%sx, %sy, %d, %n, %n, %n, %n, %n) <{opcode = 0 : i64, prec = 0 : i32, opt = 0 : i32}> : (!mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token) -> ()
  // CHECK: "mncore.fence"()
  "mncore.fence"() : () -> ()
  // CHECK: "mncore.wait"() <{tag = 1 : i32}>
  "mncore.wait"() <{tag = 1 : i32}> : () -> ()
  // CHECK: "mncore.maskdir"() <{spec = 3 : i32}>
  "mncore.maskdir"() <{spec = 3 : i32}> : () -> ()
  return
}

// CHECK-LABEL: func.func @typed
func.func @typed(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) {
  // CHECK: "mncore.vec.fma"({{.*}}) <{wmask = -1 : i64, zmask = -1 : i64}>
  %r = "mncore.vec.fma"(%a, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK: "mncore.vec.max"
  %m = "mncore.vec.max"(%r, %a) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK: "mncore.mat.write"
  "mncore.mat.write"(%m) <{face = 0 : i64, row = 2 : i64}> : (vector<4xf32>) -> ()
  // CHECK: "mncore.mat.read"() <{face = 1 : i64, row = 2 : i64}>
  %v = "mncore.mat.read"() <{face = 1 : i64, row = 2 : i64}> : () -> vector<4xf32>
  // Typed values may not escape (the Layer B pass rejects it), so consume it.
  "mncore.mat.write"(%v) <{face = 1 : i64, row = 3 : i64}> : (vector<4xf32>) -> ()
  // CHECK: "mncore.vec.load"
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %w = "mncore.vec.load"(%lb) <{mode = 0 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%w) <{face = 0 : i64, row = 4 : i64}> : (vector<4xf32>) -> ()
  return
}
