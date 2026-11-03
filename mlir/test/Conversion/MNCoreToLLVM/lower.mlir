// RUN: mlir-opt %s --convert-mncore-to-llvm | FileCheck %s

// Every token builder becomes a llvm.mncore.* intrinsic call, and the result
// is a plain i64. `mncore.none` is the ABI sentinel -1.

// CHECK-LABEL: llvm.func @tokens
func.func @tokens() -> !mncore.token {
  // CHECK: llvm.call_intrinsic "llvm.mncore.peop"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64) -> i64
  %pe = "mncore.peop"() <{space = 1 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.lbop"
  %lb = "mncore.lbop"() <{mem = 1 : i32, addr = 4 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.matop"
  %mat = "mncore.matop"() <{face = 1 : i32, row = 3 : i64, width = 0 : i32, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.maskop"
  %mask = "mncore.maskop"() <{kind = 0 : i32, value = 5 : i32, flags = 1 : i32}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.mvop"
  %mv = "mncore.mvop"() <{kind = 1 : i32, addr = 0 : i64, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.mvparams"
  %mp = "mncore.mvparams"() <{size = 64 : i64, tag = 1 : i32, nd = 0 : i32, prio = 0 : i32}> : () -> !mncore.token
  // CHECK: llvm.mlir.constant(-1 : i64) : i64
  %n = "mncore.none"() : () -> !mncore.token
  // The conversion must not leave the token type behind.
  // CHECK-NOT: !mncore.token
  return %pe : !mncore.token
}

// CHECK-LABEL: llvm.func @alu
func.func @alu() {
  %sx = "mncore.peop"() <{space = 1 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %sy = "mncore.peop"() <{space = 2 : i32, addr = 0 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %d = "mncore.peop"() <{space = 3 : i32, addr = 16 : i64, width = 2 : i32, flags = 8 : i32, aux = 0 : i64}> : () -> !mncore.token
  %n = "mncore.none"() : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.alu"({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
  "mncore.alu"(%sx, %sy, %d, %n, %n, %n, %n, %n) <{opcode = 0 : i64, prec = 0 : i32, opt = 0 : i32}> : (!mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token, !mncore.token) -> ()
  // CHECK: llvm.call_intrinsic "llvm.mncore.fence"() : () -> ()
  "mncore.fence"() : () -> ()
  // CHECK: llvm.call_intrinsic "llvm.mncore.wait"(%{{.*}}) : (i64) -> ()
  "mncore.wait"() <{tag = 1 : i32}> : () -> ()
  return
}

// CHECK-LABEL: llvm.func @vecfma
func.func @vecfma(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) {
  // CHECK: llvm.call_intrinsic "llvm.mncore.fma.v4f32"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, i64, i64) -> vector<4xf32>
  %r = "mncore.vec.fma"(%a, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // Typed values cannot escape the kernel; consume it with a typed op.
  // CHECK: llvm.call_intrinsic "llvm.mncore.matwrite.v4f32"
  "mncore.mat.write"(%r) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// CHECK-LABEL: llvm.func @vecmin
func.func @vecmin(%a: vector<8xf16>, %b: vector<8xf16>) {
  // CHECK: llvm.call_intrinsic "llvm.mncore.min.v8f16"
  %r = "mncore.vec.min"(%a, %b) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<8xf16>, vector<8xf16>) -> vector<8xf16>
  "mncore.mat.write"(%r) <{face = 0 : i64, row = 0 : i64}> : (vector<8xf16>) -> ()
  return
}

// CHECK-LABEL: llvm.func @matops
func.func @matops(%x: vector<2xf64>, %y: vector<2xf64>) {
  // CHECK: llvm.call_intrinsic "llvm.mncore.matvec.v2f64"
  %v = "mncore.mat.vec"(%x, %y) <{face = 1 : i64}> : (vector<2xf64>, vector<2xf64>) -> vector<2xf64>
  // CHECK: llvm.call_intrinsic "llvm.mncore.matwrite.v2f64"
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 1 : i64}> : (vector<2xf64>) -> ()
  // CHECK: llvm.call_intrinsic "llvm.mncore.matread.v2f64"
  %r = "mncore.mat.read"() <{face = 1 : i64, row = 1 : i64}> : () -> vector<2xf64>
  "mncore.mat.write"(%r) <{face = 1 : i64, row = 2 : i64}> : (vector<2xf64>) -> ()
  return
}

// CHECK-LABEL: llvm.func @vecload
func.func @vecload() {
  // CHECK: llvm.call_intrinsic "llvm.mncore.lbop"
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  // CHECK: llvm.call_intrinsic "llvm.mncore.l1load.v4f32"
  %v = "mncore.vec.load"(%lb) <{mode = 0 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token) -> vector<4xf32>
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// CHECK-LABEL: llvm.func @dma
func.func @dma() {
  // CHECK: llvm.call_intrinsic "llvm.mncore.mvop"
  // CHECK: llvm.call_intrinsic "llvm.mncore.mvparams"
  // CHECK: llvm.call_intrinsic "llvm.mncore.mv"
  // CHECK: llvm.call_intrinsic "llvm.mncore.lbop"
  // CHECK: llvm.call_intrinsic "llvm.mncore.l2b"
  "mncore.dma"() <{src_space = 1 : i32, src_addr = 0 : i64, l2_addr = 0 : i64,
                    l1_addr = 0 : i64, size = 64 : i64, tag = 1 : i32,
                    mv_flags = 0 : i32}> : () -> ()
  return
}
