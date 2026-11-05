// RUN: not mlir-opt %s 2>&1 | FileCheck %s --check-prefix=ERR

// Negative test: the 64-PE-distinct distribution (l1bmd) has no
// double-longword form, so a 2-longword vector cannot use it. This is the
// guard against silently emitting l1bmp when PEs must hold different values.

// ERR: distributed load sources must be 1-longword operands
func.func @distributed_2lw(%a: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %v = "mncore.vec.load"(%lb, %hi) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  "mncore.vec.add"(%v, %a) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  return
}

// ERR: 8-longword aligned
func.func @unaligned_mab(%a: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 4 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hn = "mncore.none"() : () -> !mncore.token
  %v = "mncore.vec.load"(%lb, %hn) <{mode = 1 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  "mncore.vec.add"(%v, %a) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  return
}

// The f32 matrix roles are 4 longwords; a 2-longword f32 value there would
// leave half of the hardware operand stale.
// ERR: expects a 4-longword f32 matrix operand (<8 x f32>) for src
func.func @matwrite_2lw(%v: vector<4xf32>) {
  "mncore.mat.write"(%v) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}

// ERR: expects a 4-longword f32 matrix operand (<8 x f32>) for y
func.func @matvec_narrow_y(%x: vector<4xf32>, %y: vector<4xf32>) {
  %r = "mncore.mat.vec"(%x, %y) <{face = 0 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  return
}

// ERR: expects a 2-longword f32 vector (<4 x f32>) for b
func.func @gemmtile_wide_b(%alo: vector<8xf32>, %b: vector<8xf32>, %c0: vector<8xf32>) {
  %r = "mncore.gemm.tile"(%alo, %alo, %b, %c0) <{face = 0 : i64}> : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  "mncore.mat.write"(%r) <{face = 0 : i64, row = 0 : i64}> : (vector<8xf32>) -> ()
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

// ERR: a relocatable source must live in DRAM
func.func @reloc_pdm() {
  "mncore.dma"() <{src_space = 0 : i32, src_addr = 0 : i64, l2_addr = 0 : i64,
                    l1_addr = 0 : i64, size = 64 : i64, tag = 1 : i32,
                    src_arg = 0 : i32}> : () -> ()
  return
}

// ERR: src_arg must be -1 (absolute address) or a kernel argument index
func.func @reloc_arg_range() {
  "mncore.dma"() <{src_space = 1 : i32, src_addr = 0 : i64, l2_addr = 0 : i64,
                    l1_addr = 0 : i64, size = 64 : i64, tag = 1 : i32,
                    src_arg = 300 : i32}> : () -> ()
  return
}
