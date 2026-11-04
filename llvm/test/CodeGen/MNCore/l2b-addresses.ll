; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; L2BM expressions use the [35:4] L1B address field (addr<<4), while L1BM
; expressions use [17:2] (addr<<2). Nonzero addresses catch a packer/printer
; mismatch that zero-address tests cannot.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare void @llvm.mncore.l2b(i64, i64, i64, i64)

define void @load_plane() {
; CHECK-LABEL: load_plane:
; CHECK: l2bmb $lc64 $lb64
  %lc = call i64 @llvm.mncore.lbop(i64 1, i64 64, i64 1, i64 0)
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 64, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 0, i64 0, i64 %lc, i64 %lb)
  ret void
}

define void @store_plane() {
; CHECK-LABEL: store_plane:
; CHECK: l2bmd $lb64 $lc64
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 64, i64 1, i64 0)
  %lc = call i64 @llvm.mncore.lbop(i64 1, i64 64, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 1, i64 0, i64 %lb, i64 %lc)
  ret void
}
