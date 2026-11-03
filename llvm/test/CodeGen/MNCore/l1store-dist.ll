; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Distributed writeback (SDM 3.6.8.21): l1bmd collect has no
; double-longword form, so the two longword planes are collected separately.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare void @llvm.mncore.l1store.v4f32(<4 x float>, i64, i64, i64, i64)

define void @dist(<4 x float> %v) {
; CHECK-LABEL: dist:
; CHECK: l1bmd $lr0v $lb0
; CHECK: l1bmd $lr4v $lb64
  %lo = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  %hi = call i64 @llvm.mncore.lbop(i64 0, i64 64, i64 1, i64 0)
  call void @llvm.mncore.l1store.v4f32(<4 x float> %v, i64 3, i64 %lo, i64 %hi, i64 0)
  ret void
}
