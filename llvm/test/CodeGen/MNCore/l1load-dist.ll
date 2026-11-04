; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Distributed load (SDM 3.6.8.20): l1bmd has no double-longword form, so a
; 4xf32/PE vector is two transfers, one per longword plane. The PE slot's
; longwords are addressed W/2 and W/2+4.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.l1load.v4f32(i64, i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, i64, i64)

define void @dist(<4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: dist:
; CHECK: l1bmd $lb0 $lm0v
; CHECK: l1bmd $lb64 $lm4v
; CHECK: nop/2
; CHECK: fvfma $llm0v $llr0v $lls0v $lln0v
  %lo = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  %hi = call i64 @llvm.mncore.lbop(i64 0, i64 64, i64 1, i64 0)
  %a = call <4 x float> @llvm.mncore.l1load.v4f32(i64 3, i64 %lo, i64 %hi, i64 -1, i64 -1)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  ret void
}
