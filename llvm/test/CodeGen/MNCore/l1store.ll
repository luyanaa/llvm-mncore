; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; PE -> L1BM writeback: MAB collect -> l1bmm@<set>, MAB4 -> l1bmm4@<set>.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, i64, i64)
declare void @llvm.mncore.l1store.v4f32(<4 x float>, i64, i64, i64)

define void @mab(<4 x float> %v, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: mab:
; CHECK: fvfma $llr0v $lls0v $llm0v $lln0v
; CHECK: l1bmm@1 $lln0v $llb0
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 2, i64 0)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %v, <4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  call void @llvm.mncore.l1store.v4f32(<4 x float> %r, i64 1, i64 %lb, i64 1)
  ret void
}

define void @mab4(<4 x float> %v, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: mab4:
; CHECK: l1bmm4@2 $lln0v $llb0
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 2, i64 0)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %v, <4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  call void @llvm.mncore.l1store.v4f32(<4 x float> %r, i64 2, i64 %lb, i64 2)
  ret void
}
