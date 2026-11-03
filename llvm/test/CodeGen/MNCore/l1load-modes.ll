; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; L1BM distribution modes: MAB -> l1bmm, MAB4 -> l1bmm4. Broadcast is covered
; by layerb.ll. Distributed (l1bmd) is rejected by the mncore verifier and not
; wired in the intrinsic selector.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.l1load.v4f32(i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, i64, i64)

define void @mab(<4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: mab:
; CHECK: l1bmm $llb0 $llm0v
; CHECK: nop
; CHECK: fvfma $llm0v $llr0v $lls0v
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 2, i64 0)
  %v = call <4 x float> @llvm.mncore.l1load.v4f32(i64 1, i64 %lb, i64 -1, i64 -1)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %v, <4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  ret void
}

define void @mab4(<4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: mab4:
; CHECK: l1bmm4 $llb0 $llm0v
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 2, i64 0)
  %v = call <4 x float> @llvm.mncore.l1load.v4f32(i64 2, i64 %lb, i64 -1, i64 -1)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %v, <4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  ret void
}
