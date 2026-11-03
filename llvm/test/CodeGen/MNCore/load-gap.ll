; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; An independent expression may share the load's step; only the load ->
; consumer pair must be separated. This is what replaces the explicit
; `fence` the Layer B pass used to emit.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.l1load.v4f32(i64, i64, i64, i64, i64)
declare <4 x float> @llvm.mncore.add.v4f32(<4 x float>, <4 x float>, i64, i64)
declare <4 x float> @llvm.mncore.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, i64, i64)

define void @gap(<4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: gap:
; CHECK: l1bmp $llb0 $llm0v; fvadd $llr0v $lls0v $lln0v
; CHECK: nop/2
; CHECK: fvfma $llm0v $lln0v $lls0v $llr0v
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 2, i64 0)
  %a = call <4 x float> @llvm.mncore.l1load.v4f32(i64 0, i64 %lb, i64 -1, i64 -1, i64 -1)
  %x = call <4 x float> @llvm.mncore.add.v4f32(<4 x float> %b, <4 x float> %c, i64 -1, i64 -1)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %a, <4 x float> %x, <4 x float> %c, i64 -1, i64 -1)
  ret void
}
