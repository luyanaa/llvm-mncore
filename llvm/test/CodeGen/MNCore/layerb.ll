; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Layer B: typed <4 x float> fma. The mncore-lower-typed-intrinsics pass
; performs the PE slot allocation (GRF0, 2-longword, 4 words apart, no
; reuse yet), assigns slots 0/4/8 to the vector arguments and 12/16 to the
; results, and rewrites the calls to Layer A peop + mau. A hazard nop may
; appear between the two fvfma steps because the second reads slot 12.

target triple = "mncore2"

declare <4 x float> @llvm.mncore.fma.v4f32(<4 x float>, <4 x float>, <4 x float>, i64, i64)
declare i64 @llvm.mncore.maskop(i64, i64, i64)

define void @k(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: k:
; Slots rotate over GRF0/GRF1/LM0/LM1 (one address field per space), so
; the arguments take $llr0v/$lls0v/$llm0v and the result $lln0v.
; The first result takes the remaining space (LM1), the second reuses GRF0
; after %a dies (its freed slot 0 is reused); destinations never collide
; with a live value's space.
; CHECK: fvfma $llr0v $lls0v $llm0v $lln0v
; CHECK: nop
; CHECK: fvfma $lln0v $lls0v $llm0v $llr0v
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, i64 0, i64 0)
  %s = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %r, <4 x float> %b, <4 x float> %c, i64 0, i64 0)
  ret void
}

; Typed mask forwarding: the wmask argument is a maskop token; the pass
; forwards it to Layer A, which prints /0111.
define void @masked_fma(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: masked_fma:
; CHECK: fvfma $llr0v $lls0v $llm0v $lln0v/0111p
  %wm = call i64 @llvm.mncore.maskop(i64 0, i64 7, i64 0)
  %r = call <4 x float> @llvm.mncore.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, i64 -1, i64 %wm)
  ret void
}
