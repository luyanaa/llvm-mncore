; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Layer B coverage in ONE function: <2 x double>, <4 x float>, <8 x half>
; for fma/mul/add/max/min, plus matvec/matwrite/matread. Coverage/syntax
; test: verified with assemble3's hazard checker disabled (the stream is
; intentionally dense). Hazard-on validation uses layerb.ll and the MIR
; EMIT tests.

target triple = "mncore2"

declare <4 x float> @llvm.mncore.mul.v4f32(<4 x float>, <4 x float>, i64, i64)
declare <4 x float> @llvm.mncore.max.v4f32(<4 x float>, <4 x float>, i64, i64)
declare <2 x double> @llvm.mncore.fma.v2f64(<2 x double>, <2 x double>, <2 x double>, i64, i64)
declare <8 x half> @llvm.mncore.add.v8f16(<8 x half>, <8 x half>, i64, i64)
declare <2 x double> @llvm.mncore.max.v2f64(<2 x double>, <2 x double>, i64, i64)
declare <2 x double> @llvm.mncore.min.v2f64(<2 x double>, <2 x double>, i64, i64)
declare <8 x half> @llvm.mncore.max.v8f16(<8 x half>, <8 x half>, i64, i64)
declare <8 x half> @llvm.mncore.min.v8f16(<8 x half>, <8 x half>, i64, i64)
declare <4 x float> @llvm.mncore.matvec.v4f32(i64, <4 x float>, <4 x float>)
declare void @llvm.mncore.matwrite.v4f32(i64, i64, <4 x float>)
declare <4 x float> @llvm.mncore.matread.v4f32(i64, i64)

define void @all(<4 x float> %a, <4 x float> %b, <4 x float> %c,
                 <2 x double> %d, <2 x double> %e, <2 x double> %f,
                 <8 x half> %g, <8 x half> %h, <8 x half> %i) {
; CHECK-LABEL: all:
; CHECK: fvmul
; CHECK: fmax
; CHECK: dvfmau
; CHECK: hvadd
; CHECK: dmax
; CHECK: dmin
; CHECK: hmax
; CHECK: hmin
; CHECK: fmfma $lx
; CHECK: fmwrite $
; CHECK: fmread $lx1
  %r1 = call <4 x float> @llvm.mncore.mul.v4f32(<4 x float> %a, <4 x float> %b, i64 0, i64 0)
  %r2 = call <4 x float> @llvm.mncore.max.v4f32(<4 x float> %a, <4 x float> %b, i64 0, i64 0)
  %r3 = call <2 x double> @llvm.mncore.fma.v2f64(<2 x double> %d, <2 x double> %e, <2 x double> %f, i64 0, i64 0)
  %r4 = call <8 x half> @llvm.mncore.add.v8f16(<8 x half> %g, <8 x half> %h, i64 0, i64 0)
  %r5 = call <2 x double> @llvm.mncore.max.v2f64(<2 x double> %d, <2 x double> %e, i64 0, i64 0)
  %r6 = call <2 x double> @llvm.mncore.min.v2f64(<2 x double> %d, <2 x double> %e, i64 0, i64 0)
  %r7 = call <8 x half> @llvm.mncore.max.v8f16(<8 x half> %g, <8 x half> %h, i64 0, i64 0)
  %r8 = call <8 x half> @llvm.mncore.min.v8f16(<8 x half> %g, <8 x half> %h, i64 0, i64 0)
  %v = call <4 x float> @llvm.mncore.matvec.v4f32(i64 0, <4 x float> %r1, <4 x float> %r2)
  call void @llvm.mncore.matwrite.v4f32(i64 0, i64 2, <4 x float> %v)
  %r = call <4 x float> @llvm.mncore.matread.v4f32(i64 0, i64 1)
  ret void
}
