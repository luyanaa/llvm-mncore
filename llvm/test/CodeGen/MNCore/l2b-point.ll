; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; L1BM -> L2BM point transfer: `l2bm@<l1badr> $lb<addr> $lc<addr>` (opcode 2).
; Unlike the collect (l2bmd, opcode 1) this copies exactly one L1B's stream,
; so a replicated plane stays contiguous in L2BM. The L1B selector rides in
; the intrinsic's second argument.

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare void @llvm.mncore.l2b(i64, i64, i64, i64)

define void @point_l1b0() {
; CHECK-LABEL: point_l1b0:
; CHECK: l2bm@0 $lb256 $lc256
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 256, i64 1, i64 0)
  %lc = call i64 @llvm.mncore.lbop(i64 1, i64 256, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 2, i64 0, i64 %lb, i64 %lc)
  ret void
}

define void @point_l1b5() {
; CHECK-LABEL: point_l1b5:
; CHECK: l2bm@5 $lb320 $lc384
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 320, i64 1, i64 0)
  %lc = call i64 @llvm.mncore.lbop(i64 1, i64 384, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 2, i64 5, i64 %lb, i64 %lc)
  ret void
}
