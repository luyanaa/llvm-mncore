; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; L1BM -> L2BM collect: l2bmd $lb<addr> $lc<addr> (opcode 1).

target triple = "mncore2"

declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare void @llvm.mncore.l2b(i64, i64, i64, i64)

define void @collect() {
; CHECK-LABEL: collect:
; CHECK: l2bmd $lb0 $lc0
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  %lc = call i64 @llvm.mncore.lbop(i64 1, i64 0, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 1, i64 0, i64 %lb, i64 %lc)
  ret void
}
