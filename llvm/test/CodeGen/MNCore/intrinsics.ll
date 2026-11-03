; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
; The chain operand of a machine node is the LAST operand; InstrEmitter
; ignores a trailing MVT::Other and treats a leading one as a register.
; The custom RET carries its chain last and retargets the DAG root with
; ReplaceNode, so side-effecting intrinsics stay reachable and emit.
;
; llvm.mncore.* Layer A, first slice: fence lowers to one nop step (a
; conservative ordering barrier) and wait to WAIT with its constant tag.
; Token builders fold to constants; a dynamic operand is rejected.

target triple = "mncore2"

declare void @llvm.mncore.fence()
declare void @llvm.mncore.wait(i64)

define void @ctrl() {
; CHECK-LABEL: ctrl:
; CHECK: nop
; CHECK: wait i05
  call void @llvm.mncore.fence()
  call void @llvm.mncore.wait(i64 5)
  ret void
}

declare i64 @llvm.mncore.peop(i64, i64, i64, i64, i64)
declare i64 @llvm.mncore.lbop(i64, i64, i64, i64)
declare i64 @llvm.mncore.mvop(i64, i64, i64)
declare i64 @llvm.mncore.mvparams(i64, i64, i64, i64)
declare void @llvm.mncore.alu(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @llvm.mncore.l1b(i64, i64, i64, i64, i64, i64, i64)
declare void @llvm.mncore.mv(i64, i64, i64, i64)

; ALU: opcode 0 = fmin, single destination, no masks.
define void @alu1() {
; CHECK-LABEL: alu1:
; CHECK: fmin $lm0v $lm0v $lr0v
  %m = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 8, i64 0)
  %r = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.alu(i64 0, i64 2, i64 0, i64 %m, i64 %m, i64 %r,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

; L1BM: opcode 0 = l1bmp (L1B->PE), $lb0 -> $lr0v.
define void @l1b1() {
; CHECK-LABEL: l1b1:
; CHECK: l1bmp $lb0 $lr0v
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  %r = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.l1b(i64 0, i64 1, i64 %lb, i64 %r,
                             i64 -1, i64 -1, i64 -1)
  ret void
}

; MV: opcode 0 = mvp, L2BM@0 -> DRAM@0.
define void @mv1() {
; CHECK-LABEL: mv1:
; CHECK: mvp/n64i01 $lc0 $d0
  %s = call i64 @llvm.mncore.mvop(i64 2, i64 0, i64 0)
  %d = call i64 @llvm.mncore.mvop(i64 1, i64 0, i64 0)
  %p = call i64 @llvm.mncore.mvparams(i64 64, i64 1, i64 -1, i64 -1)
  call void @llvm.mncore.mv(i64 0, i64 %p, i64 %s, i64 %d)
  ret void
}

declare void @llvm.mncore.mau(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @llvm.mncore.l2b(i64, i64, i64, i64)
declare void @llvm.mncore.maskdir(i64)

; MAU vector: opcode 0 = fvfma (sx, sy, sz, dst).
define void @mau_vec() {
; CHECK-LABEL: mau_vec:
; CHECK: fvfma $lm0v $ln0v $lr0v $ls0v
  %a = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 8, i64 0)
  %b = call i64 @llvm.mncore.peop(i64 4, i64 0, i64 1, i64 8, i64 0)
  %c = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  %d = call i64 @llvm.mncore.peop(i64 2, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.mau(i64 0, i64 2, i64 0, i64 %a, i64 %b, i64 %c, i64 %d,
                            i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

; MAU matrix: opcode 4 = matvec, 5 = mwrite, 6 = mread.
define void @mau_matvec() {
; CHECK-LABEL: mau_matvec:
; CHECK: fmfma $lx $ln0v $lr0v $ls0v
  %m = call i64 @llvm.mncore.matop(i64 0, i64 0, i64 1, i64 0)
  %x = call i64 @llvm.mncore.peop(i64 4, i64 0, i64 1, i64 8, i64 0)
  %y = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  %d = call i64 @llvm.mncore.peop(i64 2, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.mau(i64 4, i64 2, i64 %m, i64 %x, i64 %y, i64 -1, i64 %d,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

define void @mau_mwrite() {
; CHECK-LABEL: mau_mwrite:
; CHECK: fmwrite $lm0v $lx0
  %m = call i64 @llvm.mncore.matop(i64 0, i64 0, i64 1, i64 0)
  %s = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.mau(i64 5, i64 1, i64 %m, i64 %s, i64 -1, i64 -1, i64 -1,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

define void @mau_mread() {
; CHECK-LABEL: mau_mread:
; CHECK: fmread $lx0 $lr0v
  %m = call i64 @llvm.mncore.matop(i64 0, i64 0, i64 1, i64 0)
  %d = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.mau(i64 6, i64 1, i64 %m, i64 -1, i64 -1, i64 -1, i64 %d,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

; L2BM: opcode 0 = l2bmb, set=0 (all L1Bs), $lc0 -> $lb0.
define void @l2b1() {
; CHECK-LABEL: l2b1:
; CHECK: l2bmb $lc0 $lb0
  %c = call i64 @llvm.mncore.lbop(i64 1, i64 0, i64 1, i64 0)
  %b = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  call void @llvm.mncore.l2b(i64 0, i64 0, i64 %c, i64 %b)
  ret void
}

; Multi-line write mask: spec 646 = maskrs 5 (r, s enabled, entry 5).
define void @maskdir1() {
; CHECK-LABEL: maskdir1:
; CHECK: maskrs 5
  call void @llvm.mncore.maskdir(i64 646)
  ret void
}

; Dynamic addressing (T-indirect, 3.6.1.6): the LM0 address comes from the
; T register at run time, so no run-time packing of the operand field is
; needed — the token is still a constant ($lmt).
define void @tindirect() {
; CHECK-LABEL: tindirect:
; CHECK: fmin $lmt $ln0v $lr0v
  %t = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 32, i64 0)
  %y = call i64 @llvm.mncore.peop(i64 4, i64 0, i64 1, i64 8, i64 0)
  %d = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.alu(i64 0, i64 2, i64 0, i64 %t, i64 %y, i64 %d,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}

; Masked ALU and masked L1BM broadcast.
define void @masked_ops() {
; CHECK-LABEL: masked_ops:
; CHECK: fmin $lm0v $ln0v $lr0v/0111
; CHECK: l1bmp $lb0 $lr0v/0111
  %a = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 8, i64 0)
  %b = call i64 @llvm.mncore.peop(i64 4, i64 0, i64 1, i64 8, i64 0)
  %d = call i64 @llvm.mncore.peop(i64 1, i64 0, i64 1, i64 8, i64 0)
  call void @llvm.mncore.alu(i64 0, i64 2, i64 0, i64 %a, i64 %b, i64 %d,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 57)
  %lb = call i64 @llvm.mncore.lbop(i64 0, i64 0, i64 1, i64 0)
  call void @llvm.mncore.l1b(i64 0, i64 1, i64 %lb, i64 %d,
                             i64 -1, i64 -1, i64 57)
  ret void
}

; LM base-address register as an ALU destination (3.6.1.7): a peop with
; flags 256 and no address prints $lmb.
define void @lmbar() {
; CHECK-LABEL: lmbar:
; CHECK: lpassa $ls0v $lmb
  %s = call i64 @llvm.mncore.peop(i64 2, i64 0, i64 1, i64 8, i64 0)
  %b = call i64 @llvm.mncore.peop(i64 3, i64 0, i64 1, i64 256, i64 0)
  call void @llvm.mncore.alu(i64 4, i64 2, i64 0, i64 %s, i64 -1, i64 %b,
                             i64 -1, i64 -1, i64 -1, i64 -1, i64 -1)
  ret void
}
