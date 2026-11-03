; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Plain LLVM IR must lower to assemble3-legal .vsm text: physical GRF
; registers print as packed PE operands ($lrNv / $lsNv), operands are
; space-separated (no commas), and every longword address is even.
; Kernel parameters are live in even GRF0 slots (0, 2, 4, ...).

target triple = "mncore2"

define i64 @k_add(i64 %a, i64 %b) {
; CHECK-LABEL: k_add:
; CHECK: ladd $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = add i64 %a, %b
  ret i64 %1
}

define i64 @k_sub(i64 %a, i64 %b) {
; CHECK-LABEL: k_sub:
; CHECK: lsub $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = sub i64 %a, %b
  ret i64 %1
}

define i64 @k_and(i64 %a, i64 %b) {
; CHECK-LABEL: k_and:
; CHECK: land $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = and i64 %a, %b
  ret i64 %1
}

define i64 @k_or(i64 %a, i64 %b) {
; CHECK-LABEL: k_or:
; CHECK: lor $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = or i64 %a, %b
  ret i64 %1
}

define i64 @k_xor(i64 %a, i64 %b) {
; CHECK-LABEL: k_xor:
; CHECK: lxor $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = xor i64 %a, %b
  ret i64 %1
}

define i64 @imm_kernel(i64 %a) {
; CHECK-LABEL: imm_kernel:
; CHECK: imm i"42" $lr{{[0-9]+}}v
; CHECK: ladd $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = add i64 %a, 42
  ret i64 %1
}
