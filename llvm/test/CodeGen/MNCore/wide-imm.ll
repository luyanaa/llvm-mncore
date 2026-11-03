; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Constants wider than a signed 32-bit word (materialization beyond the
; single `imm` literal). The `imm` word literal is replicated into all
; four word slots (xxxx), `immu` zeros the odd slots (xoxo), so:
;   * low half zero           -> one immu (the word lands in the high half);
;   * the two halves equal    -> one imm;
;   * anything else           -> imm(hi), imm(lo), imm(32), lsl, lor.
; This is the M1 register-syntax bringup path; the assembly-level typed
; literals are covered by imm-fixed-emit.mir.

target triple = "mncore2"

; 0x1122334455667788 — neither half zero nor equal.
define i64 @wide(i64 %a) {
; CHECK-LABEL: wide:
; The register spelling is the legal .vsm PE operand ($lr<addr>v, no
; commas); all longword operands must be 2-aligned.
; CHECK: imm i"32" $lr{{[0-9]+}}v
; CHECK: imm i"287454020" $lr{{[0-9]+}}v
; CHECK: imm i"1432778632" $lr{{[0-9]+}}v
; CHECK: lsl $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
; CHECK: lor $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
; CHECK: ladd $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = add i64 %a, 1234605616436508552
  ret i64 %1
}

; 0x1234567800000000 — low word is zero: xoxo form.
define i64 @lowzero(i64 %a) {
; CHECK-LABEL: lowzero:
; CHECK: immu i"305419896" $lr{{[0-9]+}}v
; CHECK: ladd $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = add i64 %a, 1311768464867721216
  ret i64 %1
}

; 0x1122334411223344 — both halves equal: xxxx form.
define i64 @samehalves(i64 %a) {
; CHECK-LABEL: samehalves:
; CHECK: imm i"287454020" $lr{{[0-9]+}}v
; CHECK: ladd $lr{{[0-9]+}}v $lr{{[0-9]+}}v $lr{{[0-9]+}}v
  %1 = add i64 %a, 1234605615291183940
  ret i64 %1
}

; A signed 32-bit constant keeps the original single-imm path.
define i64 @narrow(i64 %a) {
; CHECK-LABEL: narrow:
; CHECK: imm i"-7" $lr{{[0-9]+}}v
  %1 = add i64 %a, -7
  ret i64 %1
}
