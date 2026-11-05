; RUN: llc -mtriple=mncore2 %s -o - | FileCheck %s
;
; Relocatable MV operands (mncore_kernel_abi_v1.md §13). `llvm.mncore.mv.reloc`
; is `llvm.mncore.mv` plus one packed relocation record. The record's operand is
; emitted with a placeholder address 0, the line carries a
; `# @mncore.fixup <site>` comment, and the YAML sidecar block at end of file
; describes each site. The instruction text itself is unchanged, so assemble3
; still accepts the stream after the comments are stripped (§13.3 step 4).
;
; Site numbers restart at 0 for each kernel entry and the sidecar groups its
; records by entry, so the primary key is (kernel entry, site). A module-wide
; emission order would make a bare site number ambiguous once a module has
; more than one entry.

target triple = "mncore2"

declare i64 @llvm.mncore.mvop(i64, i64, i64)
declare i64 @llvm.mncore.mvparams(i64, i64, i64, i64)
declare void @llvm.mncore.mv(i64, i64, i64, i64)
declare void @llvm.mncore.mv.reloc(i64, i64, i64, i64, i64)

; arg 0, DRAM, longword, width 32, operand 1 (src), mvp, addend 0.
define void @reloc_src() {
; CHECK-LABEL: reloc_src:
; CHECK: # @mncore.kernel reloc_src
; CHECK: mvp/n64i05 $d0 $lc256 # @mncore.fixup 0
  %src = call i64 @llvm.mncore.mvop(i64 1, i64 0, i64 0)
  %dst = call i64 @llvm.mncore.mvop(i64 2, i64 256, i64 0)
  %p = call i64 @llvm.mncore.mvparams(i64 64, i64 5, i64 -1, i64 -1)
  call void @llvm.mncore.mv.reloc(i64 0, i64 %p, i64 %src, i64 %dst,
                                 i64 1179648)
  ret void
}

; A non-relocated MV line keeps the plain form and gets no comment.
define void @absolute_src() {
; CHECK-LABEL: absolute_src:
; CHECK: # @mncore.kernel absolute_src
; CHECK: mvp/n64i06 $d128 $lc256
; CHECK-NOT: @mncore.fixup
  %src = call i64 @llvm.mncore.mvop(i64 1, i64 128, i64 0)
  %dst = call i64 @llvm.mncore.mvop(i64 2, i64 256, i64 0)
  %p = call i64 @llvm.mncore.mvparams(i64 64, i64 6, i64 -1, i64 -1)
  call void @llvm.mncore.mv(i64 0, i64 %p, i64 %src, i64 %dst)
  ret void
}

; arg 2, DRAM, longword, width 32, operand 2 (dst), mvp, addend 64: the DRAM
; destination of a writeback. This is a different kernel, so its site number
; restarts at 0 rather than continuing at 1.
define void @reloc_dst() {
; CHECK-LABEL: reloc_dst:
; CHECK: # @mncore.kernel reloc_dst
; CHECK: mvp/n64i07 $lc256 $d0 # @mncore.fixup 0
  %src = call i64 @llvm.mncore.mvop(i64 2, i64 256, i64 0)
  %dst = call i64 @llvm.mncore.mvop(i64 1, i64 0, i64 0)
  %p = call i64 @llvm.mncore.mvparams(i64 64, i64 7, i64 -1, i64 -1)
  call void @llvm.mncore.mv.reloc(i64 0, i64 %p, i64 %src, i64 %dst,
                                 i64 274880135170)
  ret void
}

; The sidecar is one block at end of file, grouped by kernel entry.
; CHECK: # @mncore.fixup.begin
; CHECK-NEXT: # kernel: reloc_src
; CHECK-NEXT: # fixups:
; CHECK-NEXT: #   - site: 0
; CHECK-NEXT: #     arg: 0
; CHECK-NEXT: #     space: DRAM
; CHECK-NEXT: #     unit: longword
; CHECK-NEXT: #     addend_units: 0
; CHECK-NEXT: #     width_bits: 32
; CHECK-NEXT: #     opcode: mvp
; CHECK-NEXT: #     operand: 1
; CHECK-NEXT: # kernel: reloc_dst
; CHECK-NEXT: # fixups:
; CHECK-NEXT: #   - site: 0
; CHECK-NEXT: #     arg: 2
; CHECK-NEXT: #     space: DRAM
; CHECK-NEXT: #     unit: longword
; CHECK-NEXT: #     addend_units: 64
; CHECK-NEXT: #     width_bits: 32
; CHECK-NEXT: #     opcode: mvp
; CHECK-NEXT: #     operand: 2
; CHECK-NEXT: # @mncore.fixup.end
