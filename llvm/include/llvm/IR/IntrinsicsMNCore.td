//===- IntrinsicsMNCore.td - MN-Core2 intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Layer A of the MN-Core2 intrinsic ABI: one intrinsic per PE instruction
// expression (or MV line). Operands are semantic tokens built by the pure
// peop/lbop/matop/maskop/mvop helpers, packed into the .vsm operand encoding
// by the backend. See llvm/lib/Target/MNCore/docs/mncore_intrinsics_design.md.
//
// Sentinel: an absent optional operand is the constant i64 -1.
//
// NOTE: declarations only. Selection (TargetIntrinsicInfo + lowering) lands
// in the M1 milestone; until then these IDs exist for the ABI and for
// frontends to target.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "mncore" in {

//===----------------------------------------------------------------------===//
// Token builders (pure, foldable).
//===----------------------------------------------------------------------===//

// PE operand: space, addr, width, flags, aux.
//   space: 1=GRF0 2=GRF1 3=LM0 4=LM1 5=T
//   width: 0=word 1=longword 2=2-longword 3=T (implicit 2-lw)
//   flags: 1='-' 2='e' 4='r' 8=v 16=j 32=t-indirect 64=flat
//   aux  : j<madpe> or v<adri>, else 0
def int_mncore_peop
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

// L1B/L2BM operand: mem (0=L1B, 1=L2BM), addr, width, flags (1=ll,2=turn).
def int_mncore_lbop
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

// Matrix operand: face (0=x,1=y), row/column, width, flags.
def int_mncore_matop
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

// Mask specifier: kind (0=fixed pattern, 1=$imr), value, flags
// (1=ll, 2=zero-flush, 4='t', 8='p').
def int_mncore_maskop
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

// MV operand: kind (0=PDM 1=DRAM 2=L2BM 3=DRAMi), addr, flags.
def int_mncore_mvop
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

// MV parameters: size, tag, nd, priority.
def int_mncore_mvparams
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, IntrSpeculatable]>;

//===----------------------------------------------------------------------===//
// Layer B: typed vector intrinsics (lowered to Layer A by the
// mncore-lower-typed-intrinsics pass, which performs PE slot allocation).
//===----------------------------------------------------------------------===//

class MNVec3<LLVMType vt> : DefaultAttrsIntrinsic<[vt],
    [vt, vt, vt, llvm_i64_ty, llvm_i64_ty],
    [IntrHasSideEffects, IntrInaccessibleMemOnly]>;
class MNCVec2<LLVMType vt> : DefaultAttrsIntrinsic<[vt],
    [vt, vt, llvm_i64_ty, llvm_i64_ty],
    [IntrHasSideEffects, IntrInaccessibleMemOnly]>;
class MNMatVec<LLVMType vt> : DefaultAttrsIntrinsic<[vt],
    [llvm_i64_ty, vt, vt], [IntrHasSideEffects, IntrInaccessibleMemOnly]>;
class MNMatWrite<LLVMType vt> : DefaultAttrsIntrinsic<[],
    [llvm_i64_ty, llvm_i64_ty, vt],
    [IntrHasSideEffects, IntrInaccessibleMemOnly]>;
class MNMatRead<LLVMType vt> : DefaultAttrsIntrinsic<[vt],
    [llvm_i64_ty, llvm_i64_ty], [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

def int_mncore_fma_v2f64 : MNVec3<llvm_v2f64_ty>;
def int_mncore_mul_v2f64 : MNCVec2<llvm_v2f64_ty>;
def int_mncore_add_v2f64 : MNCVec2<llvm_v2f64_ty>;
def int_mncore_max_v2f64 : MNCVec2<llvm_v2f64_ty>;
def int_mncore_min_v2f64 : MNCVec2<llvm_v2f64_ty>;
def int_mncore_fma_v4f32 : MNVec3<llvm_v4f32_ty>;
def int_mncore_mul_v4f32 : MNCVec2<llvm_v4f32_ty>;
def int_mncore_add_v4f32 : MNCVec2<llvm_v4f32_ty>;
def int_mncore_max_v4f32 : MNCVec2<llvm_v4f32_ty>;
def int_mncore_min_v4f32 : MNCVec2<llvm_v4f32_ty>;
def int_mncore_fma_v8f16 : MNVec3<llvm_v8f16_ty>;
def int_mncore_mul_v8f16 : MNCVec2<llvm_v8f16_ty>;
def int_mncore_add_v8f16 : MNCVec2<llvm_v8f16_ty>;
def int_mncore_max_v8f16 : MNCVec2<llvm_v8f16_ty>;
def int_mncore_min_v8f16 : MNCVec2<llvm_v8f16_ty>;

def int_mncore_matvec_v2f64 : MNMatVec<llvm_v2f64_ty>;
def int_mncore_matvec_v4f32 : MNMatVec<llvm_v4f32_ty>;
def int_mncore_matvec_v8f16 : MNMatVec<llvm_v8f16_ty>;
def int_mncore_matwrite_v2f64 : MNMatWrite<llvm_v2f64_ty>;
def int_mncore_matwrite_v4f32 : MNMatWrite<llvm_v4f32_ty>;
def int_mncore_matwrite_v8f16 : MNMatWrite<llvm_v8f16_ty>;
def int_mncore_matread_v2f64 : MNMatRead<llvm_v2f64_ty>;
def int_mncore_matread_v4f32 : MNMatRead<llvm_v4f32_ty>;
def int_mncore_matread_v8f16 : MNMatRead<llvm_v8f16_ty>;

// Layer B memory access: L1BM -> PE slot. Arguments:
//   mode: 0=broadcast(l1bmp) 1=MAB(l1bmm) 2=MAB4(l1bmm4) 3=distributed(l1bmd)
//   lbop: the L1BM address token
//   zmask/wmask: i64 sentinels (-1 = absent)
// The PE slot is allocated by mncore-lower-typed-intrinsics.
class MNVecLoad<LLVMType vt> : DefaultAttrsIntrinsic<[vt],
    [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
    [IntrHasSideEffects, IntrInaccessibleMemOnly]>;
def int_mncore_l1load_v2f64 : MNVecLoad<llvm_v2f64_ty>;
def int_mncore_l1load_v4f32 : MNVecLoad<llvm_v4f32_ty>;
def int_mncore_l1load_v8f16 : MNVecLoad<llvm_v8f16_ty>;

//===----------------------------------------------------------------------===//
// Instruction expressions.
//===----------------------------------------------------------------------===//

// ALU expression (3.6.12). opcode/prec/opt select the concrete def; the
// d0..d2 and omr operands are PE destinations (-1 = absent), zmask/wmask
// are mask specifiers (-1 = absent).
def int_mncore_alu
    : DefaultAttrsIntrinsic<[],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// MAU expression (3.6.9-3.6.11): matvec/vector/mwrite/mread share this.
def int_mncore_mau
    : DefaultAttrsIntrinsic<[],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// L1BM expression (3.6.8). dir: 1=L1B->PE, 2=PE->L1B.
def int_mncore_l1b
    : DefaultAttrsIntrinsic<[],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
         llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// L2BM expression (3.6.7): set, first operand, second operand.
def int_mncore_l2b
    : DefaultAttrsIntrinsic<[],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// MV command (3.5): params, src, dst.
def int_mncore_mv
    : DefaultAttrsIntrinsic<[],
        [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

//===----------------------------------------------------------------------===//
// Control.
//===----------------------------------------------------------------------===//

// wait <tag>: must co-issue with another PE expression (3.6.13).
def int_mncore_wait
    : DefaultAttrsIntrinsic<[], [llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// Multi-line write-mask directive (3.6.2.1); a control line, not an expr.
def int_mncore_maskdir
    : DefaultAttrsIntrinsic<[], [llvm_i64_ty],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

// Ordering barrier for frontends: no instruction intrinsic moves across it.
def int_mncore_fence
    : DefaultAttrsIntrinsic<[], [],
        [IntrHasSideEffects, IntrInaccessibleMemOnly]>;

}
