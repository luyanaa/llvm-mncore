# `mncore` MLIR dialect — 设计契约 (2026-11-03)

目标：为 MLIR 前端（Triton/TileLang 原型）提供与 LLVM 后端 **同一 ABI** 的
方言。所有 opcode 数值来自
[`MNCoreIntrinsicOps.h`](../../../../include/llvm/Target/MNCore/MNCoreIntrinsicOps.h)（append-only 稳定枚举）。

## 1. 分层

* `mncore.token`（value type）：Layer A 的 packed 操作数（i64）。只由
  token 构造 op 产生，只能作为指令 op 的操作数。
* 指令 op：与 `llvm.mncore.*` 指令 intrinsic 一一对应，属性携带稳定
  `opcode` 枚举；转换到 LLVM IR 时直接生成对应 intrinsic 调用。
* typed op（`mncore.vec.*` / `mncore.mat.*`）：操作数为 `vector<...>`，
  对应 Layer B。转换时与 C++ pass 遵循同一契约（槽位分配、掩码、ABI）。

## 2. 类型

```
!mncore.token                  // Layer A operand token (i64)
vector<2xf64> / vector<4xf32> / vector<8xf16>   // typed 向量
```

矩阵面用 `i64 face`、行号 `i64 row` 属性/操作数表达（与 Layer B 签名一致）。

## 3. Ops

### 3.1 token 构造（纯、可折叠）
| op | 操作数 | 对应 intrinsic |
|---|---|---|
| `mncore.peop` | space, addr, width, flags, aux | `llvm.mncore.peop` |
| `mncore.lbop` | mem, addr, width, flags | `llvm.mncore.lbop` |
| `mncore.matop` | face, row, width, flags | `llvm.mncore.matop` |
| `mncore.maskop` | kind, value, flags | `llvm.mncore.maskop` |
| `mncore.mvop` | kind, addr, flags | `llvm.mncore.mvop` |
| `mncore.mvparams` | size, tag, nd, prio | `llvm.mncore.mvparams` |

### 3.2 指令
| op | 属性 | 操作数（-1 缺省） | intrinsic |
|---|---|---|---|
| `mncore.alu` | opcode: ALUOp, prec, opt | sx, sy, d0, d1, d2, omr, zmask, wmask | `.alu` |
| `mncore.mau` | opcode: MAUOp, prec | mat, sx, sy, sz, d0..d2, omr, zmask, wmask | `.mau` |
| `mncore.l1b` | opcode: L1BOp, dir | src, dst, set, zmask, wmask | `.l1b` |
| `mncore.l2b` | opcode: L2BOp | set, a, b | `.l2b` |
| `mncore.mv` | opcode: MVOp | params, src, dst | `.mv` |
| `mncore.wait` | tag | — | `.wait` |
| `mncore.fence` | — | — | `.fence` |
| `mncore.maskdir` | spec | — | `.maskdir` |
| `mncore.dget` | mem, addr, nwords | — | backend `DGET` (debug) |

### 3.3 typed（与 Layer B 同名语义）
`mncore.vec.fma/mul/add/max/min`（3/2/2/2/2 输入 + zmask/wmask）、
`mncore.mat.vec(face,x,y)`、`mncore.mat.write(face,row,src)`、
`mncore.mat.read(face,row)`、`mncore.vec.load(lbop)`（L1BM → PE slot，
对应 `llvm.mncore.l1load.*` → `l1bmp`；Triton `ttg.local_load` 的目标）。

## 4. 验证器规则（与 LLVM 后端一致）

* token 构造参数必须是常量；非常量 → 报错（运行期打包不可行，见主文档 §11）。
* 掩码：常量或 `maskop` 结果；常量 0 = 无掩码。
* typed 结果只能被 typed op 使用；向量参数 ABI：第 i 个参数位于
  space = 1+i%4、地址 16*(i/4)（word）。typed 槽位步进 16 word。
* `wait` tag != 0 且必须与其他 PE 表达式共步；`fence` 为一步 `nop`。
* 同表达式同空间两个输入地址必须相同（一个地址字段）；写-写同空间冲突。
* 矩阵 token 的 face 位选择 X/Y def；masked mwrite 非法（ISA 限制）。

## 5. 转换路径

`mncore` dialect → LLVM IR：
1. token 构造 op → `llvm.mncore.*` 调用（常数折叠由后端完成）；
2. 指令 op → 对应 intrinsic 调用（属性→i64 参数）；
3. typed op → 两种选择：
   a. 直接复用 LLVM 侧 `mncore-lower-typed-intrinsics`（推荐，保证与现有
      后端同一契约）；或
   b. 在 MLIR 侧实现同一槽位分配，再生成 Layer A op。
   两条路径必须由共享的 golden 测试对拍（同 IR 输入 → 同 `.vsm`）。

## 6. 里程碑

1. 在启用 MLIR 的构建中加入 dialect 骨架（`MNCoreDialect.td`、token 类型、
   9 个指令 op、6 个构造 op），lit 验证器测试。
2. 转换 pass 到 LLVM IR；与现有 `intrinsics.ll`/`layerb*.ll` 做文本对拍。
3. `d set`/`d get` golden 覆盖扩展到矩阵/向量与数值算子。
4. Triton 原型：`tt.dot`/elementwise/local_alloc → typed op。

---

## 7. 实现状态 (2026-11-03)

本文 §1–§6 的契约已落地为树内 MLIR 代码（`mlir/` 子工程）：

| 内容 | 位置 |
|---|---|
| dialect / 类型 / ops | `mlir/include/mlir/Dialect/MNCore/IR/MNCore{ Dialect,Ops }.td`、`MNCoreOps.h`、`MNCoreTypes.h` |
| 验证器 | `mlir/lib/Dialect/MNCore/IR/MNCoreOps.cpp` |
| `mncore → LLVM` | `mlir/include/mlir/Conversion/MNCoreToLLVM/`、`mlir/lib/Conversion/MNCoreToLLVM/MNCoreToLLVM.cpp` |
| pass 注册 | `mlir/include/mlir/Conversion/Passes.td`（`convert-mncore-to-llvm`） |
| 测试 | `mlir/test/Dialect/MNCore/roundtrip.mlir`、`invalid.mlir`（负向）、`mlir/test/Conversion/MNCoreToLLVM/lower.mlir`、`llvm/test/CodeGen/MNCore/l1load-modes.ll` |

**与初版设计的差异（有意为之）**：

1. **缺省操作数用 `mncore.none` 而不是 `i64 -1`**。`mncore.none` 是显式 token，
   在 MLIR 层可被验证器识别，lowering 时才变成 ABI 哨兵 `-1`。
2. **token 构造参数全部是属性**（`space/addr/width/flags/aux`）。这直接编码
   §11 的结论：运行期打包不可行，非法字段在 MLIR 层即报错。
3. **指令 op 的 token 列表是定长 `Variadic`**，由验证器检查元数
   （alu 8、mau 10、l1b 5、l2b 3、mv 3），避免 `AttrSizedOperandSegments`
   与可选操作数的解析复杂度。
4. **typed op 直接 lower 回 Layer B intrinsic**，槽位分配仍只在
   `MNCoreLowerTypedIntrinsics.cpp` 实现一份。
5. **已提供 `mncore.dma`**（DRAM/PDM → L2BM → L1BM，lower 为已验证的
   `mvp + l2bmb`）与 `mncore.vec.load`（带 `mode`：目前仅
   `Broadcast` → `l1bmp`；`MAB4/MAB16/Distribute` 由验证器拒绝，等待
   shared-layout/PE 映射实现，见 `mncore_frontend_integration.md` §2.3）。
   **`mncore.dget` 暂未提供**：后端尚无对应的 `llvm.mncore.dget` intrinsic
   （`DGET` 目前只是 MIR 伪指令）。数值 golden 的 MLIR 入口在补 intrinsic
   后追加。
6. **typed 值不得逃逸**：`mncore.vec.*`/`mncore.mat.*` 的结果只能被其他 typed op
   消费；返回、store、普通算术都会在 llc 的 Layer B pass 中被拒绝
   （`MNCoreLowerTypedIntrinsics.cpp` 的契约）。前端 kernel 必须以写矩阵/
   LM 或 `d set` 的形式结束，不能返回向量。
7. 目前使用 **generic assembly form**（`"mncore.alu"(%a, %b, ...) {opcode = 0 : i64, ...}`）；
   自定义 pretty syntax 是后续工作，不影响转换与验证。

**构建**：

```sh
cmake -G Ninja -S llvm -B build-mlir \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD= \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=MNCore
ninja -C build-mlir mlir-opt llc
```

**端到端**：

```sh
build-mlir/bin/mlir-opt input.mlir --convert-mncore-to-llvm \
  | build-mlir/bin/mlir-translate --mlir-to-llvmir \
  | build-mlir/bin/llc -mtriple=mncore2 -filetype=asm -o -
```

前端（Triton/TileLang）的接入方式见
[`mncore_frontend_integration.md`](mncore_frontend_integration.md)。

---

## 8. L1BM 分发模式与 verifier 契约 (2026-11-04)

依据 MN-Core 2 Software Developer Manual §3.6.8（.6/.8/.14/.20），
L1BM→PE 的分发不是“一种 load”，而是四种物理传输模式。`mncore.vec.load`
携带显式 `mode`，由前端根据源 memdesc 的 shared layout 与目标 distributed
layout 计算后选择；方言本身不再默认任何模式。

### 8.1 模式与指令

| `mode` | 指令 | 每周期读取 | 语义 |
|---|---|---|---|
| `Broadcast` | `l1bmp` | 1 longword | 同一值广播到全部 64 PE |
| `MAB` | `l1bmm` | 4 longwords | 4 个 PE 各取不同值，16 个 MAB 重复 |
| `MAB4` | `l1bmm4` | 16 longwords | 4 组数据，每组由 4 个 MAB 共享 |
| `Distributed` | `l1bmd` | 64 longwords | 64 个 PE 各取不同值 |

### 8.2 索引映射（m = MAB 0..15，p = MAB 内 PE 0..3）

* `MAB4`：`index(m,p) = 4 * floor(m / 4) + p`
* `Distributed`：`index(m,p) = 4 * m + p`

这两个公式是数值级 layout 回归（`A[m,p] = 100m + p`）的直接依据：
`l1bmm` 在 MAB 0..15 上不随 m 变化；`l1bmm4` 每 4 个 MAB 切换到下一组；
`l1bmd` 随 m 连续推进。

### 8.3 对齐与宽度约束（verifier 必须检查）

| 指令 | 单 longword 基本形式 | double-longword 形式 |
|---|---|---|
| `l1bmp` | 无额外对齐 | 允许，但存在非连续读取与地址范围限制（方言无法静态检查，文档标注） |
| `l1bmm` | 4 longwords | 8 longwords |
| `l1bmm4` | 16 longwords | 32 longwords |
| `l1bmd` | 64 longwords | **不存在**；必须拒绝 |

因此对当前 `<4 x f32>`（2 longwords）：

* `Distributed` 不能用一个等价的双 longword 传输表达；首版必须由 verifier
  拒绝 `mode=Distributed` + `lbop.width=2` 的组合；
* 若将来支持，需要按 longword 平面拆成两条 `l1bmd`，并要求 L1BM/shared
  布局把同一平面连续排放（低 longword 平面、高 longword 平面各自连续）。

### 8.4 与 `ttg.local_load` 的关系

* `ttg.local_alloc` 首版统一映射为 **L1BM allocation**（不同时使用 LM0/LM1，
  避免 memdesc 存储域与 load 指令域不一致）；
* `ttg.local_load` 必须由 `MNCoreLayoutAnalysis` 推导 `mode`，再生成
  `mncore.vec.load`；
* 无法证明布局对应关系时拒绝，而不是回退到 `l1bmp`。

### 8.5 负向测试（必须存在）

* 64 个 PE 预期持有不同值（`Distributed`）却只能生成 `l1bmp` 时，pass 必须
  报错；
* `Distributed` + 双 longword 必须由 verifier 拒绝；
* 各模式地址不满足对齐时必须由 verifier 拒绝。

### 8.6 数值级 layout regression（待模拟器工作线）

输入 `A[m,p] = 100m + p`，读取 PE 侧结果并对拍：

| 指令 | MAB 0 / PE 0..3 | MAB 1 / PE 0..3 | MAB 4 / PE 0..3 |
|---|---|---|---|
| `l1bmp` | A,A,A,A | A,A,A,A | A,A,A,A |
| `l1bmm` | 0,1,2,3 | 0,1,2,3 | 0,1,2,3 |
| `l1bmm4` | 0,1,2,3 | 0,1,2,3 | 16,17,18,19 |
| `l1bmd` | 0,1,2,3 | 100,101,102,103 | 400,401,402,403 |

该回归需要 gpfn3 的 `d set`/`d get` 与矩阵/PE 观测支持（见
`golden_mem.mir` 的现有最小闭环）。
