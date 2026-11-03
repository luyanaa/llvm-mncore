# `mncore` MLIR dialect — 设计契约 (2026-11-03)

目标：为 MLIR 前端（Triton/TileLang 原型）提供与 LLVM 后端 **同一 ABI** 的
方言。所有 opcode 数值来自
[`MNCoreIntrinsicOps.h`](../MNCoreIntrinsicOps.h)（append-only 稳定枚举）。

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
`mncore.mat.read(face,row)`。

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
