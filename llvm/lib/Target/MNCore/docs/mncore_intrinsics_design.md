# MN-Core2 `llvm.mncore.*` intrinsic ABI — 设计草案 (2026-11-03)

目标：让 LLVM IR 能表达 MN-Core2 的**设备内存语义**、**PE 向量运算**与**矩阵寄存器运算**，
使 Triton（及后续 TileLang/OpenCL）可以 offload 到本后端，同时不引入通用 global
address space。

本文定义两层 ABI：

* **Layer A（raw / backend contract）**：操作数是*已解包的语义 token*（`i64`），
  一个 intrinsic 恰好对应一条 PE 指令表达式或一条 MV 行。后端负责把 token
  打包成 `.vsm` packed 操作数。**这是唯一被后端实现和验证的层。**
* **Layer B（typed / frontend-facing）**：向量/矩阵值直接作为 SSA 操作数，
  由未来的 `mncore` dialect pass 做 **PE 槽位分配**后降级为 Layer A。
  Layer B 的签名在本文中定稿，但实现依赖“PE 槽位分配器”（见 §8）。

---

## 1. 非目标

* 不把 GRF/LM/L1BM/L2BM 暴露为通用 `ptr addrspace(N)` 去参与 GEP/load/store。
  LLVM 指针假定字节寻址，而 MN-Core 各空间按 word/longword 寻址、宽度与对齐
  规则由操作数本身决定；用通用指针会把 §3.6.1 的访问模式埋在 alias 分析里。
* 不在 intrinsic 内部做多指令展开。需要展开的（宽常量、掩码序列、tag/wait 配对）
  由专门 pass 显式完成，保证 packetizer/hazard 仍能看到每一步。
* 不承诺 MemorySSA/alias 精度。设备内存的读写顺序由 `mncore.fence` / tag 语义
  **显式**表达，而不是靠 LLVM 的普通内存模型推断。

## 2. 地址与操作数模型（Layer A）

所有地址都是**该操作数访问宽度下的单位地址**（word / longword / 2-longword），
与手册 §3.2.1 一致。地址是 `i64` 立即数或运行期值；后端在发射时再打包。

### 2.1 空间编号

| id | 空间 | 单位 | 备注 |
|---:|---|---|---|
| 1 | GRF0 | word | `$r` |
| 2 | GRF1 | word | `$s` |
| 3 | LM0 | word | `$m`；BAR 形式为 `<l>mb` |
| 4 | LM1 | word | `$n`；BAR 形式为 `<l>nb` |
| 5 | T | — | 无地址，恒 2 长语 |
| 6 | L1BM | longword | `$lb` |
| 7 | L2BM | longword | `$lc` |
| 8 | PDM | longword | `$p` |
| 9 | DRAM | longword | `$d` / `$di` |
| 10 | MASKREG | entry | 只写作目的操作数 |
| 11 | MATRIX | row/column | 由 face + row 组成 |

宽度编码（`width` 参数）：`0=word, 1=longword, 2=2-longword, 3=T(隐式 2 长语)`。

### 2.2 token 构造 intrinsic（纯函数，可常量折叠）

```llvm
; PE 操作数。flags: 1='-' 2='e' 4='r' 8=v 16=j 32=t-indirect 64=flat
; aux: j 的 madpe，或 v<adri> 的 adri（否则 0）
i64 @llvm.mncore.peop(i32 space, i64 addr, i32 width, i32 flags, i64 aux)
; L1B/L2BM 操作数。mem: 0=L1B 1=L2BM；flags: 1=ll 2=turnaround($lbi)
i64 @llvm.mncore.lbop(i32 mem, i64 addr, i32 width, i32 flags)
; 矩阵操作数：写/转置读带 row；matvec 用 face 整面（row 忽略）
i64 @llvm.mncore.matop(i32 face, i64 row, i32 width, i32 flags)
; 掩码指定符。kind: 0=fixed 4-bit 1=$imr；flags: 1=ll 2=zero-flush 4=t 8=p
i64 @llvm.mncore.maskop(i32 kind, i32 value, i32 flags)
; MV 操作数与参数（§3.5）
i64 @llvm.mncore.mvop(i32 kind, i64 addr, i32 flags)
i64 @llvm.mncore.mvparams(i64 size, i32 tag, i32 nd, i32 prio)
```

后端在 lowering 时**只**通过一个共享函数 `packPEOperand(token)` 生成 packed 立即数；
`MNCoreRegions.h` 的解码是该函数的逆。两层必须共用同一份字段定义（用户 #4 要求）。

## 3. 指令 intrinsic（Layer A）

每条 intrinsic 对应*一条*表达式；无返回值，带内存/副作用属性。缺省操作数用
`i64 -1` 表示“不存在”（例如无掩码、无 OMR 目的操作数）。

```llvm
; ALU（§3.6.12）。opcode 为本后端 MNCoreInstrALU.td 中的枚举 id。
void @llvm.mncore.alu(i32 opcode, i32 prec, i32 opt,
                      i64 sx, i64 sy, i64 d0, i64 d1, i64 d2,
                      i64 omr, i64 zmask, i64 wmask)
; MAU（§3.6.9–3.6.11）。mat 为 matop；向量/矩阵形式共用。
void @llvm.mncore.mau(i32 opcode, i32 prec,
                      i64 mat, i64 sx, i64 sy, i64 sz,
                      i64 d0, i64 d1, i64 d2, i64 omr,
                      i64 zmask, i64 wmask)
; L1BM（§3.6.8）。dir: 1=L1B->PE 2=PE->L1B；src/dst 为 lbop/peop
void @llvm.mncore.l1b(i32 opcode, i32 dir, i64 src, i64 dst,
                      i64 set, i64 zmask, i64 wmask)
; L2BM（§3.6.7）。set 为 L1B 子集（缺省 0xff 语义）
void @llvm.mncore.l2b(i32 opcode, i64 set, i64 a, i64 b)
; MV（§3.5）。opcode: mvp/mvb/mvb2/mvb4/mvd/mvr<op>…
void @llvm.mncore.mv(i32 opcode, i64 params, i64 src, i64 dst)
; 控制
void @llvm.mncore.wait(i32 tag)          ; 必须与其他 PE 表达式共步（§3.6.13）
void @llvm.mncore.maskdir(i32 spec)      ; 多行写掩码指令（§3.6.2.1）
void @llvm.mncore.fence()                ; 调度/重排屏障（前端保证顺序用）
```

### 3.1 属性与顺序

* `peop/lbop/matop/maskop/mvop/mvparams`：`[IntrNoMem, IntrSpeculatable,
  IntrWillReturn]`，可折叠。
* 指令 intrinsic：`[IntrHasSideEffects, IntrInaccessibleMemOnly]`；
  搬运类另加 `[IntrReadMem]` 或 `[IntrWriteMem]`（读改写如 `mwrite` 两者都加）。
* **顺序契约**：后端按 IR 顺序形成 MIR，再由 packetizer/hazard 调度；
  LLVM 不得跨越 `fence` 重排指令 intrinsic。tag/wait 的配对由前端或一个
  “tag 分配 pass”完成，后端只验证（`wait` 的 tag 已存在且非 0）。
* 每个 intrinsic 的 token 参数必须是 `peop/lbop/...` 的调用或常量；后端 verifier
  在 lowering 前检查（错误信息带 space/addr/width），避免把非法组合带进 AsmPrinter。

### 3.2 与现有 MI 的映射

| intrinsic | MI opcode 家族 | 备注 |
|---|---|---|
| `alu` | `MNCoreInstrALU.td`（含 `_M` 掩码孪生） | opcode/prec/opt → 具体 def |
| `mau` | `MNCoreInstrMAU.td`（含 `matface2` 多输出） | |
| `l1b` | `MNCoreInstrL1BM.td`（含多输出、掩码、turnaround） | `set` 仅 L2BM 侧 |
| `l2b` | `MNCoreInstrL2BM.td` | |
| `mv` | `MNCoreInstrMV.td` | 参数打包复用 `printMVParams` |
| `maskdir` | `MASK_DIR` | 独立行，不占步 |

实现路径：`SelectionDAGISel` 中新增 `Intrinsic` 分支 → 调 `packPEOperand` →
`getMachineNode(opcode, ...)`；或写 TableGen `Pat<(int_mncore_alu ...), ...>`。
多输出/可变掩码组合较多，建议统一走 C++ 选择函数，TableGen 仅用于简单形式。

## 4. Vector 语义

MN-Core 的“向量”是**每 PE 每步 2 长语**的数据通路，不是打包 SIMD 指令：

| 精度 | 每 2 长语元素数 | IR 向量类型（Layer B） |
|---|---:|---|
| d (f64) | 2 | `<2 x double>` |
| f (f32) | 4 | `<4 x float>` |
| g (pseudo-single) | 4 | `<4 x float>`（数值语义另定） |
| h (f16) | 8 | `<8 x half>` |

Layer B 的 `fma/mul/add/max/min/relu/cvt` 以这些向量类型为操作数，例如：

```llvm
%r = call <4 x float> @llvm.mncore.fma.v4f32(
        <4 x float> %a, <4 x float> %b, <4 x float> %c,
        i64 %zmask, i64 %wmask)
```

降级：`mncore` pass 为每个 SSA 值分配 PE 槽位（GRF0/GRF1/LM0/LM1 的地址 +
宽度），把读写地址变成 `peop` token，再发射 `mau`。**在槽位分配器完成前，Layer B
只存在于设计，不接后端**。

## 5. Matrix 语义

矩阵寄存器每 MAB 两面（x/y），物理 16 行；逻辑行数：d=4、f/g=8、h=16。
`matop(face,row,width,flags)` 足以表达：

* `mwrite`：`mau(opcode=mwrite, src=peop, mat=matop)`（§3.6.10）
* `mread`：`mau(opcode=mread, mat=matop, d0=peop)`（§3.6.11）
* `matvec`：`mau(opcode=mfma/mmul, mat=matop(face,row=0,width), sx, sy, d0…)`

Layer B（面向 Triton）：

```llvm
; 把 <rows x cols> 的 tile 写入本 PE 的矩阵面；rows 由精度决定
void  @llvm.mncore.mat.write.v4f32(i64 %face, i64 %row, <4 x float> %src)
; 转置读出一行到向量
<4 x float> @llvm.mncore.mat.read.v4f32(i64 %face, i64 %row, i64 %zmask)
; 矩阵-向量积（每 PE 的 face × 向量 → 向量）
<4 x float> @llvm.mncore.matvec.v4f32(i64 %face, i64 %zmask, <4 x float> %x)
```

## 6. 内存搬运语义（DRAM/L2BM/L1BM/LM）

* 层级：DRAM/PDM ⇄ L2BM（`mv`）→ L1BM（`l2b`）→ PE 存储（`l1b`）。
* **静态地址优先**：Layer A 的地址可以是常量或运行期 `i64`，但后端在打包时要求
  宽度/对齐可静态验证的部分必须静态（§3.6.1、§3.6.7.1）；无法静态验证的组合
  直接拒绝，而不是交由 assemble3 晚失败。
* **tag/wait**：`mv` 的 `params` 携带 tag；`wait` 与之配对。README 中的
  “MV 行与 PE 行同流”由 AsmPrinter 的步骤布局保证。
* **不引入 global address space**：DRAM 访问必须经过 `mvop(space=9)` + `mv`；
  Triton 的指针参数在 typed 层被描述为 `!mncore.tensor<...>` 描述符，由
  `mncore` pass 决定放在 DRAM 还是 PDM，并生成 `mv` 序列。

## 7. Triton offload 映射（目标形态）

证明该 ABI 足以承载 Triton，而不需要 LLVM IR 的通用指针：

| Triton（TTIR/TTGIR） | 本后端 |
|---|---|
| `tt.load` / `tt.store`（DRAM tensor） | `mncore.dma` → `mv` + `l2b` + `l1b` 序列 |
| `ttg.local_alloc`（shared） | LM0/LM1 槽位（`peop` 3/4） |
| `ttg.local_load/store` | `l1b`/`alu passa` 或 `peop` 直接访问 |
| `tt.dot` | `mau(mwrite)` + `mau(matvec mfma)` + `mau(mread)` |
| elementwise（fma/mul/add/max/min） | `mau` 向量形式（Layer B → A） |
| `ttg.convert_layout` | `l1bmp/l1bmm/l1bmd`（广播/分配/收集） |
| mask/select | `maskop` + `wmask/zmask` 参数 |
| barrier | `fence` + tag/wait |

需要一个轻量 `mncore` dialect（或在 Triton 后端里直接生成 Layer B 调用），负责：
PE 槽位分配、精度/宽度选择、block → 每 PE 2 长语的切分、tag 分配、
（未来）double buffering。

## 8. 分阶段实现

1. **M1（骨架）**：Layer A token intrinsics + `fence`/`wait`/`maskdir`；
   用 C++ 选择器实现 `alu`（单输出）、`l1b`（broadcast）、`mv` 三条通路。
   验证：IR → FileCheck（文本）→ assemble3（strict hazard）。
2. **M2**：`mau` 向量与矩阵（mwrite/mread/matvec）、多输出与 OMR。
3. **M3**：Layer B 语义 intrinsic 与 **PE 槽位分配器**（GRF/LM 地址、宽度、
   live range；受 §3.6.4 单地址/单写者约束），Triton `ttgir → mncore` 原型。
4. **M4**：DRAM/PDM 描述符、`mv` 参数与 tag 分配、静态内存规划、
   double buffering（PIKG 的 head/main/tail 契约作为设计参考）。
5. **M5**：端到端：`llc` → assemble3(strict) → gpfn3，数值对拍（`--dump` +
   `check_dump.py`）。

## 9. 验证准则

* 每个 intrinsic 至少一个 `FileCheck` 文本测试；合法/非法各一。
* 非法用例必须由**后端**拒绝（token 空间/宽度/对齐、wait tag、imm 与 LM0 冲突），
  不得依赖 assemble3 报错。
* 远端必须跑 `run_regression.sh --strict`（hazard + deadlock checker 开启）；
  `emulate_gpfn3.sh --dump` 的结果用 `check_dump.py` 做数值断言。

## 10. 尚待定案

* Layer A 的 `opcode` 枚举是否直接复用 MI opcode 编号（本文假定复用，需在
  TableGen 中导出稳定编号）。
* `i64 -1` 缺省值 vs `optional` 参数：LLVM intrinsic 不支持 optional，
  故用哨兵；需要 verifier 拒绝 `-1` 出现在必填位置以外。
* Layer B 的 PE 槽位分配与 §3.6.4 的跨表达式约束如何交互（可能需要在分配器里
  直接把“同一步的单地址/单写者”作为图着色约束）。
* 是否把 `mask` 多行指令建模为 intrinsic（当前 `maskdir`）还是函数级元数据。

## 11. 运行期打包的可行性结论 (2026-11-03)

结论：**任意 token 字段的运行期打包不可行，也不需要**。

* `.vsm` 的 PE 操作数地址是*指令内静态编码*的（`$lm0v`、`$lr2v` 等），
  没有“地址来自通用寄存器”的形式；`v<adri>`/flat 也只是静态的多地址
  编码。掩码图案/`$imr` 地址、MV 参数同理。
* 运行期可变的地址只有 ISA 明确提供的间接形式：
  * **T-indirect**（仅 LM0）：`$lmt`，地址由 T 寄存器在运行期给出。
    已接线：`peop(space=3, addr=0, width=1, flags=32)`（bit48）→ `$lmt`，
    经 assemble3 验证。
  * **LM 基地址寄存器（BAR）**：`$[l]mb` / `$[l]nb`，已建模（见 §13）。
* 因此 Layer A 继续要求 token 为常量；前端要表达动态地址时，应先写入 T
  或 BAR，再使用相应的静态 token。动态*mask 条目*可用 `$imr<adr>`
  （地址本身仍是静态的 1..15）。

## 12. Layer B 首个垂直切片 (2026-11-03)

已实现：`llvm.mncore.fma.v4f32`（`<4 x float>`）+ `mncore-lower-typed-intrinsics`
pass（在 `addIRPasses` 中、isel 之前运行）：

* **PE 槽位分配**（朴素版）：2 长语宽度、地址 4 字对齐、**在
  GRF0/GRF1/LM0/LM1 之间轮转**（每个空间一个地址字段，同一表达式的两个
  同空间不同地址输入会被 assemble3 判为 field conflict）；不复用（无
  live-range 分析）。向量函数参数按顺序取首批槽位。
* 每个 typed call 被改写为 `llvm.mncore.peop` token + 一条 `llvm.mncore.mau`
  （opcode 0 = fvfma），原 call 删除；后续 isel/packetizer/hazard 看到的是
  普通 Layer A 指令。
* 约束：仅单精度 `<4 x float>`；掩码必须为常量 0；非向量/非常量操作数报错。

示例输出（三个参数 + 结果 + 第二次调用）：

```
fvfma $llr0v $lls0v $llm0v $lln0v
nop/2
fvfma $lln0v $lls0v $llm0v $llr4v
```

（`nop/2` 是 LM1 写→读的端口间隔，assemble3 在 hazard checker 开启下
验证通过；gpfn3 3 步仿真通过。）

后续（未做）：其他精度/算子（`<2 x double>`、`<8 x half>`、mul/add/max/min、
matvec/mwrite/mread 的 typed 形式）、跨基本块与循环的 slot 复用与生存期、
掩码/多输出、与 §3.6.4 约束耦合的分配（同一步单区域、单写空间）。


## 13. LM 基地址寄存器 (BAR) — 已接线 (2026-11-03)

* peoperand 以 **bit 49** 标记 BAR（仅当 flat 位 bit 9 为 0，避免与 flat
  地址字段冲突）；kind 2/3 分别表示 LM0/LM1。
* 打印：`$mb` / `$lmb`（LM0），`$nb` / `$lnb`（LM1）；width 0=word、1=longword。
* 打包：`peop(..., flags=256)`；解码为独立空间 `MS_LM0BAR`/`MS_LM1BAR`
  （只写、无地址、无对齐检查）；`peInstructionConflict` 要求 BAR 只能出现在
  目的操作数位置。
* intrinsic：`alu` opcode 4 = `lpassa`（可写 BAR）。
* 验证：`lmbar.mir`（EMIT）与 `intrinsics.ll` 的 `lmbar` 函数；assemble3 与
  gpfn3 均通过。

## 14. Layer B 覆盖扩展与分配器 v2 (2026-11-03)

* **类型/算子**：`<2 x double>`、`<4 x float>`、`<8 x half>` ×
  `fma/mul/add/max/min`（15 个），以及同三类精度的
  `matvec/matwrite/matread`（9 个）。
* **分配器 v2**：2 长语槽位、地址 4 字对齐、在 GRF0/GRF1/LM0/LM1 轮转；
  线性扫描的**生存期释放**（某值最后一次使用后槽位回收复用）；目的空间
  必须不属于本表达式的读空间；同一表达式内同空间不同地址的输入直接报错。
* **Layer A 映射**（新增 opcode id）：`mau` 10=dvfma 11=dvmul 12=dvadd
  13=hvfma 14=hvmul 15=hvadd 16=d-matvec 17=h-matvec 18=d-mwrite
  19=h-mwrite 20=d-mread 21=h-mread；`alu` 10=dmax 11=dmin 12=hmax
  13=hmin。f 精度沿用 0..6。
* 仍限制：掩码必须为 0；跨基本块/循环未做生存期分析；matvec 语义定义为
  `dst = A*x + y`（两个向量输入）。
