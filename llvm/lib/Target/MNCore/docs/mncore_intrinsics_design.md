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
; 可重定位的 MV（kernel ABI §13）。第 5 个参数是打包的 fixup 记录
; （arg/space/unit/width_bits/operand/opcode/addend_units）；被重定位的
; 操作数地址字段写 0（占位），asm printer 在该行末尾追加
; `# @mncore.fixup <site>` 并在文件尾输出 sidecar 注释块。原 `mv` 的
; 4 参数 ABI 不变，无 buffer 参数的 kernel 不受影响。
void @llvm.mncore.mv.reloc(i32 opcode, i64 params, i64 src, i64 dst,
                           i64 fixup)
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

Layer B 已实现 `fma/mul/add/max/min`（`relu/cvt` 尚未实现）。二元算子
签名为 `(vt, vt, zmask, wmask)`，`fma` 为 `(vt, vt, vt, zmask, wmask)`——
不存在“被忽略的第三输入”。例如：

```llvm
%r = call <4 x float> @llvm.mncore.fma.v4f32(
        <4 x float> %a, <4 x float> %b, <4 x float> %c,
        i64 %zmask, i64 %wmask)
```

降级：`mncore-lower-typed-intrinsics` 已实现，为每个 SSA 值分配 PE 槽位
（GRF0/GRF1/LM0/LM1，2 长语，槽位步进 16 word），把读写地址变成 `peop`
token，再发射 `mau`/`alu`。契约见 §15。

## 5. Matrix 语义

矩阵寄存器每 MAB 两面（x/y），物理 16 行；逻辑行数：d=4、f/g=8、h=16。
`matop(face,row,width,flags)` 足以表达：

* `mwrite`：`mau(opcode=mwrite, src=peop, mat=matop)`（§3.6.10）
* `mread`：`mau(opcode=mread, mat=matop, d0=peop)`（§3.6.11）
* `matvec`：`mau(opcode=mfma/mmul, mat=matop(face,row=0,width), sx, sy, d0…)`

Layer B（已实现的实际签名，掩码目前只在 Layer A 可用）：

```llvm
void  @llvm.mncore.matwrite.v4f32(i64 %face, i64 %row, <4 x float> %src)
<4 x float> @llvm.mncore.matread.v4f32(i64 %face, i64 %row)
; dst = A*x + y
<4 x float> @llvm.mncore.matvec.v4f32(i64 %face, <4 x float> %x, <4 x float> %y)
```

masked matvec/mread 已以 Layer A `mau` id 4/6 提供；masked mwrite 因唯一
目的操作数是矩阵寄存器而不可表达。

> **注意（2026-11-04）**：上面这三条 typed 签名只保证“能发指令”，不保证与
> SDM 的操作数宽度/值宽度一致（y/dst 在 SDM 中是每 PE 8 f32、longword 访问，
> x 是每 PE 4 f32、word 访问）。矩阵指令至今 **没有数值 golden**，任何据此
> 构建的 GEMM 会在错误处静默出错。几何、差异表与实测探针见 §22；`tt.dot`
> 因此在 `tritongpu-to-mncore` 中被门禁拒绝。

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
| `tt.dot` | `mau(mwrite)` + `mau(matvec mfma)` + `mau(mread)`（**未实现，门禁拒绝**；见 §22） |
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
2. **M2（部分完成）**：`mau` 向量与矩阵（mwrite/mread/matvec）已接通；
   多输出与 OMR 在 intrinsic 层仍被拒绝（Layer A MI 已有多输出/OMR def）。
3. **M3（已部分完成）**：Layer B 语义 intrinsic 与 PE 槽位分配器已实现
   （单块线性生存期、空间轮转、16-word 区域）；Triton `ttgir → mncore`
   原型尚未开始。
4. **M4**：DRAM/PDM 描述符、`mv` 参数与 tag 分配、静态内存规划、
   double buffering（PIKG 的 head/main/tail 契约作为设计参考）。
5. **M5**：端到端：`llc` → assemble3(strict) → gpfn3，数值对拍（`--dump` +
   `check_dump.py`）。

## 9. 验证准则

* 每个 intrinsic 至少一个 `FileCheck` 文本测试；合法/非法各一。
* 非法用例必须由**后端**拒绝（token 空间/宽度/对齐、wait tag、imm 与 LM0 冲突），
  不得依赖 assemble3 报错。
* 远端回归分为两阶段：语法（checker 关闭）与 `HAZARD_OK` 上的
  hazard/race checker（开启）。`--strict` 仍可给所有 EMIT 测试加 `--hazard`。
* 数值断言尚未接入：`emulate_gpfn3.sh --dump` 对当前内核产生**空转储**
  （后端未发射 `d get`），实测 `/tmp/mncore_layerb.ll.dmp` 为 0 字节；
  `check_dump.py` 仍只有合成 self-test。

## 10. 尚待定案

* ~~Layer A 的 `opcode` 枚举~~ **已定案**：稳定数值由
  `MNCoreIntrinsicOps.h` 导出（append-only，不复用 MI opcode 编号）；isel 与
  Layer B pass 共用。
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

示例输出（槽位步进 16 后的当前形态；`masked_fma` 另见测试文件）：

```
fvfma $llr0v $lls0v $llm0v $llr16v
nop
fvfma $llr16v $lls0v $llm0v $lls16v
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
* **分配器 v2**：2 长语槽位、槽位步进 16 word（一个 2 长语区域）、在
  GRF0/GRF1/LM0/LM1 轮转；
  线性扫描的**生存期释放**（某值最后一次使用后槽位回收复用）；目的空间
  必须不属于本表达式的读空间；同一表达式内同空间不同地址的输入直接报错。
* **Layer A 映射**（新增 opcode id）：`mau` 10=dvfma 11=dvmul 12=dvadd
  13=hvfma 14=hvmul 15=hvadd 16=d-matvec 17=h-matvec 18=d-mwrite
  19=h-mwrite 20=d-mread 21=h-mread；`alu` 10=dmax 11=dmin 12=hmax
  13=hmin。f 精度沿用 0..6。

### 14.1a L1B collect 的 MAB 语义 (2026-11-04，已实测更正)

SDM §3.6.8.21 的效果式给出精确语义：

```
for cycle = 0:4
  forall mab, pe:
    dst_mab = (mab + mabdiff) % 16
    data[dst_mab*4 + pe] = PE[mab][pe]
  L1BM[addr_b + cycle*64 : +64] = data[0:64]
```

* **一次 `l1bmd` collect 写入全部 64 lane**（每 cycle 64 长语 × 4 cycle）。
  `mabdiff` 是 **MAB 编号旋转（置换）**，不是掩码：`+k` 时
  `dest[j] = lane[(j - 4k) mod 64]`，`-k` 时 `dest[j] = lane[(j + 4k) mod 64]`
  （j 的低 2 位 = PE 号保持不变）。gpfn3 实测 `+1 → lane61`、`-1 → lane5`
  与上式一致。
* 之前"每次 collect 只取 4 lane、需 16 次旋转 collect"的结论**错误**，
  已由 §3.6.8.21 原文和直接 gpfn3 探针推翻；`mabdiff` 只用于把某个 MAB 的
  数据搬到另一个 MAB 槽位，不承担覆盖率职责。
* 唯一的硬件约束：collect 的 `$lb` 目标地址必须 **64 长语对齐**（assemble3
  报 `L1BM address for combining operation must be aligned to the multiple
  of 64`）。
* 因此 Distributed 写回只需**每平面一次 collect**。E2E 的真实故障不在
  collect，而在其后的 L1BM→L2BM：`l2bmd`（§3.6.7.8）按
  `data[8*l1b : 8*l1b+8] = L1BM[l1b][…]` 从 **8 个 L1B 依次各取 8 长语**
  拼接。Triton 的 distributed tile 在各 L1B 上是**复制**的，于是 collect
  把 8 份相同的 8 长语块交织进 L2BM，平面被打散。写回改用
  `l2bm@<l1badr>`（§3.6.7.5，单 L1B 点对点搬运 16 长语/cycle）取 L1B 0
  的连续 64 长语即可，DRAM 与 CPU golden 逐元素一致
  （`tests/run_e2e_add.py` 通过）。`l2bmd` 只在各 L1B 持有互不相同的数据时
  才是正确原语。


### 14.1 分配器 v2.1 (2026-11-04)

* **目的空间避开存活值**：分配新结果时，先统计当前调用点仍存活的 SSA 值所占
  的 PE 空间（kernel 参数在最后一次使用前视为存活），目的空间优先选择不在
  该集合中的空间；四个空间全满时才退回旧的轮转策略，并由同空间输入检查
  报错。这修复了 `vec.load` 结果与参数落在同一空间、随后被同一表达式读取
  的冲突。
* **typed load 的强制隔步**：`l1load` 发出的 `l1b` 之后自动补一条
  `llvm.mncore.fence`（一个 NOP 步）。PE 存储允许同一步读写（3.6.3.9），
  但读观察到的是旧值，因此 load→consumer 必须隔步；hazard pass 再补足
  LM 端口所需的其余 nop。
  **已改为调度驱动 (2026-11-04)**：Layer B pass 不再自动插入 `fence`。
  packetizer 的 `l1bPEStageRace` 规则禁止同一步内 L1B 传输与 PE 空间的
  producer/consumer 配对，由 hazard pass 在步间补 `nop/2`。
  `load-gap.ll` 验证独立 ALU 可与 load 同一步发射，仅 consumer 前隔步
  （`l1bmp ...; fvadd ...` → `nop/2` → `fvfma`）。
* **新增 typed 内存 op**：`llvm.mncore.l1load.<ty>(mode, lbop, zmask, wmask)`
  （L1BM→PE slot；mode 0/1/2 = l1bmp/l1bmm/l1bmm4）与
  `llvm.mncore.l1store.<ty>(value, mode, lbop, set)`（PE→L1BM collect；
  mode 1/2 = l1bmm@/l1bmm4@）。对应 MLIR `mncore.vec.load/store`。
* **`mncore.dma`**（MLIR 层）：`DRAM/PDM → L2BM → L1BM`，lower 为
  `mvop + mvparams + mv + lbop + l2b`；`mvp` 的方向由操作数 kind 决定。
* **2 长语操作数的 longword 布局（2026-11-04 数值修正）**：两个 longword
  位于槽位 `W/2` 与 `W/2+4`（不是 +2）。分布式 load/store 的平面偏移必须
  使用 +4；`golden_roundtrip.mir` 在 gpfn3 上以
  `load(1.0/2.0) → fvadd → store` 验证低平面 2.0、高平面 4.0。
* 仍限制：掩码必须为 0；跨基本块/循环未做生存期分析；matvec 语义定义为
  `dst = A*x + y`（两个向量输入）。

## 15. Layer B 契约 (2026-11-03, 修订)

**SSA 值 ↔ 物理槽位**
* 每个 typed 结果独占一个 2 长语槽位；槽位区域为 **16 word**（4 周期 ×
  每周期 4 word），因此槽位地址步进 16，杜绝区域重叠（重叠曾被 assemble3
  判为 data race）。
* 槽位空间轮转 GRF0/GRF1/LM0/LM1（每个空间一个地址字段；同一表达式的两个
  同空间不同地址输入 = field conflict）。
* 单基本块函数按线性 last-use 释放并复用；多基本块函数**保守不复用**。
* **使用约束**：typed 结果只能被其他 typed intrinsic 使用；返回值、store、
  普通算术等使用会报错（否则删除 call 会丢失数据流）。

**向量参数 ABI（host/launcher 必须遵守）**
* 第 i 个向量参数预装在 space = 1 + (i mod 4)、地址 = 16 × (i div 4) 的
  2 长语槽位（单位 word）；后端对这些参数返回 undef，因为其使用已被 token
  取代。
* 若要改变布局，必须同时修改 launcher 与 `SlotOf` 的参数分配循环。

**算子签名**
* 二元向量算子（mul/add/max/min）签名为 `(vt, vt, zmask, wmask)`；
  fma 为 `(vt, vt, vt, zmask, wmask)`。不存在“被忽略的第三输入”。
* 掩码必须是常量或 `maskop` token；其他值报错。常量 0 = 无掩码。

**数值验证的边界**
* 现有回归证明：汇编合法（assemble3，含 hazard/race checker）、仿真器正常
  结束（gpfn3）。尚未建立 golden-value 对拍：`check_dump.py` 仅有合成
  self-test，且后端尚未发射 `d get` 语句，`-d` 转储的可观测内容有限。
  因此**不能**据现有回归推断向量/矩阵运算数值正确。


## 16. 仍存边界处理 (2026-11-03)

* **区域重叠的 race 判定**：`mncore::regionsOverlap` / `accessRegionWords`
  （word=4、longword=8、2-longword=16 word）。hazard 的 GRF/T 数据竞争与
  L1B/L2B 同址规则改为比较区域，而不是基址；`region-overlap.mir` 验证
  “2 长语 @0 写 vs longword @4 读”会插入 nop，“@16 读”不会。
* **masked 矩阵输出**：新增 `XFMFMA_M`/`YFMFMA_M`（matvec）与
  `W_XFMREAD_M`/`W_YFMREAD_M`（转置读），Layer A `mau` id 4/6 在带掩码时
  选择；`fmfma …/0111p`、`fmread …/0111p` 通过 assemble3（语法）。
  **masked mwrite 仍不可表达**（唯一目的操作数是矩阵寄存器，assemble3
  拒绝），属 ISA 限制而非后端缺口。
* 现有单元测试同时覆盖 `p` 后缀（2 长语目的 + 长语掩码）。


## 17. 现状核查与缺口 (2026-11-03)

**已由代码/回归证实的部分**
* Layer A：`fence/wait/alu/l1b/l2b/mau/mv/maskdir` + 6 个 token 构造器；
  对已接通的 opcode id 子集，IR → `.vsm` → assemble3（含 hazard/race
  checker，`layerb`/`layerb2`）→ gpfn3 全链路通过。
* Layer B：3 精度 × 5 算子 + matvec/matwrite/matread；SSA→槽位→Layer A
  改写、非 typed 使用拒绝、掩码转发、单块生存期。
* 规则：§3.6.4 区域比较、区域重叠 race、BAR、T-indirect、`nop/N` 压缩。

**缺口（按优先级）**
1. **数值 golden 验证**：`--dump` 为空（无 `d get`），`check_dump.py` 未接入
   主回归；目前只能声明“可汇编 + 可仿真”，不能声明数值正确。
2. **Layer A opcode 表是手写子集**，非稳定导出枚举；多输出/OMR 在 intrinsic
   层仍被拒绝；`l2b` 仅 l2bmb，`mv` 仅 mvp。
3. **Layer B 跨块/循环**：多块保守不复用；无 PHI/循环携带值；typed 结果不得
   逃逸；向量参数 ABI 仅文档化，无 launcher 实现或校验。
4. **typed 矩阵**：无掩码参数（仅 Layer A）；matvec/mwrite/mread 选择 X 面
   def，即使 token 的 face 位为 Y（资源模型与文本 face 可能不一致）。
5. **`fence` 降为一条 `nop` 步**（不是真正的零开销屏障）；孤立 `wait`、
   tag=0 未由后端拒绝。
6. **packetizer 同址规则**仍按 base+width 比较（等价于同区域），不做区域
   重叠；写-写同空间一律冲突（与字段模型一致）。
7. **Masked 矩阵**：matvec/mread 已提供；mwrite 不可表达（ISA）。
8. Triton 映射表是目标形态，尚无 dialect/launcher/描述符实现。


## 18. 缺口 1/4/5 闭合 (2026-11-03)

* **1 — 数值 golden 验证（初版）**：新增非 PE 伪指令 `DGET`
  （`d get <mem> <nwords>`，操作数打包 `[3:0]mem [20:4]addr [28:21]nwords`）
  与 `golden.mir`：`imm i"42" $llr0v` + `d get $lr0 1`。实测 assemble3 接受
  `d get`，gpfn3 `-d` 产出真实转储，`check_dump.py` 断言
  `GREG0:n0c0b0m0p0:0=0x2A0000002A` 通过；回归新增独立 “golden numeric”
  阶段。仍限：仅 GRF d-get，无 `d set`/矩阵/向量 golden，覆盖 `v:` 字段。
* **4 — 矩阵 face**：Layer A `mau` id 4/5/6/16..21 现在按 mat token 的
  face 位选择 X/Y def（含 masked Y：`YFMFMA_M`、`W_YFMREAD_M`）；
  `intrinsics.ll` 的 `face_y` 输出 `fmfma $ly …`、`fmread $ly3 …`。
* **5 — 控制语义**：`wait` tag=0 在发射期拒绝；孤立 `wait` 步在 packetizer
  后检查并报错（`lone-wait.mir` 负例）。`fence` 仍降为一条 `nop` 步
  （保守屏障，未变）。

**仍开**：2（稳定导出的 opcode ID 枚举）、3（跨块/循环生存期与 launcher
级向量参数 ABI 校验）。


## 19. 稳定 opcode 枚举与 MLIR 准备 (2026-11-03)

* `MNCoreIntrinsicOps.h` 定义 `ALUOp/MAUOp/L1BOp/L2BOp/MVOp` 的稳定数值；
  SelectionDAG isel 与 Layer B pass 已改用枚举名（数值不变）。规则：只追加，
  不重编号；未列出的 ID 由选择器拒绝。
* **MLIR 已落地**（2026-11-03）：`MNCoreIntrinsicOps.h` 迁至公共 include
  (`llvm/include/llvm/Target/MNCore/`)；树内 `mncore` dialect 与
  `convert-mncore-to-llvm` 位于 `mlir/include|lib/{Dialect,Conversion}/MNCore*`。
  验证器规则、op 清单与状态见
  `llvm/lib/Target/MNCore/docs/mncore_mlir_dialect.md`；前端接入见
  `mncore_frontend_integration.md`。`MNCoreOps.cpp` 对 33 个 opcode 值做
  `static_assert`，与 `MNCoreIntrinsicOps.h` 编译期同步。
* 构建：`build-mlir`（`-DLLVM_ENABLE_PROJECTS=mlir`），可执行
  `mlir-opt --convert-mncore-to-llvm | mlir-translate --mlir-to-llvmir |
  llc -mtriple=mncore2` 全链路（已用 `fvfma`/`fmwrite` 内核验证）。


## 20. 门禁状态 (2026-11-03)

**Correctness Gate (P0) — 达成**
* opcode：max/min 反转修复；稳定枚举 `MNCoreIntrinsicOps.h`（append-only）。
* slot reuse：16-word 区域步进、last-use 去重释放、单块复用/多块保守；
  契约拒绝非 typed 使用与非法掩码。
* 验证脚本：`verify_assemble3.sh` `pipefail` + 分段退出码 + `mktemp`；
  负向测试 `operand-placement` / `pe-alignment` / `lone-wait` 均真实失败并被
  断言；远端失败计入总数并返回非零。

**Layer A Signoff (P0) — 部分达成**
* 合法性 + hazard/race：全部 9 个 EMIT MIR 测试与 `layerb`/`layerb2` 在
  assemble3 checker 开启下通过（`peoperand-modes` 需 `--flat`），gpfn3 正常
  结束。
* 数值 golden：`golden.mir` 实测 `imm 42` → `0x2A0000002A`（GRF0）与
  `50+7=57` → `0x3900000039`（GRF1，ALU `uladd`），由真实 dump +
  `check_dump.py` 断言。
* **未覆盖**：MAU / MV / L1B / L2B 的数值 golden（缺少 `d set` 输入与可观测的
  搬运数据）；这些族目前只有合法性 + hazard + 仿真退出码证据。

**Layer B v3 (P1) — 未达成**
* 已有：SSA 使用约束（非 typed 使用拒绝）、单块线性生存期与复用、跨块保守
  不复用、空间/区域约束、掩码转发。
* 缺：真正的 SSA-preserving lowering（值可逃逸/PHI/跨块复用）、可靠的
  跨块生存期、与 §3.6.4 耦合的槽位求解（当前是启发式）、launcher 级向量参数
  ABI 校验。


## 21. d set 搬运 testbench 与 v3 评估 (2026-11-03)

**已实现**
* `DSET` 伪指令：`d set <mem> <nwords> <16-hex>`（复用 d get 的 spec 打包）；
  `golden_mem.mir`：`d set $lc0 1 …39` → `l2bmb $lc0 $lb0` → `nop/2` →
  `l1bmp $lb0 $lr0v` → `d get $lr0 1`，实测 dump `v:0x39`，由
  `check_dump.py` 断言（L2BM→L1BM→PE 数值搬运闭环）。
* **同一步 L1B/L2B 写读互斥**：packetizer 新增 `sameStageRace`——L1B/L2B 的
  producer 写与 consumer 读在同一表达式组中非法（3.6.3.3/.7），因为 hazard
  pass 只能在步之间插 nop。**唯一豁免**是 `$lbi` 折返对（写 `$lbi` + 读
  `$lbi`，3.6.8.2），由 `Turn` 标记识别。该规则由 `golden_mem` 与
  `l1bml2bm-emit`（折返 bundle）双向验证。

**Layer B v3 的阻塞判断**
* 后端 `Select` 明确拒绝 `BR/BRCOND/BR_CC/BR_JT/BRIND`（无控制流 lowering）。
  因此**多基本块 IR 根本无法进入 typed lowering 的可测试路径**：跨块生存期、
  PHI、跨块槽位复用在当前架构下不可验证，属“架构 A（控制流转换阶段）”的
  前置条件，而不是分配器本身能单独闭合的项。
* 已强制：typed 结果只能被 typed op 消费（PHI/返回/store 等即报错）；
  单块线性生存期复用；跨块保守不复用；向量参数 ABI 契约文档化。
* §3.6.4 耦合求解：当前为启发式（每表达式目的空间选择 + 同空间不同地址
  拒绝 + 16-word 区域），完整图着色求解仍待控制流阶段之后处理。

**结论**：在无控制流 lowering 之前，Layer B v3 只能以“单块 + 契约限制”
形态存在；本轮不实现不可验证的跨块生存期代码。

## 22. MAU 矩阵几何与 GEMM 契约 (2026-11-04)

本节记录 SDM §3.6.9–§3.6.11 的矩阵事实（用于 `tt.dot`/GEMM 路径），并
显式区分 **手册直述**、**由伪码推出的推论** 与 **尚未实测的开放项**。凡标注
“推论”的条目在 gpfn3 数值 golden 之前不得当作签核依据。

### 22.1 矩阵寄存器（手册直述，SDM pp.65, 111–116, 123–128）

* 每个 MAB 一个 MAU，矩阵寄存器 **按 MAB** 共享，不按 PE 私有；两面
  `$lx`/`$ly`，各面 **物理 16 行**。
* 逻辑行数：d=4、f=8、g=8、h=16；物理行映射：double 用 0,4,8,12；single /
  pseudo-single 用 0,2,…,14；half 用全部 16。
* 逻辑列数：d 4×4、f/g 8×8、h 16×16。
* **一行 = 每 PE 一个长语**（d 1×f64 / f 2×f32 / h 4×f16），分布在 MAB 的
  4 个 PE 上。f32 显式映射：`fmwrite` 的 `refer_matreg[(addr+cycle)%8][j] =
  src_data[j/2][j%2]`，即逻辑列 j ↔ PE j/2 的 32-bit word j%2。
* f32 面向量 8 列里 **只有偶数逻辑列 0,2,4,6 被 matvec 使用**
  （“単精度MAU演算ではMSB側単語を利用する”, p.136），故 matvec 的有效矩阵是
  8 行 × 4 列。

### 22.2 `fmfma` 语义（手册直述 + 伪码，p.114）

单精度一条指令覆盖 **4 个 cycle**，同一面 A 不变，x/y/dst 按 cycle 取址：

```
for cycle = 0:4
  A[8][8]      = refer_matreg(side, SingleWord)      ; 只用列 0,2,4,6
  x[4]         = refer_pemem(src_x, cycle)           ; 每 PE 1 word
  y[4][2]      = refer_pemem(src_y, cycle)           ; 每 PE 2 word
  dst[i] += Σ(j=0..3) A[i][2j]*x[j] + y[i], i=0..7
  refer_pemem(dst, cycle) = dst[pe*2:(pe+1)*2]
```

数学形式即 `dst = A·x + y`，行索引是输出索引（不是 `xᵀ·A`）。一条
`fmfma` 的 4 个 cycle 各取一个 x，因此 **一条指令可以消费 B 的 4 列**，
构成 8×4（M×K）乘 4×4（K×N）块的累积——这是由伪码推出的组成方式，
手册并未把它描述成矩阵乘。

其它精度：d `dmfma(u|d)` 4×4、K=4，u|d 选 4 行中的 2 行（另 2 行得
0+y）；g `gmfma` 8×8、K=8（伪单精度，尾数 18 bit）；h `hmfma` 16×16、
K=16，单精度累积。

**ISA 没有矩阵×矩阵指令**：§3.6.9.1 只列 `mfma`（3 输入）与 `mmul`
（y=0），每条只带一个矩阵面。

### 22.3 `mwrite` / `mread`（pp.123–128）

* `(f|g)mwrite <src> $l(x|y)<addr>`：一行/cycle，4 行/指令，行号
  `(addr+cycle) mod 8`；两指令写满 8×8。src 的 word/longword 访问决定
  每行取 2 还是 4 f32/PE。
* `hmwrite` 的 `ll` 形式 2 行/cycle（addr 必须偶），8 行/指令。
* `mread` 的地址是 **起始列号**，且是 **转置读**：PE 号 ↔ 行号，所以
  “写入再读出 = 转置”。无“非转置读”指令。`mread` 纯按位复制，不做浮点
  规范化（也可用于整数矩阵转置）。
* 列为性能事实：`fmwrite` 一条指令 4 行，`fmread` 一条指令 4 列。

### 22.4 与当前 Layer B 的已知差异（**`fmwrite`/`fmread` 已实测闭合，见 §22.7；`fmfma` 仍未测**）

> **2026-11-04 更新**：本节表格中 `fmwrite`/`fmread` 两行的期望 width（=1，
> longword）已由 gpfn3 数值探针**实测确认**，`width=0`/`width=2` 不复现
> SDM 的列/行落点；Layer B 对这两条指令统一发 `width=2` 是**确定错误**。
> `fmfma` 的三行（x / y / dst）**仍未实测**，故 §22.5 的门禁继续有效。
> 2026-11-04 追加：G2 探针已跑，但 gpfn3 的 `fmfma` 不做乘加（`dst := y`），
> 详见 §22.9。门禁因此不能停在“待测”，而是“实测到参考模型缺项”。
> **2026-11-04 再追加**：下表四条已由厂商 SDK 头文件
> `mncl/device/mncore2.h` 的 intrinsic 签名独立确认（Level 2，见 §22.10）：
> `fmwrite`/`fmread` 每 PE 4 长语（width=1），`fmfma` 的 x 为 `f32x4`
> （2 长语）、y/dst 为 `f32x4x2`（4 长语），且 **y 与 dst 是普通 IEEE f32**。
> 结论、证据与模型见 §22.7。
> **2026-11-04 已实施**：`MNCoreLowerTypedIntrinsics` 改为按 (opcode, role)
> 发访问宽度——`fmfma` x = 0、y/dst = 1，`fmwrite`/`fmread` = 1，向量形式保持
> 2（该路径已在 gpfn3 逐元素验证，不得连带改动）。`layerb2.ll` 现用正则钉住
> 前缀：矩阵行出现 `$ll`（2 长语）即判回归。
> **仍未闭合的形状缺口**：typed 签名只有 4 f32（`<4 x float>` = 2 长语），而
> `fmwrite`/`fmread`/`fmfma.y`/`fmfma.dst` 的活动硬件操作数是 `f32x4x2` = 8 f32
> （4 长语）。宽度改对之后这四条仍只提供一半的值，因此**不能**据此宣称矩阵
> 路径可用，`tl.dot` 门禁不变；类型加宽与 ABI 槽位的冲突见 §22.11.3。

`MNCoreLowerTypedIntrinsics` 历史上对 **所有** PE 操作数统一发 `width=2`
（2 长语，4-cycle 区域 16 word）；2026-11-04 起改为按 (opcode, role) 发宽度
（见上方更新）。矩阵三指令按 SDM 的每 cycle 访问量应为：

| 指令 | 操作数 | 每 cycle/PE | 4-cycle 区域 | 期望 width | 每 PE 值数 |
|---|---|---|---|---|---|
| `fmfma` | src_x | 1 word | 4 word | 0 (word) | 4 f32 |
| `fmfma` | src_y / dst | 2 word | 8 word | 1 (longword) | 8 f32 |
| `fmwrite` | src | 2 word | 8 word | 1 (longword) | 8 f32（4 行） |
| `fmread` | dst | 2 word | 8 word | 1 (longword) | 8 f32（4 列） |

而 Layer B 的 typed 签名（`matvec.v4f32(face, <4 x f32>, <4 x f32>)`、
`matwrite`/`matread` 的 `<4 x f32>`）只给 4 f32。访问宽度已按角色修正，
但**值宽度仍比 SDM 窄一档**（4 f32 vs 8 f32）。

该差异目前只是“由伪码 + `accessRegionWords` 推出的不一致”，不是已测缺陷：
向量路径（`vec.add` 等，同样是 width=2）已在 gpfn3 上元素级验证过，说明
width=2 的 16-word 区域模型对 **向量** 指令是对的；矩阵指令是否共享同一
约定，必须用 golden 实测，不能靠推断定案。**在实测闭合前，任何 GEMM
lowering 都不得建立在这三条指令之上。**

最小实测探针（待做）：`d set` 一段已知数据 → `l2bmb`/`l1bmp` 取出 →
`fmwrite` 写入 A → `fmfma`（x/y 从另外的 PE 槽）→ `fmread` → `d get`，
与 CPU 端 `dst = A·x + y` 逐元素比对；分别用 width=0/1/2 三种打包各跑一次，
以确定每种操作数的合法宽度与数据落点。该探针同时能定 capacity
（16 行 × 4 PE × 1 longword = 64 longword/面）与 `(addr+cycle)` 回绕。

### 22.5 `tt.dot` 契约（已实现为 **fail-closed 门禁**）

`third_party/mncore/lib/Conversion/TritonGPUToMNCore.cpp` 的
`rejectDot` 在进入线性遍历前拒绝任何 `tt.dot`，并按失败子句给出诊断：

1. 精度：只定义 f32（`fmfma`）。h/f64 需 `hmfma`/`dmfma`，未接线；
   TF32/FP8 输入模式在 MAU 不存在。前端 `MncoreOptions` 已声明
   `allowed_dot_input_precisions=("ieee",)`。
2. 分块：矩阵面 8 行（M），一条 `fmfma` 覆盖 4 列（N）与 K=4，故
   M%8、N%4、K%4 必须为 0。
3. 布局：真实 `tl.dot` 到达的 `#ttg.dot_op<parent=#blocked>` 由
   `fmaDotToLinearLayout` 生成，**把整个 K 维复制进每线程寄存器**
   （32×32×32 时每线程 32 个 K 元素），而 MAU 需要 K 以 **每 MAB 每 PE
   一个元素** 的形式进入矩阵面。两者不兼容，必须先给 mncore 专属 dot
   布局，否则降级只会算错而非算慢。

**状态**：`tl.dot` 前端可用（TTIR→TTGIR 通过，`#blocked` 给每线程 4×4
C tile），mncore 侧为门禁 + 契约，尚无 lowering。2026-11-04：G2 探针实测
gpfn3 的 `fmfma` 不做乘加（§22.9），故门禁**不放行**，理由由“未测”升级为
“参考模型缺项”。形状层面的独立证据见 §22.11：ABI v1 的 2-LW 向量槽只够
x / `fbfn`，y 与 dst 需 4 LW。回归：
`third_party/mncore/tests/run_t9.py`（前端 + 三条拒绝子句）。

### 22.6 未决项

1. ~~22.4 的宽度差异（实测探针未做）~~ → **`fmwrite`/`fmread` 已实测闭合
   (§22.7)**；`fmfma` 的 x / y / dst 三行仍未测，属 G2 —— **G2 已跑但被
   gpfn3 的模型缺项挡住（`dst := y`，§22.9）**，宽度无法由数值结果反证。
   宽度的**非模拟器证据**见 §22.11.2（`$l`/`$ll` 增量 1/2/4）与 §22.11.3
   （ABI v1 向量槽 2 LW 与厂商签名的对齐与冲突）。
2. `v<adri>` 默认步长与 `fmfma` 4-cycle 地址推进的交互（手册未把两者
   写在一起）。§22.7 只测了 `v`（单 access unit）形式下的步长 = 1 longword。
3. 矩阵读入 matvec 前必须 **块浮点 (BF) 转换**，且单精度 BF 块为 4 word
   （每 PE 1 word）共用一个指数；不一致块为非法 BF，行为未定义。写 A 的
   路径必须先做 BF 量化——这决定 `tt.dot` 的数值契约不是 IEEE f32。
   ~~**`fbfn` 目前在 LLVM 后端与 MLIR dialect 中都不存在**~~ → **已新增并
   实测闭合 (§22.8)**：`ALUOp::FBFN = 20` / `MNCore_ALUOp_FBFN`，
   isel 映射到 `BFN_f` / `BFN_f_M`（单输入单目的）。`dbfn` / `gbfn` 仍未接线。
4. 单精度乘积省略两输入尾数低 5 bit（p.148），求和为精确整数加、最后
   一次舍入；即 `fmfma` **不是** bit-exact IEEE FMA。任何数值 golden 必须
   按该规则建模，不能用 `-ffast-math` 或双精度反推。

### 22.7 **G0 实测：矩阵搬运的物理落点（2026-11-04，gpfn3）**

**证据等级**：模拟器实测（Level 3/4 之间：gpfn3 是模型，不是硅片）。本节
所有条目都是 dump 直接读出，不是从伪码推断；但**不得**当作对真实芯片的
断言——凡"硬件为什么这样"的解释都属未知，只有"gpfn3 这样"是可用的契约。

#### 22.7.1 先钉住的读出口径（否则一切 dump 都不可解）

* PE 寄存器文件里 **longword k 占 word (2k, 2k+1)**，dump 的 `<idx>` 是
  **word 地址**（= 2k），`v:` 是该 longword 的 64 位值；`i:` 是它拆成 4 个
  16-bit 的视图（不是地址）。
* `d set <mem> <addr> ...` 的 `addr` 是 **word 地址**，且一次写整个
  longword、**向偶地址对齐**：对 `$lr1` 的写落在 longword 0。
  （实测：依次写 `$lr0..$lr7`，只剩 `$lr1,$lr3,$lr5,$lr7` 的图案，
  分别在 label 0,2,4,6。）
* 因此"写 4 片、每片 64 PE"的填充手法是：`L1B[64k+p] = pattern`，再
  `l1bmd $lb<64k> $lr<2k>v` → 第 k 个 longword。

#### 22.7.2 实测映射（f32，两条指令都用 **longword / width=1**）

```
fmwrite <src> $l(x|y)<row>
  cycle c (0..3)：取 PE q 的 longword，地址 = src_word_addr + 2c（步长 = 1 长语）
                  写入矩阵行 (row + c) mod 8
                  该 longword 的 LOW 半 → 列 2q，HIGH 半 → 列 2q+1
fmread  $l(x|y)<col> <dst>
  cycle c (0..3)：读矩阵列 (col + c)
                  PE q 收到 行 2q → **HIGH 半**，行 2q+1 → LOW 半
                  写入 longword (dst_word_addr/2 + (c XOR 1))
```

三条自洽性证据（缺一都不能定案）：

1. **写侧**：源 slot k 的值出现在矩阵行 k（`$lx6` 起写时，slots 0..3 落到
   行 6,7,0,1，读回时 PE0 拿到 0x22/0x23（slots 2,3），PE3 拿到 0x20/0x21
   （slots 0,1））。
2. **列侧**：`$lx0` / `$lx4` 起读分别取列 0..3 / 4..7，列 4..7 的值来自
   MAB 自己的 PE2/PE3（0x12/0x13 与 0x20+k）。
3. **`c XOR 1`**：读起始列 0 时目的 longword 的**奇偶列交换**（lw16←列1、
   lw17←列0、lw18←列3、lw19←列2）；把起始列改成 1 时整体前移一列
   （lw16←列0、lw17←列1 …）。任何"纯步长"模型都无法同时解释这两组，
   故 XOR 不是拟合出来的自由度，而是被两组独立数据夹出来的。
   **其物理原因未知**，作为 gpfn3 契约记录；真机若不同，改这一个函数即可。

另外实测：

* **矩阵按 MAB 私有**：每个 MAB 的矩阵只由它自己那 4 个 PE 喂入。用
  `pattern(k,p) = 0x10+p` 时，MAB m 的列 0 读回 `0x10+4m`，与 MAB 0 不同。
* **两个 face 几何相同**：`$ly` 与 `$lx` 逐位同构（同一探针在 Y 面复现 X 面
  结果），两面互不干扰。
* **两条 mwrite 可拼满 8 行**：行 0..3 与 4..7 分两次写入互不破坏（
  `golden_mau_g0.mir` 的 8×8 全检）。
* **`width=0` / `width=2` 不复现该映射**：
  - `width=0`（word）：一条指令只覆盖 4 个 word / PE，写侧把同一行的
    奇偶列拆到不同行，读侧只写回 4 个 word 且不出现在列应有位置；
  - `width=2`（2 长语）：一个 cycle 吃两个 slot（两行），读侧只落在 PE0、
    PE2 的四个非连续 longword。
  二者都不是 SDM 描述的 f32 矩阵语义，**不得**用于 `mwrite`/`mread`。

#### 22.7.3 对 Layer B 的结论

`Tok`/`TokSlot` 的无条件 `width=2` 对 `vec.*` 是对的（已在 gpfn3 上做过
元素级验证），但对 `fmwrite`/`fmread` **是错的**。这些指令需要按
(opcode, operand role) 决定宽度：

| 指令 | operand | 实测宽度 | 每 PE 值数 |
|---|---|---|---|
| `fmwrite` | src  | 1 (longword) | 4 row × 2 f32 |
| `fmread`  | dst  | 1 (longword) | 4 col × 2 f32 |
| `fmfma`   | src_x / src_y / dst | **未测**（SDM 提示 0 / 1 / 1） | — |

#### 22.7.4 回归夹具

* `golden_mau_g0.mir` — 8×8 全矩阵（行 0..3 用图案 A、行 4..7 用图案 B），
  两次写 + 两次读，逐位比对；
* `golden_mau_g0_y.mir` — Y 面同构性；
* `golden_mau_smoke.mir` — MAU 是否可执行 / `assemble3` 是否接受的最小冒烟；
* `scripts/check_mau_g0.py` — 把 §22.7.2 的模型写成代码，对 64 个 PE 的
  每个 longword 逐格比对（含"窗口内其余位置必须为 0"与"dump 必须完整"）；
* `scripts/run_regression.sh` 的 `MAU` 阶段同时跑 FileCheck 与该 checker。
  `emulate_gpfn3.sh --dump golden_mau_g0.mir` 后：

```sh
python3 -I scripts/check_mau_g0.py /tmp/mncore_golden_mau_g0.dmp   --case full
python3 -I scripts/check_mau_g0.py /tmp/mncore_golden_mau_g0_y.dmp --case y
```

2026-11-04 gpfn3 结果：`OK (case full, 1024 cells over 64 PEs)`、
`OK (case y, 512 cells over 64 PEs)`。

#### 22.7.5 仍未闭合（G1/G2/G3）

* `fmfma` 的 x / y / dst 宽度与 4-cycle 地址推进（G2）——**形状已由
  §22.10 的厂商头文件定案，数值被 §22.9 判为当前 gpfn3 不可测**；
* ~~`fbfn`（块浮点转换）在 LLVM/MLIR 中完全不存在（G1）~~ → **已闭合
  (§22.8)**；
* 跨指令累加（G3）：dst 能否作为下一条 `fmfma` 的 y，以及同 slot 复用时
  的 hazard；
* `v<adri>` 显式步长与 MAU 4-cycle 的交互（本轮只测了 `v` 默认步长）。

### 22.8 **G1 实测：`fbfn` 块浮点转换（2026-11-04，gpfn3）**

**证据等级**：与 §22.7 相同——gpfn3 模拟器实测，不是硅片。

#### 22.8.1 指令形状

| 项 | 实测 |
|---|---|
| cycle 数 | 4 |
| 每 PE 每 cycle 值数 | 1 longword（`width=1` 原生） |
| cycle c 读 | longword `src_word/2 + c` |
| cycle c 写 | longword `dst_word/2 + c` |
| `width=2` | 合法，但只用每个 2 长语访问的 **MSB longword** |
| 块的范围 | 一个 32-bit word × 一个 MAB 的 4 个 PE = 4 word 块 |

关键在于**一个 longword 内的两个 32-bit word 是两个互相独立的块**：
`fbfn` 不是"每 PE 8 个 f32 一起换算"，而是每 PE 每 cycle 各有 2 个独立
4-word 块，块内的 4 个 word 来自同一 MAB 的 4 个 PE。故 `fbfn` 天然与
§22.7.2 的矩阵落点同构：转换后的 word 可直接 `fmwrite` 成矩阵列。

#### 22.8.2 SDM 4.4 的编码（与 IEEE 不同，必须显式建模）

BF 采用**显式前导 1**：

```
value = (-1)^s * 2^(E-127) * sum_{j=1..23} 2^(1-j) * M_j
```

即尾数域按 1.xxx 解释（首项权重 2^0 = 1），**不是** IEEE 的 1.xxx 隐含 1。
因此 1.0 在公共指数 0 下的 BF 位型是 `0x3FC00000`——IEEE 会读成 1.5。
**原始位型永远不是 IEEE 数值**，任何比对必须先按上式解码。

转换算法（`scripts/check_mau_bfn.py` 的 `bf32_block`）：
1. 公共指数 = 块内各元素指数域的最大值；
2. 若存在指数等于该最大值的元素且其尾数全 1，公共指数 +1（RNE 的进位）；
3. 例外：公共指数 ≥ `0xFF` → 全部输出 inf，符号取各自输入；
   某输入指数为 0（零/次正规）→ 输出指数 = 公共指数、尾数 0；
   全块指数为 0 → 全零；
4. 其余元素把 24 bit 有效数（隐含 1 + 尾数）右移 `common - e + 1`，按
   **就近偶数**舍入；下溢保留公共指数、尾数置 0。

#### 22.8.3 实测覆盖

`golden_mau_bfn.mir` 用 4 个 plane × 4 cycle 覆盖：

* 普通下移（不同指数混在一个块里）；
* 进位 bump（全 1 尾数把公共指数顶上去，导致 1.999… 的邻居全被舍成 2.0）；
* `inf` / 零 / 次正规的例外分支；
* RNE 的三种 tie（`1+2^-23`、`1+2^-22`、`1+3*2^-23`）；
* 两个 word lane 各自独立成块；
* 特殊组覆盖全部 4 个 MAB、4 个 cycle、两种访问宽度。

结果：`check_mau_bfn: OK (512 cells x 64 units over 64 PEs)`，逐位一致。

#### 22.8.4 后端接线

* `mncore::ALUOp::FBFN = 20`（`MNCoreIntrinsicOps.h`，新号段）；
* MLIR `MNCore_ALUOp_FBFN`（`MNCoreOps.td`）+ `MNCORE_ENUM_SYNC`；
* isel：`case ALUOp::FBFN → BFN_f / BFN_f_M`，`OneInput = true`；
* 汇编层 `BFN_f` / `BFN_f_M` 原已存在于 `MNCoreInstrALU.td`；
* `dbfn` / `gbfn` 的 def 存在但**未映射**（`ALUOp` 只加了 FBFN）。

#### 22.8.5 回归夹具

* `golden_mau_bfn.mir` — 上述输入图案；
* `scripts/check_mau_bfn.py` — SDM 4.4 模型，逐格比对 64 个 unit；
* `run_regression.sh` 的 `BFN` 阶段（FileCheck + dump 模型检查），
  并列入 `HAZARD_OK`。
* `intrinsics.ll::alu_bfn` — 走 `llvm.mncore.alu(opcode=20)` 的 isel 路径
  （含掩码形）。

#### 22.8.6 仍未闭合

* `dbfn` / `gbfn`（f64 / 伪单精度）未测也未映射；
* `fbfn` 与 `fmfma` 的**组合**未测：SDM 说 matvec 前必须先 BF 量化，
  但"`fbfn` 输出直接作矩阵源"这一路径尚无端到端 golden（属 G2/G3 的
  数值前提）。**2026-11-04 补充（Level 2，§22.10.2）**：厂商
  `__builtin_matrix_loadf32` 的注释明确该序列是 `fbfn` → `fmwrite`，
  且 `fmwrite` 本身不做转换。所以组合的**指令顺序**已有权威依据，
  缺的只是数值 golden——而数值一侧被 §22.9 挡死；
* 非有限值在后续 `fmfma` 中的传播未测。

### 22.9 **G2 实测（未闭合）：gpfn3 的 `fmfma` 只做操作数搬运，不做乘加（2026-11-04）**

**证据等级**：模拟器实测（Level 4，`gpfn3` 是模型不是硅片）。本节结论
**只描述 gpfn3 的行为**，不能读成"芯片的 MAU 不算乘加"。

#### 22.9.1 探针与判据

探针（`/tmp/gen_p6.py`、`/tmp/gen_p7.py`、`/tmp/gen_p8.py`，未入库，仅作本轮
取证）：`d set` 写 L1B → `L1BMD` 进 GRF0 → `PASSA_l` 分出 y 块 →
`W_XFMWRITE` 写 A 面 → 一条 `XFMFMA`/`YFMFMA` → 40…200 个 `NOP` →
`DGET` GRF0/GRF1 全量 dump。

写法统一用**位型**表述，避免 BF/IEEE 混淆（见 §22.10.3）：

* A 面 rows 0..3 的每个长语 = `0x000000003FC00000`，即 A[i][2p] 为 BF 1.0；
* x（`f32x4`，2 长语）= 4 个 `0x3FC00000`，即 [1,1,1,1]（BF）；
* y（`f32x4x2`，4 长语）= 前两长语 `0x40C0000040C00000`，后两长语 0。

`A·x = 1+1+1+1 = 4.0`，y 的前两长语按 **IEEE f32** 解码为 6.0
（`0x40C00000` 在 BF 下读作 4.0，两种口径见 §22.10.3）。若乘加成立，
dst 前两长语应为 **10.0 = `0x41200000`**。

#### 22.9.2 实测结果

| 变化量 | 取值 | dst 实测 |
|---|---|---|
| y 前两长语 | `0x00000000` / `0x3FC00000` / `0x40C00000` | **恒等于 y，逐位** |
| x 空间 | GRF0 / LM0 | 无差别 |
| dst 空间 | GRF0 / GRF1 | 无差别 |
| A 面 | 只写 rows 0..3 / 两个面都写 | 无差别 |
| `fmfma` mat imm | 0 / 4 | 无差别 |
| 矩阵面 | X 面（`XFMFMA`）/ Y 面（`fmwrite ... $ly0` + `fmfma $ly`） | 无差别 |
| 延迟窗口 | 40 / 200 NOP | 无差别 |

判决性的一组（A、x 全为 BF 1.0，y 前两长语 `0x40C00000`）：

```
GREG0 idx=0  v=0x40C0000040C00000     <- dst lw0，逐位等于 y lw0
GREG0 idx=2  v=0x40C0000040C00000     <- dst lw1，逐位等于 y lw1
GREG0 idx=4  v=0x0                    <- dst lw2 = y lw2 = 0
GREG0 idx=6  v=0x0                    <- dst lw3 = y lw3 = 0
```

预期 `10.0 = 0x41200000`，实测逐位等于 **y**。即
**`dst := y`（每 cycle 一个长语的原样拷贝），A·x 项恒为 0**。

#### 22.9.3 gpfn3 确实读并校验了操作数

不是"整条指令被忽略"：

* dst 的 4 个长语被**全部重写**（`y = 0` 时被写成 0，而不是保持原值）——
  写通路是通的；
* 喂原始 IEEE 位型会触发
  `Invalid block float format. The exponent of 4th element is not equal to
  the common exponent ...`，说明 MAU 侧有块浮点解码与校验；
* `fmwrite`/`fmread` 的搬运（G0，§22.7）在同一程序里正确工作；
* 逐位复制说明 y 的通路是精确的，缺的只是乘积项。

#### 22.9.4 对 G2 / G3 与 `tt.dot` 的影响

* **G2 无法在本模拟器上闭合**：用户指令里的"数值 golden"这一环缺少可用的
  参考实现。G3（跨指令 K 累积走 y）同样落在同一条失效通路上。
* §22.5 的 fail-closed 门禁**继续有效**，且理由从"未测"加强为"实测到模型
  缺项"。`tl.dot` 不得打开。
* 数值契约只能从两条路取得，二者都还没做：
  1. 真实芯片/TEG 上的 MAU 实测；
  2. 按 SDM p.114 伪码 + p.148 的截断/舍入规则自建参考模型（会与 gpfn3 的
     行为冲突，必须显式标注为**模型**而非实测）。
* **可用结论**（结构性，不依赖乘加）：`fmfma` 的 dst 每 cycle 写 1
  longword / PE，4 cycle 共 4 longword，与 G0 的 longword 访问宽度结论
  一致；y 与 dst 同宽同位，因此 `dst` 与 `y` 可以取同一寄存器做就地累积。

#### 22.9.5 仍未闭合

* `fmfma` 的 x/y/dst 三行 width 是否与 §22.4 表格一致——本次只验证了
  width=1 下 dst 可写、可读回，**没能**通过"结果是否正确"来反证 x 的宽度；
* 向量形（`fvfma` 及 `_M` 掩码形）未测：从 MIR 直接写 `XFVFMA` 时 llc
  不产出指令，属后端接线缺口，与 MAU 语义无关；
* mask 寄存器对 `fmfma` 的作用未测。

### 22.10 厂商 SDK 头文件对操作数形状的独立确认（Level 2，2026-11-04）

来源（`suzhou`）：`/opt/pfn/pfcomp/hpc/mncl/include/mncl/device/mncore2.h`
与 `.../mncore2_builtin_wrappers.h`。这是厂商给 MNCL 用的 C++ intrinsic 声明，
**不是**我们从伪码推的期望，也不是模拟器行为——属于 §5 的 Level 2 同谱系
先验，可以单独支撑"形状/宽度"，但**不能**支撑数值。

#### 22.10.1 类型即访问宽度

| 类型 | 定义 | 每 PE 值数 | 长语数 |
|---|---|---|---|
| `f32x4` | `typedef DLW f32 f32x4` | 4 | 2 |
| `f32x4x2` | `typedef FULLCYCLE_LW f32 f32x4x2` | 8 | 4 |
| `bf32x4` | `struct { f32x4 __inner; }` | 4 | 2 |
| `bf32x4x2` | `struct { f32x4x2 __inner; }` | 8 | 4 |

#### 22.10.2 三条矩阵指令的权威签名

```
f32x4x2 __builtin_fmfma_raw(MAB m32x8x8*, f32x4 /*x*/, f32x4x2 /*y*/);
void    __builtin_fmwrite0..7(f32x4x2 /*src*/, MAB m32x8x8*);
f32x4x2 __builtin_fmread0..7(MAB m32x8x8*);
```

由此定案（与 §22.7 的实测一致）：

* `fmwrite` / `fmread` 每指令每 PE 搬 **4 个长语**（`f32x4x2`），即
  **width = 1**。§22.4 表里这两行是对的。
* `fmwrite{N}` 的 N 是**起始行**：`__builtin_matrix_loadf32(_with_bf)` 就是
  `fmwrite0 + fmwrite4` 两条铺满 8 行，所以一条 `fmwriteN` 覆盖 N..N+3 四行
  （头文件的 "Write row N" 是简写，不能按字面读）。这独立确认了 §22.7 的
  `mat` 立即数 = 2 × 起始行（MIR 0 和 8 → `$lx0`/`$lx4`，起始行 0 和 4）。
* `fmread{N}` 的 N 是**起始列**（头文件写明 "starting from column N"），
  与 §22.7 的"`fmread` cycle c 读第 `start_col + c` 列"一致。
* `fmfma` 的 x 是 **`f32x4` = 2 长语**（4 个 cycle 合计，每 cycle 1 word），
  y 与 dst 是 **`f32x4x2` = 4 长语**。§22.4 表中 x 记作 "4 word / width 0"
  在总量上正确。
* **`fmfma` 的加数与结果都是普通 IEEE f32，不是块浮点**：
  `__builtin_fmfma` 的注释写明 "The addend is single precision"，只有 x 经
  `__builtin_fbfnx4` 转成 `bf32x4`。"The matrix uses columns 0, 2, 4 and 6 of
  the 8x8 register" 也与 §22.1 一致。
* 存在 `__builtin_fmmul`（`d = A·x`，无加数）与 `_raw` 变体；`fvfma/fvmul/
  fvadd/fvpassa` 的 f32 形态是 `f32x4x2` 全宽向量运算。
* **写矩阵的标准序列是 `fbfn` → `fmwrite`**：`__builtin_matrix_loadf32`
  的注释是 "by emitting `fbfn` then `fmwrite` matrix-register write
  instructions"，而 `__builtin_fmwrite0` 明确 "no `fbfn` is emitted, so
  `from` is written as it stands"。这就把 §22.8.6 里"`fbfn` 与 `fmfma` 的
  组合未测"从**厂商文档**层面补上了：GEMM 的正确数据通路是
  f32 → `fbfn` → `fmwrite` → `fmfma`，且 `fbfn` 与 `fmwrite` 是两条独立
  指令（对应我们已有的 G1 与 G0 夹具，只是尚未在同一程序里串起来）。

#### 22.10.3 对 §22.9 的反向核对（关键）

既然 y 与 dst 都是 **IEEE f32**，§22.9 探针里 `y = 0x40C00000` 的真值是
**6.0**（同一个位型在 BF 口径下读作 4.0——这正是必须写清口径的原因），
而 dst 读回逐位等于 `0x40C00000`，即 6.0。
即：**在正确的 IEEE 口径下 dst 仍然逐位等于 y，A·x 一项贡献为 0**。
§22.9 的结论不依赖 BF/IEEE 的口径误读，成立。

同时这也修正了本节初稿里的措辞错误：`0x40C00000` 曾被写成 "BF(4.0)"，
而它作为 IEEE f32 是 **6.0**。结论不变——dst 是 y 的**逐位**复制，与用哪种
口径解读无关；只有"若乘加成立应得 10.0"这一句需要按 IEEE 口径写。

### 22.11 按 ABI 文档 + MLSDK 文档重建的契约（Level 2/3，2026-11-04）

指定的两份材料是 `llvm/lib/Target/MNCore/docs/mncore_kernel_abi_v1.md` 与
`https://dev.mn-core.com/sdk/0.9/MLSDK/docs/en/getting_started.html`。
重建的结论必须先说清楚：**两份材料都不含 MAU 数值语义**。能重建的是
"MAU 操作数与 ABI 参数槽的接口形状"，以及一条此前只有模拟器侧证据的宽度
单位；数值仍缺，门禁不动。

#### 22.11.1 各来源实际覆盖的范围（逐条核对，未外推）

| 来源 | 与 MAU 有关的内容 | 能否支撑数值 |
|---|---|---|
| `mncore_kernel_abi_v1.md` | ① 参数物理布局 v1（向量 slot = **2 长语**）；② kernel metadata 的资源保留项 `matrix`；③ 示例 entry 名为 `gemm_f32` | 否。全文对 `fmfma`/`fbfn`/块浮点**零提及** |
| MLSDK `getting_started` | 设备串 `"mncore2:auto"` / `"emu2"` / `"pfvm:cpu|cuda"`；`Context.compile` → `CompiledFunction` → `TensorProxy.cpu()`；`float_dtype`；`model.app` 与 `dump_gpfnapp` | 否。无 MAU / matmul / 操作数尺寸 |
| MLSDK `hardware_specification` | "Matrix processing units contained within each MAB"；存在 block floating-point 与 pseudo-single 格式，主要用于矩阵-向量点积；访问宽度 `$l`/`$ll` 的增量为 1/2/4 | 否。明确把 ISA 细节留给 dev manual |
| MLSDK `technical_notes` | 只有图层面的 GEMM dtype/layout 规划 | 否 |
| MLSDK `api_reference` | `float_dtype=mixed`（默认）把 **GEMM 的输入输出降为 half**；`float` 强制全 f32 | 这是**精度口径**，不是 MAU 语义 |

（`getting_started` 原文只在 setup/编译/取数上；`technical_notes` 原文只在
MNGraph/Layout/Scheduler 上。两者都没有 `m32x8x8`、`bf32x4`、`fbfn`。）

#### 22.11.2 唯一新获得的独立证据：访问宽度单位

`hardware_specification` 写明 `$l`/`$ll` 是**访问宽度说明符**、与地址单位
无关；`$m8`、`$lm8`、`$llm8` 从**同一地址**起、增量分别为 **1/2/4**。
（该页在 `suzhou:/opt/pfn/pfcomp/codegen/MLSDK/docs/en/hardware_specification.html`
有同版本副本，已核对；同一段落把块浮点细节指给
`mncore2_dev_manual_{en,ja}.pdf`，即 §22.1 已在用的 SDM，没有新增数值。）
这正是 packed `peoperand` 的 `width` 字段（0/1/2）的含义，此前只有 gpfn3
侧的实证（§22.7）。现在它有了一条**不依赖模拟器**的 Level 2 来源，因此
"按 (opcode, role) 生成访问宽度"的后端改造在**形状层面**已可开工——它不
需要数值 golden，也不会因此产生任何数值断言。

#### 22.11.3 ABI 参数槽 vs MAU 操作数宽度：一处真实冲突

ABI v1 冻结的向量参数落位是"`space = 1 + i % 4`，地址 `16 * (i / 4)` word，
**2 长语**"。与 §22.10 的厂商签名对齐后：

| 操作数 | 宽度 | 与 ABI v1 向量 slot 的关系 |
|---|---|---|
| `fmfma` 的 x（`f32x4`） | 2 LW | 恰好一个 slot |
| `fbfn` 的输入 / 输出（`f32x4` / `bf32x4`） | 2 LW | 恰好一个 slot |
| `fmfma` 的 y / dst（`f32x4x2`） | 4 LW | **两个 slot**，或新参数 kind |
| `fmwrite` / `fmread` 的 src / dst（`f32x4x2`） | 4 LW | **两个 slot** |

结论：GEMM 内核的 y 与 dst **不能**沿用"一个向量参数 = 一个 MAU 操作数"的
假设。**禁止**把 v1 的 2-LW slot 就地加宽——参数布局是冻结项，改动会破坏
已发布的 argument placement。允许的两条路：

1. 连续两个向量 slot 拼成 `f32x4x2`。受 `space = 1 + i % 4` 约束，只有
   `i % 4 ∈ {0,1}` 或 `{2,3}` 的相邻对落在同一 space、地址相差 16 word；
   lowering 必须显式校验这一点，否则拒绝。
2. 在 metadata 里新增 `kind: matrix`，由 lowering 直接生成 4-LW 落位。

选哪条要在 GEMM lowering 落地前一并定案，本文不预先决定。另外
`resources.matrix` 目前只有"存在这个保留项"，**没有任何占用单位定义**；
metadata 生成器只能写不透明占用并由 runtime 拒绝未知单位，**不得凭空给
一个数**。

#### 22.11.4 文档仍然完全没有定义的三项

1. `fmfma` 是否执行乘加（文档只给伪码；gpfn3 实测为 `dst := y`，§22.9）。
2. `fbfn` 的**公共指数选择规则**：4 个元素中谁定指数、如何舍入、溢出/
   下溢行为。SDM 只给解码式（§22.8）。
3. 掩码（`/0111` 等）对 MAU 的语义。

这三项**不能**从文档重建，只能实测或取得 dev manual。

#### 22.11.5 `emu2` 已实测可运行，且确为指令级数值模型（2026-11-04）

此前记录的"不可运行"是**环境缺口**，不是 SDK 限制。补齐后实测通过的
最小配方（suzhou，2026-11-04）：

1. python3.12：`~/miniforge3/bin/mamba create -y -n mncore312 python=3.12`
   （本机 `conda create` 因 libmamba 求解器插件损坏不可用）。
2. torch **2.9.0+cpu**，索引 `https://download.pytorch.org/whl/cpu/`。
   版本不是猜的：MLSDK 容器构建参数写死 `ARG torch_version=2.9.0` /
   `ARG torch_index_url=https://download.pytorch.org/whl/cpu/`，
   `examples/ncf_training/requirements_training.txt` 同样 pin `torch==2.9.0`。
   实测 2.15 / 2.6 / 2.4.1 / 2.3.1 全部失败：`libpfvm_runtime.so` 需要
   `c10::SymInt::sym_gt_slow_path` 与
   `c10::impl::cow::materialize_cow_storage` **同时**存在；2.15 只有前者、
   2.3–2.6 只有后者，2.9 是唯一两个都导出的版本
   （`nm -D .../torch/lib/libc10.so` 两个均为 `T`）。
3. 依赖按 import 缺口逐个补齐：`pfio numpy retrying pytorch-pfn-extras
   onnx onnxscript accelerate`。
4. **必须在 `sde64 -hsw` 下运行**：直接执行 python 会在
   `grape_pfn::codegen::MemoryUnitSet::MemoryUnitSet()` 处 SIGILL
   （宿主 CPU 缺 AVX-512，与 assemble3 / gpfn3 同一原因，不是 MLSKD 的问题）。
5. 环境变量：
   `source /opt/pfn/pfcomp/codegen/build/codegen_pythonpath.sh` 加
   `LD_LIBRARY_PATH=$P/lib:$P/lib/python3.12/site-packages/torch/lib`。

实测结论（`a(8×4) @ b(4×4)`，`options={"float_dtype": "float"}`）：

* **能算**：整数数据下结果与 CPU `a @ b` **逐位相同**（max abs diff = 0）。
* **但不是 f32 直算**：把输入换成混合量级的满尾数随机数后
  max rel diff = **4.38e-6**（不再逐位相等）。emu2 复现的是降低精度的
  MAU 数据通路，不是宿主 f32 matmul。**推论：`float_dtype=float` 不等于
  "精确 f32 GEMM"，验收不能用 f32 位精确作判据。**
* **确实是指令级**：厂商为同一张图生成的 `model.vsm` 里，GEMM 节点由
  `gmwrite` / `gbfn` / `gmmul` 组成（`gmmul $ly $aluf -> $ln0/1000`），
  **不是 `fmfma`**。把这串指令交给 gpfn3 时，模拟器会逐指令校验块浮点载荷，
  非法载荷直接抛
  `Invalid block float format. The exponent of 2th element is not equal to
  the common exponent 0. Maybe BFn/BFe is lacking.`
  —— 同时证明 (a) 模拟器在执行指令而非跑图；(b) `g*` 家族要求操作数先经
  `gbfn` 转换（把裸 f32 直接喂给 `gmwrite` 会被拒）。

对 G2 的直接影响：

* `emu2` 现在是**可用的指令级参考模型**，这正是 §22.11.4 第 1 项的解锁条件。
* 但它首先暴露的是**指令选择问题**：厂商的 f32 GEMM 走 `gmmul`（伪单精度
  家族），我们的后端只发 `fmfma`（f32 家族）。两者都挂在 `m32x8x8` 上，
  `mncore2.h` 明确把 `g*` 描述为 "pseudo-single-precision counterparts …
  select the MAU's pseudo-single-word mode"，且
  `__builtin_fmfma_raw(MAB, f32x4, f32x4x2)` **独立确认了 §22.11.3 的按角色
  宽度**（x = 4 word = 2 LW，y/dst = 8 word = 4 LW；`__builtin_gmmul`
  则取 `f32x4x2` 载荷）。
* 因此 **G2 仍未闭合**，但阻塞点已经从"没有参考模型"变成
  "`fmfma` 在本模拟器上不产生乘积（§22.9 的 `dst := y`），而厂商 GEMM
  根本不用它"。下一步最小实验：用合法 `gbfn` 载荷复现 `gmmul`
  （8×8 矩阵需 8 次 `gmwrite`），再决定 lowering 是改发 `gmmul` 序列，
  还是先用 `mncl` 的 `__builtin_matmulf32` / `__builtin_gmmul_bpf`
  生成对照。

**`pfvm:cpu` / `pfvm:cuda` 不是 MAU 的参考模型（2026-11-04 补充）**：厂商文档
写的是 PFVM "traces PyTorch programs to construct and optimize computation
graphs"，"both cpu and cuda execute computations by calling PyTorch's C++ API"，
而 "Unlike emu2, which runs directly on the CPU"。即 `emu2` 走的是编译后的
MN-Core 程序（指令级），`pfvm:*` 走的是 PyTorch 图执行——没有 `fmfma`、
没有矩阵面、没有块浮点，只能复现"数学上应得的结果"（本机 numpy 已经能做）。
因此 `pfvm` **不能**替代 G2 的单指令 golden，也不能证明我们的 lowering 正确；
它至多用于端到端黑箱对照，且同样受 `float_dtype` 口径与 torch 依赖（本机无）
限制。`pfvm:cuda` 还要求有 CUDA 设备，本机无。

→ 状态 UNVERIFIED。G2/G3 不放行，`tl.dot` 保持 fail-closed（§22.5）。

### 22.12 mncl→emu2 实测：矩阵面可用，`$mauf` 读回的乘积不可用（2026-11-05）

**证据等级**：模拟器实测（Level 4）+ 厂商二进制字符串（Level 2）。本节把
§22.11.5 的"下一步最小实验"做完了，结论是**否决了这条取证路线**，
并给出否决的理由与它对 lowering 的约束。

#### 22.12.1 基线与反例（先说通路没问题）

* **基线**：厂商自带例子
  `share/mncl/examples/f32_f16_roundtrip`（纯 ALU/传输，**无 MAU**）用
  `mnclc` 编成 `.bin`、在本机 `libmncl` 的
  `CL_DEVICE_TYPE_EMU_MNCORE2` 上跑通：`max_rel_err = 9.76e-4`、`[OK]`
  （边角输入 f16 上下溢全部按预期）。所以 `mnclc` 产物 + emu2 通路本身
  是好的，下面所有异常都不来自"kernel 根本没跑"。
* **反例（同一 emu2、同一 host 程序）**：MLSDK 编译的 `a(8×4) @ b(4×4)`
  与 torch 结果**逐位相同**。所以 emu2 的 MAU **是实现了的**。

#### 22.12.2 矩阵面本身可用

`probe5`（`/tmp/mncl_t/mau_probe5.c`）：
`fbfn → fmwrite0/fmwrite4 → fmread0/fmread4`，上四行写 1.0、下四行写 2.0。
读回（PE0）为 `1.5 1.5 1.5 1.5 3 3 3 3`。

这不是错误：`fmread` 返回的是**块浮点原始字型**，`1.0` 的 BF 字是
`0x3FC00000`，按 IEEE f32 读正是 1.5；`2.0` 的 BF 字 `0x40400000` 读作 3.0。
与 §22.8.2 的"显式前导 1"编码自洽，也与 §22.10 的"`b*` 载荷是不透明的"
一致。**结论：写矩阵、读矩阵、BF 编码在 emu2 上都是对的。**

#### 22.12.3 `$mauf` 读回的乘积不可用

四个探针（`mau_probe2/3/4/6.c`）覆盖 f/g 两族 × `mmul`/`mfma`，
全部走 mnclc 生成的 `… -> $nowrite` + `lpassa $mauf -> $lm…`：

| 探针 | 意图 | 实测 |
|---|---|---|
| probe2 | f 族；行常矩阵 vs 列幂矩阵；`x={1,2,4,8}` | 两组矩阵**同一结果**，三组输出**同一结果** |
| probe3 | g 族，同上，`x={1,…,128}` | 同上（数值恰为 probe2 的 3 倍） |
| probe4 | 规范填充（`gmwrite0/4`）；全 1 矩阵 vs 全 0 矩阵；`gfma` 加数 `{100,…,800}` | **四个输出全为 12**：零矩阵非 0、加数完全不出现 |
| probe6 | 同 probe4，但 BF 向量改成**单次使用临时量**，迫使分配器靠近使用点 | 仍为 **12**，且 `-O0/-O1/-Osize` 下产物一致 |

判决性的一条：`gmmul` 对**全零矩阵**与**全 1 矩阵**给出**逐位相同**的结果。
任何线性模型都不可能同时成立，所以这不是"列映射猜错"，而是**乘积没有
进到被读的那个寄存器**。

#### 22.12.4 差异在哪里：操作数空间

对比同一 emu2 能跑对的厂商 GEMM（§22.7.5 的 `model.vsm`）与 mnclc 产物：

| | 厂商 MLSDK 生成的 GEMM | mnclc 生成的 MAU |
|---|---|---|
| 矩阵写入源 | `gmwrite $lbf -> $lx0` / `$aluf` | `gmwrite $lm0v -> $lx0` |
| 乘法向量源 | `gmmul $ly $aluf` | `gmmul $lx $lm156v` |
| 结果去处 | `-> $ln0/1000`（真实目的寄存器） | `-> $nowrite`，靠 `lpassa $mauf` 读回 |

`libmncl.so` 内含字符串
`convert_instruction_to_uop: MAU input supports only GRF / dram / forward /
treg yet (temporary)`——emu2 的 MAU **计算输入**只实现 GRF / DRAM /
转发锁存器 / T 寄存器，**本地内存 LM 不在其中**；而 mnclc 恰恰一律把
MAU 输入放在 `$lm…`，并从 `$mauf` 取结果。

这是**解释**而不是**证明**：字符串只表明该检查存在，没有日志证明它在本
次运行里触发（`EMU_LIME_UOP_CHECK=1`、`GLOG_v=2` 都没有新输出；
`EMU_IEEE754_MAU` / `EMU_LIME_MAU_SOA` /
`EMU_MAU_FLUSH_UPPER_SINGLEWORD` 三个开关也不改变结果）。但四组合完全
一致地指向同一处，且反例（厂商形式）在同一模拟器上正确，这是目前唯一
能同时解释全部数据的假设。

#### 22.12.5 对 G2 / lowering 的影响

* **G2 仍在原位，但取证路线被否决**：`emu2` 能算的只有**厂商那种操作数
  形式**（转发锁存器 + 真实目的寄存器），`gpfn3` 能接受的是 vsm 但乘积项
  缺失（§22.9）。两条模拟器路线在"用我们自己后端发出的指令去验证乘积"
  这件事上**同时不通**，且原因是同一个：**指令形式**，不是数值语义。
* **对 lowering 的硬约束**：若希望产物可在 emu2 上数值验证，必须发出
  `$aluf`/`$lbf` 一类转发锁存器作 MAU 输入、并把结果写进**真实目的
  寄存器**，不得用 `-> $nowrite` + `$mauf`。这条约束先于"选 f 族还是 g 族"
  ——两族在 emu2 上都受同一限制。
* **数值验收口径**（来自 §22.11.5，仍是唯一可用的实测口径）：整数数据
  逐位一致；满尾数相对误差 ~4.4e-6 属 MAU 正常的降低精度行为，不能用
  f32 位精确当判据。
* 产物与探针留在 `suzhou:/tmp/mncl_t/`（`mau_host.cc` 为 host 驱动，
  `mau_probe{2,3,4,5,6}.c/.bin`，`probe*_dump.txt`），未入库：它们是
  一次性取证工具，不是回归夹具。

#### 22.12.6 仍未闭合

* 尚无任何一次"我们后端发出的 GEMM 指令流"的数值 golden；上面所有数值
  都属于**厂商编译器的产物**。
* `$mauf` 在真机上是否等价于 `-> $ln` 的目的寄存器语义，未知；本轮只能
  确认 emu2 不给出等价行为。
* `dmfma`/`hmfma` 与 f16/f64 矩阵面在 emu2 上的行为未测。

### 22.13 f32 矩阵角色宽度落地与 `mncore.gemm.tile` 结构金标（2026-11-05）

**证据等级**：文档+代码形状（本机）+ `assemble3` 实机接受（suzhou）。

本轮把 §22.4 / §22.10 / §22.11.2 推出的**按角色访问宽度**从纸面落到
intrinsic ABI 和 MLIR 类型上，并把一个 8×4×4 的 GEMM tile 接到后端。
**本轮没有产生任何数值 golden**——§22.12 已经把"用我们自己发出的指令流
验证乘积"这条路否决了。所以这次交付的判定词是**形状合法性**，
不是**数值闭合**，两者不得混谈（§22.12.5、§22.11.4）。

#### 22.13.1 类型契约（唯一真源：`IntrinsicsMNCore.td`）

| 角色 | 改前 | 改后 | 依据 |
|---|---|---|---|
| `fmfma` x | `<4 x float>`（2 LW） | 不变 | 1 word/cycle × 4 |
| `fmfma` y / dst | `<4 x float>` | **`<8 x float>`（4 LW）** | 1 LW/cycle × 4 |
| `fmwrite` src | `<4 x float>` | **`<8 x float>`** | 1 LW/cycle × 4 |
| `fmread` dst | `<4 x float>` | **`<8 x float>`** | 1 LW/cycle × 4 |

* `llvm.mncore.matvec.v4f32` 的名字后缀描述的是 **x**——唯一与向量族不同
  的窄角色；返回类型与 y 都是 `v8f32`。名字不再能推出全签名，
  `MNCoreLowerTypedIntrinsics` 的 `checkShapes` 逐角色重校整条调用，
  名字只用于选 opcode。
* `matwrite_v4f32` / `matread_v4f32` **删除**。留着它们等于允许前端发出
  一条只搬一半数据的 `fmwrite`/`fmread`；宁可编译期报错。

#### 22.13.2 槽位：16 word 本来就够

分配器按 **16 word / slot** 布（因为 width=2 的操作数是 4 cycle × 4 word）。
一个 4 LW 矩阵操作数按自己的自然宽度（1 LW/cycle）只访问该 slot 的
**8 word**，所以**不需要更宽的 slot，槽模型未改**。

这**不是**说 §22.11.3 的冲突已经解决：那里问的是"一个函数参数怎么喂进
4 LW"（选项 1 相邻两 slot 拼接，选项 2 新增 `kind: matrix`），本轮两条都
没有选，ABI v1 的参数布局未动。当前只做到"给定 4 LW 值，后端能正确地
发指令"。

#### 22.13.3 MLIR：`mncore.gemm.tile`

```
mncore.gemm.tile <face = 0> (a_lo, a_hi, b, c0) : (v8f32, v8f32, v4f32, v8f32) -> v8f32
```

语义 `C = A(8×4) · B(4×4) + C0`，一个矩阵面。`a_lo`/`a_hi` 是 A 的行
0–3 / 4–7，`b` 是 matvec 的 4 word 输入（对应 §22.7 的偶列 0/2/4/6
契约），`c0`/结果都是 4 LW slab。

verifier：`face ∈ {0,1}`；`a_lo`/`a_hi`/`c0`/结果必须是 `<8 x f32>`；
`b` 必须是 `<4 x f32>`。三个负向 fixture 分别只出第一条诊断：
`matwrite_2lw`、`matvec_narrow_y`、`gemmtile_wide_b`。

#### 22.13.4 lowering 产物与 assemble3 实测

`mlir/test/Integration/MNCore/gemm.mlir` 端到端（mlir-opt → mlir-translate
→ llc -mtriple=mncore2）：

```
	fmwrite $lr0v $lx0
	fmwrite $ls0v $lx4
	fmfma $lx $m0v $ln0v $lr0v; fmwrite $lr0v $ly0
```

* 两次 `fmwrite` 落在行 0 / 4（§22.7.2 的 `(row+c) mod 8` 规则），一次
  `fmfma`；结果用 `fmwrite` 落到**另一个面**，满足 packetizer 的
  "同一面每 packet 至多一次"。
* `assemble3 --instruction-mode flat`（**hazard 检查开启**）：rc=0，
  无诊断，产出 3 条二进制指令。
* `run_regression.sh`（本地 FileCheck + 远端 assemble3/gpfn3 + MAU
  G0/G1/G2 金标 + 远端 hazard 阶段）：**本地 0 失败 / 远端 0 失败**，
  `layerb`、`layerb2` 在 hazard 阶段也 PASS——即按角色宽度改过之后的
  指令流是 hazard 合法的。
* `mlir/test/Integration/MNCore/e2e.sh` 改为默认遍历目录下全部 `*.mlir`
  （并拒绝"零指令"产物），当前 6/6 通过。

#### 22.13.5 本轮**不**证明什么

* **不证明乘积**：gpfn3 上 `fmfma` 仍然 `dst := y`（§22.9），emu2 上
  mnclc 形式的 `$mauf` 回读不可用（§22.12）。
* **不证明偶列约定就是硬件的有效 8×4**：它来自 §22.7 的 matvec 契约，
  本轮没有新的实测。
* **不证明指令族选对**：厂商 GEMM 发 `g*`（§22.11.5），我们发 `f*`；
  两者在模拟器上的行为差异原样保留。
* `gemm.mlir` 文件头已显式写明它是 **STRUCTURAL golden**（钉指令选择，
  不钉数值）。
* 因此 §22.13 不改变任何门禁：G0/G1 已闭，**G2/G3 原位**，`tl.dot`
  保持 fail-closed。

#### 22.13.6 下一轮要动数值时的前置条件

不是继续加形状，而是先满足 §22.12.5 的硬约束：MAU 输入必须发转发锁存器
/GRF 形式，结果必须写进**真实目的寄存器**（不得 `-> $nowrite` + `$mauf`）。
在此之前，形状与 lowering 再完善也不会产生可验证的数。
