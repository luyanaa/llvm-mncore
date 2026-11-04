# LLVM 产物 → MNIR → `mnrt` 动态 offload：接缝调查 (2026-11-04)

本文回答一个具体问题：**LLVM 后端已经产出 `.vsm` 之后，怎样把这些产物接进
MNIR / MN-Core 运行时，最终做到动态 offload。** 所有结论都在 `suzhou` 上
可复现；本文只记录**实测到的**事实与由事实推出的设计，不写"应该大概如此"。

---

## 0. 结论摘要（先看这一段）

1. **MNIR 是编译器 IR，不是运行时格式。** 设备唯一可装载的二进制是
   `mncore::Program`（FlatBuffers）：**变长头 + N×128 B 指令块**。MNIR
   （文本或二进制 AST）只是生产 Program 的一种输入。
2. **供应商的 MNIR 只提供了两个可用接缝**：
   * `mncl::mnir::Driver::CompileAll()` —— 输入 `.mnir`/`.mnbc`，输出
     `mncore::Program`；**已验证可用**；
   * `mncl::mnir::Compile(CompileOptions)` —— **本版本是 2 字节 `ud2` 桩，
     调用即 SIGILL，禁止使用**。
3. **`--dump-mnir` 的文本不是可回灌格式。** 它打出来的 `immu 16` 无法被
   同一个解析器读回（正确拼写是 `imm u 16`），且一旦目标是 `alloc(GRF0,…)`
   变量会命中一条真实的 `ud2`。⇒ **MNIR 文本不能当交换格式用。**
4. **`.mnbc`（二进制 MNIR AST）可以当交换格式用**，且已经证明是**逐字节
   等价**的：`gen14.c` → mnclc 产 `.mnbc` → 外部驱动重新编译 → 产出的
   **49408 B 指令镜像与 `mnclc -o` 的输出完全一致**（只有头部里的内核名
   不同）。探针：`~/libmnrt/tools/mnir_probe/check_mnir.sh`。
5. **对 LLVM 后端而言，MNIR 不是合适的接缝**：LLVM 发的是指令级文本
   （`.vsm`，每条对应 292-bit PE / 82-bit MV），MNIR 在其**上一层**（有
   `alloc`/`cast`/切片/循环）。把 LLVM 降到 MNIR 等于重写一遍后端。
   正确的关系是：**两者都是 Program 的生产者，在 Program 层汇合。**
6. 因此动态 offload 的落点是 **`mnrt` 的 provider 分层**（§5），不是"让
   LLVM 说 MNIR"。

---

## 1. 供应商流水线的完整剖面（实测）

```
k.c ──clang──► LLVM IR ──mncore::translator::Translator──► MNIR AST
                                                             │
                          ┌──────────────────────────────────┴───────────────┐
                          │ mncl::mnir::Driver::CompileAll                   │
                          │  (.mnir 文本 | .mnbc 二进制 AST)                 │
                          └──────────────────────────────────┬───────────────┘
                                                             ▼
                                              mncore::Program (FlatBuffers)
                                                             │
                          ┌──────────────────────────────────┴───────────┐
                          ▼                                              ▼
                 cl host API（设备）                          assemble3（仿真器）
       clCreateProgramWithBinaryPath → clSetKernelArg        .vsm → GPFN3/4 app
```

证据（均在本机 `suzhou` 实测，2026-11-04）：

| 事实 | 证据 |
|---|---|
| `mnclc` 有 `--dump-mnir` / `--dump-mnbc` | `mnclc --help` |
| 两个 dump 都落盘为 `<entry>-<out>.{mnir,mnbc}` 与 `-opt` 变体 | 实测 gen14 |
| `Driver::CompileAll` 接受 `.mnir` 与 `.mnbc` | `libmnir.so` 字符串 `fatal: expected .mnir or .mnbc, but got unknown file format` |
| MNIR 路径只支持 MNCore2 | 目标传 1 时 `Check 'target == mncore::Target::MNCore2' failed! the MNIR path supports MNCore2 only` |
| host 侧二进制入口是 Program | `mncl/host/cl/cl.h`：`clCreateProgramWithBinaryPath` |

`libmnir.so` 导出的公开头文件（`/opt/pfn/pfcomp/hpc/mncl/include/mncl/mnir/`）
里有完整的东西：`parse.h` / `print.h` / `driver.h` / `compile.h`，以及
**ANTLR 语法 `mnir.g4`** 和 **FlatBuffers schema `mnir.fbs`**。也就是说
MNIR 文本格式本身是公开文档化的——坏的不是语法，是实现。

---

## 2. 三种容器必须分清（这是最容易混的一步）

| 容器 | 生产者 | 消费者 | 特征 |
|---|---|---|---|
| **`mncore::Program`** | `mnclc -o`、`Driver::CompileAll` | 设备 runtime（cl API） | 变长 FlatBuffers 头 + N×128 B 块 |
| **GPFN app** | `assemble3 --binary` / `--app2` | `gpfn3_package_main`（仿真器） | `--binary` 在偏移 **64** 有 magic `"GPFN3"`；`--app2` 是另一版本，无该 magic |
| **裸 `.vsm` 文本** | `llc`（本后端） | `assemble3` | 每行一条指令 |

**Program 的头部长度不是常量**，它随重定位表增长。用
`hdr = len(image) − instblock` 实测：

| 样本 | fixup 数 | args | instblock | 文件 | 头长 |
|---|---|---|---|---|---|
| gen28 | 1 | 2 | 256 | 512 | **256** |
| gen6 | 3 | 1 | 2176 | 2496 | **320** |
| gen26 | 3 | 2 | 768 | 1088 | **320** |
| gen10 | 4 | 2 | 1024 | 1344 | **320** |
| gen47 | 12 | 2 | 3072 | 3584 | **512** |

这与 [`mncore_kernel_abi_v1.md`](mncore_kernel_abi_v1.md) §13 里"模板溯源"
发现的 256/320/512 三态是**同一件事**：头部是 FlatBuffers，头部里带着
fixup 表，所以早期写死 256 的提取器才会切错。（arg 表**不**改变头长——
§6 A.1 的 1→2 args 差分里文件大小不变。）

> **推论**：LLVM 侧要直接产 Program，缺的不是"块编码"（已有
> `block_encoder.py`），而是**头部 FlatBuffers 的 schema**。schema 可通过
> 头文件或实测反解，见 §6。

---

## 3. MNIR 文本：为什么不能当交换格式

`--dump-mnir` 的输出看起来像汇编，实际上有三处不可逆：

1. **拼写不对称。** dump 打 `immu 16`，而语法（`mnir.g4`）里是
   `imm : 'imm' 'u'?;`，解析器只认 `imm u 16`。把一整份 dump 里的
   `immu` 批量改成 `imm u` 后，`mnclc` 的 MNIR 驱动仍报
   `line 6:0 no viable alternative at input 'imm'`。
2. **解析器会段错误级地挂掉。** `immu 16 -> a/1000` 单独可以过；
   加上 `var a = alloc(GRF0, 1, lw, temp)` 之后，进程在
   `libmnir.so` 的 `mncl::mnir::Parse()` 里执行到一条 `ud2`
   （vaddr `0x1e5416`，位于 `std::from_chars<unsigned long>` 的内联实现里，
   即"空输入被当成 unreachable"）。同样输入给 `zero -> a/1000` 则正常
   返回诊断。**这是实现缺陷，不是我们用法不对**——同一版本自己的 dump
   自己也读不回来。
3. dump 里带 `#MNIRRegionHandle` / `#unknown-loc` 这类只供人看的注解。

结论：**`--dump-mnir` 是调试输出，不是序列化格式。** 任何依赖"dump →
改 → 回灌"的路线都不成立。

**`.mnbc` 没有这个问题**：它是 schema 化的 FlatBuffers AST，`CompileAll`
直接吃，且结果可验证（§4）。

---

## 4. 已验证的接缝：`.mnbc` → Program，逐字节等价

复现脚本：`suzhou:~/libmnrt/tools/mnir_probe/check_mnir.sh`（用法
`check_mnir.sh <kernel.c> [entry]`）。它对同一个 C 源做两条独立路线：

```
(a) mnclc --allow_pe_args --mnir --dump --dump-mnbc -e k -o ref.bin k.c
(b) 上面的 <entry>-*-opt.mnbc ──► mnircc ──► out.bin
```

然后比较两个 Program 的**尾部 `size of instblock` 字节**：

```
PASS  gen14.c  Program 49664 B / 49664 B, image 49408 B identical
```

即：**指令镜像逐字节相同**，两个文件只在头部（内核名 `k` vs `entry`）
有差异。这条链路把"供应商的 MNIR 实现"变成了一个**可脚本化、可回归的
黑盒**，也是判断"我们的 Program 是否等于供应商的 Program"的现成 oracle。

关键一点：**带 fixup 的 kernel 同样成立**——`gen28`（fixup=1）、
`gen26`（fixup=3）、`gen10`（fixup=4）、`gen47`（fixup=12）全部 PASS。
也就是说重定位信息存在于 `.mnbc` AST 里，`Driver::CompileAll` 会把它
重新生成进 Program，**不会在这一步丢**。对动态 offload 来说这是必要
条件（§5.2）。

**全样本广度（2026-11-04，`sweep_mnir.sh`）**：`mncl_samples/` 的 116 个
kernel 中 **92 个 PASS**；24 个 FAIL **全部是 `mnclc` 自己编译失败**，
**没有一个是镜像不一致**（`grep -c "image differs" == 0`）。其中 2 个
只是 kernel 不叫 `k`（`ka4.c`→`kname`、`ka5b.c`→`k2`），用正确 entry
重跑后 PASS ⇒ 实际 **94/116 通过，22 个源码本身被供应商编译器拒绝**
（`cannot put data to far memory 5`、LM0 立即数、PDM 仅 L1IR、未知
builtin、`value.cc` / `mnir_backend.cc` 断言等）。带 fixup 的样本全部
在通过之列。

实现细节（`tools/mnir_probe/`）：

* `mnircc.cpp` —— 只调 `mncl::mnir::Driver::CompileAll`；
* `shim/` —— 三份最小 ABI 兼容桩头文件。**原因**：`driver.h` 依赖
  `flatbuffers/*` 与 `mncl/device/translator_generated.h`，而这两个在
  安装树里**都没有**（`flatbuffer_builder.h` 全机搜不到，`flatc` 也不在）。
  桩只补 `Driver` 声明所需的最小面；`Result<>` 用的是**真**头文件
  `diagnostics.h`。因此 `mnircc` 的正确性靠"输出了与 mnclc 相同的镜像"
  来证明，而不是靠"能编译过"。
* **不要**用 `mncl::mnir::Compile(CompileOptions)`：`nm -D -S` 显示其
  符号大小 **2 字节**，调用即 SIGILL（`ud2` 桩）。

---

## 5. 动态 offload 的接入设计（目标形态）

`mnrt` v1 的定位（`libmnrt/README.md`）是 "synchronous enqueue; `.vsm` text
input; bindings recorded but not relocated"。要支撑动态 offload，按
**输入层**而不是"语言层"拆 provider：

| provider | 输入 | 后端 | 状态 | 用途 |
|---|---|---|---|---|
| `emulator`（现状） | `.vsm` 文本 | `assemble3` → `gpfn3_package_main` | 已实现、已过 E3 | 开发/回归 |
| **`program`（新增，目标）** | `mncore::Program` | 直接交给设备 runtime（cl host API） | 未实现 | **真实 offload** |
| **`mnir`（新增，开发期）** | `.mnbc` | `Driver::CompileAll` → `program` provider | 原型已通（本文 §4） | 供应商等价 oracle、调试 |

要点：

1. **Program 是唯一的长期二进制 ABI。** `.vsm` 与 `.mnbc` 都只应是
   "编译期输入"，运行期只认 Program。这与设备侧 cl API 的实际契约一致。
2. **relocation 在 Program 的头部（fixup/arg 表）里**，不在指令流里。
   这解释了 §2 的头部尺寸变化，也解释了为什么 `mnrt_bind_buffer()`
   现在只能"记录"：v1 没有 Program 层的绑定实现。编译期的 v2 契约
   （`.vsm` 行尾 `# @mncore.fixup <site>` + sidecar，见
   [`mncore_kernel_abi_v1.md`](mncore_kernel_abi_v1.md) §13）是它的**输入
   来源**，不是它的替代品。
3. **异步**：v1 是同步的（`mnrt_enqueue` 内部 `system()` 两次）。真实
   offload 需要对映 `clEnqueueNDRangeKernel` + `cl_event` 的语义；
   provider 能力位（`mnrt_info.feature_bits`）已经预留，现在恒为 0
   （"no async, no relocation yet"）。
4. **多 kernel / Program 常驻**：动态 offload 意味着一个 Program 反复
   入队。v1 每次入队都要重写临时 `.vsm` 并重跑 `assemble3`，这是必须
   消掉的成本项。

---

## 6. LLVM 后端怎么落到 Program 上

三条路，按推荐顺序：

### A. 直接构造 Program（推荐，长期主线）

```
llc → .vsm ──► canonical.py ──► block_encoder.py ──► N×128 B 镜像
                                                          │
                                          + Program 头（FlatBuffers）
                                                          ▼
                                                  mncore::Program
```

* 块编码已有；**缺口是头部 schema**。
* 头部内容可从三处交叉确定：`mnclc --dump` 的头部清单（`target` /
  `# of kernels` / `name` / `# of args` / `# of fixup` /
  `size of instblock` / `debug info`）、§2 的三种实测头长、以及
  `_createRelocatedKernelVsm` 的反编译（已确认 offset 0 是头尺寸、
  后续字段为相对偏移、payload 是长度前缀的 128 B 记录向量）。
* 验收即本文 §4 的等价判据：**对同一 kernel，头部之外必须与供应商的
  Program 逐字节一致**。

#### A.1 反解的起点：已确认的结构（2026-11-04 差分实验）

头部**是真正的 FlatBuffers**（自带 vtable），不是手写结构。以 `gen14`
（1 kernel / 1 arg / 0 fixup / instblock 49408 / 文件 49664）为例：

* 偏移 0 是 root uoffset `48`，root table 在 48；
* 偏移 40 是它的 vtable：`vtable_size=8`（2 个字段）、`table_size=16`、
  `f0` 缺省、`f1`(+8) = 8；
* 偏移 80 是另一个 vtable（`size=16`、`table_size=56`、6 个字段，全为 8
  字节槽）；`--scan` 会把它算出的候选 table 同时报在偏移 96 和 128。
  二者跨 56 字节互相重叠，**最多只有一个是真的**——这是"无 schema 扫描"
  的固有限制，需要靠差分（下一条）定夺。
* 偏移 104 处的 `+48` 指向内核名：**字符串长度在 152，字符从 160 开始**
  （8 字节对齐的串布局；`k`→`kk` 差分确认）。

差分实验（同一份源码只改一处，`fbwalk.py` 逐字段对比）：

| 改动 | 文件大小 | 头长 | 变化字节 |
|---|---|---|---|
| 内核名 `k` → `kk`（1 arg） | 49664 → 49664 | 256 → 256 | 仅 **152**（串长 1→2）与 **161**（第二个字符） |
| 1 arg → 2 args | 49664 → 49664 | 256 → 256 | 仅 **112** 与 **120**：`49528/49504` → `49512/49480` |

由此可以**修正一个容易搞错的直觉**：头部长度跟着 **fixup 数**走
（0/1→256、3/4→320、12→512，见 §2），**不跟 arg 数走**——本组实验里
arg 数 1→2 时文件与头长都没变，只有两个 8 字节字段移动（它们更像
"数据区地址/长度"，随参数占用的设备空间变化）。

复现：
```bash
sde64 -hsw -- mnclc --allow_pe_args --mnir --dump -e k O -o a.bin a.c
python3 -I fbwalk.py a.bin        # 打印 root table / vtable / 字段候选
```

**头尾固定 72 字节区（已批量验证）**：对 `mncl_samples/*.bin` 的 106 个
Program 做统一检验——假设"头长 `hdr ∈ {256,320,512}`，且
`u64(hdr − 72) == len(file) − hdr`"（即头部最后 72 字节的首字段就是
instblock 字节数）：

```
HIT=104 MISS=2       (python3 -I hdr_probe.py mncl_samples/*.bin)
```

104 个命中里既有 256 头的 gen14/gen28，也有 320 头的 gen26；另外 4 个
256 B 的**纯头文件**（空镜像）按 `size == 0` 单独判定为命中。2 个未命中
都可解释，**没有一个是"规则被推翻"**：

* `wrapped42.bin` 551 B：`551 − 256 = 295` 不是 128 的倍数，
  **它根本不是 Program**（不是合法容器）；
* `gen16.bin` 2048 B：其 184..255 区被一串小整数
  （`2,1 / 4,2 / 6,3 / 8,4`）填满，instblock 大小（1792）改出现在偏移
  104。它的源文件现已无法编译（`__0 is an immediate destination but is
  fixed on LM0…`），**来源不可复现**，因此只记为"另一种布局"，不作为
  规则的反例使用。

这条规则的意义：即使 schema 未知，**块大小在头里的位置已经确定**，
程序可以先用它切分头/镜像（比按文件长度猜头长更可靠）。它独立复现了
§2 那张表的每一行——gen6 `320/2176`、gen26 `320/768`、gen47 `512/3072`、
gen28 `256/256`，**完全不依赖 `--dump`**。

**内核名字段的位置关系（3 个样本上一致，未推广）**：三个头部布局不同的
样本里，指向内核名的那个字段总在"名字前面 48 字节"，而 kernel table
的起点总在"名字前面 56 字节"：

| 样本 | root uoffset | kernel table | 指向名字的字段 | 名字长度/字符 |
|---|---|---|---|---|
| gen14（1 arg/0 fixup） | 48 | @96 | @104 (`+48`) | 152 / 160 |
| gen28（2 args/1 fixup） | 24 | @72 | @80 (`+48`) | 128 / 136 |
| gen16（另类布局） | 32 | @80 | @88 (`+48`) | 136 / 144 |

这足以说明 kernel table 的形状是**固定的**（6 个 8 字节槽），只是整体
在头里浮动；但**其余 5 个槽的语义仍未确定**，不能据此写构造器。

### B. 复用供应商 MNIR（作为 oracle，不作主路径）

`.mnbc` 路线已可用，但它要求我们先有 MNIR AST，而 LLVM 后端没有、也不
应该有。定位：**只用来验证 A**（同一 kernel 的两份 Program 互相比对）。

### C. `.vsm` → `assemble3` → GPFN app（只对仿真器）

现状即此。注意 `assemble3 --binary` / `--app2` 产的是 **GPFN app**
（`--binary` 在偏移 64 有 `"GPFN3"` magic，`--app2` 为另一版本），
**不是 Program**；且 MNCL loader 拒绝 assemble3 的封装（§13 已记录）。
所以 C 永远不能升级成真实 offload 路径。

---

## 7. 验收与回归

已加入 `~/libmnrt/tools/mnir_probe/`（独立目标，不并入 `make check`，
因为它需要 `sde64` 与 mncl 安装）：

```
tools/mnir_probe/check_mnir.sh <kernel.c> [entry]   # 单样本
  PASS  镜像逐字节等价
  FAIL  mnclc 失败 / 无 .mnbc / 镜像不一致（打印首个差异字节）

tools/mnir_probe/sweep_mnir.sh [样本目录]            # 全样本，xargs -P 3
  结果写到 $MNIR_SWEEP_LOG（默认 ~/mnir_sweep_last.txt），末尾 PASS/FAIL/TOTAL
  注意：sweep 固定用 entry `k`，非 `k` 的核会显示为 FAIL（用 check_mnir.sh
  指定 entry 重跑即可，如 ka4.c→kname、ka5b.c→k2）
```

判据是**字节级**的，不是"能跑通"。任何未来的 Program 生成器（路线 A）
应当直接复用同一个比较函数。

---

## 8. 复现命令

```bash
# 供应商流水线剖面
sde64 -hsw -- mnclc --allow_pe_args --mnir --dump --dump-mnir --dump-mnbc \
    -e k -o x.bin gen14.c
# → k-x.mnir / k-x-opt.mnir / k-x.mnbc / k-x-opt.mnbc / x.bin

# MNIR 二进制 AST → Program，并与 x.bin 比对
sde64 -hsw -- ./mnircc k-x-opt.mnbc out.bin
cmp <(tail -c 49408 x.bin) <(tail -c 49408 out.bin) && echo IDENTICAL

# 一键回归
~/libmnrt/tools/mnir_probe/check_mnir.sh ~/libmnrt/mncl_samples/gen14.c
```

---

## 9. 未解项（不得当成已解决）

1. **Program 头部的完整 FlatBuffers schema 未反解**：字段偏移、fixup/
   arg 表的表项布局、`byte_site` 的编码。路线 A 的前置条件。安装树里
   只有 `mncl/mnir/mnir.fbs`（`namespace mncl.mnir.ast;`，`root_type MNIR;`，
   纯 AST），**没有** Program 的 schema（`translator_generated.h` /
   `flatbuffers/*` 也都不在）。已在 §6 A.1 拿到起点（root table / vtable
   位置、内核名字段、arg 数触发的两个字段），但**离"能自己写出逐字节
   相同的头"还差得远**：56 字节的六字段 table 各字段语义未定，fixup 表
   尚未定位。
2. **MNIR 文本解析器的两处 `ud2`**（`immu` + GRF0 目标）是供应商实现
   缺陷；不要围绕它设计任何流程，也不要假定下一版本已修。
3. **`mnrt` 的 provider 分层尚未落地**：`program` provider、异步事件、
   Program 常驻都还是设计，不是代码。
4. **`--dump-mnir` 的 `immu` 拼写** 是否在更新版本里改正，未知；本文
   所有结论基于 `mnclc` 2026-10-15 构建。
