# LLVM 产物 → MNIR → `mnrt` 动态 offload：接缝调查 (2026-11-04)

本文回答一个具体问题：**LLVM 后端已经产出 `.vsm` 之后，怎样把这些产物接进
MNIR / MN-Core 运行时，最终做到动态 offload。** 所有结论都在 `suzhou` 上
可复现；本文只记录**实测到的**事实与由事实推出的设计，不写"应该大概如此"。

---

## 0. 结论摘要（先看这一段）

1. **MNIR 是编译器 IR，不是运行时格式。** 设备唯一可装载的二进制是
   `mncore::Program`（64 位 FlatBuffers：前缀 + 内嵌的指令镜像 + 尾部实参区，
   schema 已完整反解，见 §6 A.1）。MNIR（文本或二进制 AST）只是生产 Program
   的一种输入。
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
| **`mncore::Program`** | `mnclc -o`、`Driver::CompileAll` | 设备 runtime（cl API） | 64 位 FlatBuffers：前缀 + 内嵌镜像 + 尾部实参区（§2 下文） |
| **GPFN app** | `assemble3 --binary` / `--app2` | `gpfn3_package_main`（仿真器） | `--binary` 在偏移 **64** 有 magic `"GPFN3"`；`--app2` 是另一版本，无该 magic |
| **裸 `.vsm` 文本** | `llc`（本后端） | `assemble3` | 每行一条指令 |

**Program 是"FlatBuffer 前缀 + 内嵌镜像 + 尾部实参区"的 64 位 FlatBuffers**
（2026-11-04 完整反解，schema 与证据见 §6 A.1）。文件固定分三段：

```
[0,          L)      FlatBuffer 前缀：root table → kernels 向量 → Kernel table
                     （内核名字符串、fixup 向量、空 strings 向量都在这一段）
[L,          L+blk)  Kernel::instblock 向量的数据 = 指令镜像（位打包）
[L+blk,      EOF)    最早构建的对象：arguments_type / arguments 向量与每个
                     实参的 union payload（FlatBuffers 反向构建，越早建地址越高）
```

`blk` 就是 instblock 长度 u64，位于 `L−8`；`L` 恒为 8 的倍数，`blk` 恒为
128 的倍数。旧口径的"头长" `hdr = len(file) − instblock = L + tail` 因此
**不是常量**：fixup 越多 `L` 越大，实参越多/越复杂 `tail` 越大。

| 样本 | fixup | args | blk | 文件 | L | tail = hdr − L |
|---|---|---|---|---|---|---|
| gen14 | 0 | 1 | 49408 | 49664 | 192 | 64 |
| gen28 | 1 | 2 | 256 | 512 | 192 | 64 |
| gen6 | 3 | 1 | 2176 | 2496 | 256 | 64 |
| gen26 | 3 | 2 | 768 | 1088 | 256 | 64 |
| gen10 | 4 | 2 | 1024 | 1344 | 256 | 64 |
| gen47 | 12 | 2 | 3072 | 3584 | 448 | 64 |
| gen16 | 6 | 7 | 1536 | 2048 | 320 | 192 |
| Ls1（2 指针 + 1 标量） | 5 | 3 | 1792 | 2304 | 320 | 192 |

`L` 由 FlatBuffer 的构建顺序与对齐决定（fixup 向量每项 24 B，实参越多
payload 越多），**没有闭式公式**——实测 0–1 fixup 都是 192，3 个 256，
12 个 448。要用就直接读 `L−8` 处的 u64（或按 §6 A.1 的顺序构建后 `Finish`）。

> **`gen16.bin` 不再是反例。** 旧文档把它记成"另一种布局"：它在偏移 184
> 有一串 `2,1 / 4,2 / 6,3 / 8,4`，而镜像长度像在 104。真相是它
> `L=320 / blk=1536 / tail=192`（7 个指针实参），`prog.py` 早先的头长
> 启发式在这里选错了候选。按 schema 解码 20/20 全通过，见 §6 A.1。

所以旧文档里"头长随 fixup 数走（0/1→256、3/4→320、12→512）"和
"`u64(hdr−72) == len(file) − hdr`（104/106 命中）"都只是 `tail == 64` 的
特例：那时 `hdr−72 = L−8` 恰好就是 instblock 的长度字段。实参 ≥ 3 或含标量
时 `tail` 变为 128/192，这两条经验规则随即失效——**不要**再把它们当契约，
用 §6 A.1 的 schema 或 `mnprog.py` 直接切分。

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

* 块编码已有，头部 schema 已反解（§6 A.1）⇒ **剩下的只是写构造器**：
  按 §6 A.1 的创建顺序用 64 位 FlatBuffers 语义把六字段 Kernel 填出来，
  再用 `mnprog.py` 与本文 §4 的字节判据回归。
* 头部内容有三处独立来源可交叉验证：`mnclc --dump` 的头部清单（`target` /
  `# of kernels` / `name` / `# of args` / `# of fixup` /
  `size of instblock` / `debug info`）、§2 的实测切分、以及供应商
  `CreateProgram`/`CreateKernel` 的导出符号（§6 A.1）。
* 验收即本文 §4 的等价判据：**对同一 kernel，头部之外必须与供应商的
  Program 逐字节一致**。

#### A.1 Program 头部：schema 已完整反解（2026-11-04）

**证据来自供应商自己的符号，不是字节猜谜。** `libmncl.so` 保留了 FlatBuffers
生成代码的符号，`nm -D | c++filt` 直接给出两个构造函数的**完整参数表**：

```
mncore::CreateProgram(FlatBufferBuilderImpl<false>&, mncore::Target,
                      Offset<Vector<Offset<mncore::Kernel>>>)

mncore::CreateKernel(FlatBufferBuilderImpl<false>&,
                     Offset<String>,                       // f0 name
                     Offset<Vector<Argument>>,             // f1 arguments_type
                     Offset<Vector<Offset<void>>>,         // f2 arguments (union)
                     Offset<Vector<Fixup const*>>,         // f3 fixups  (struct vector)
                     Offset<Vector<unsigned char>>,        // f4 instblock
                     Offset<Vector<Offset<String>>>)       // f5 strings
```

同库里还有 `CreateVectorOfStructs<Fixup>`、`CreateVectorOfStructs<Location>`、
`CreateVector<Argument>`、`data<Argument>`——struct/标量向量与偏移向量分得
很清楚，字段顺序即参数顺序。据此写出的 schema：

```fbs
table Program { target: Target = MNCore2; kernels: [Kernel]; }

table Kernel {
  name:           string;        // f0
  arguments_type: [Argument];    // f1  每个实参 1 字节
  arguments:      [void];        // f2  union payload，长度必须等于 f1
  fixups:         [Fixup];       // f3  24 B struct 向量
  instblock:      [ubyte];       // f4  指令镜像（位打包）
  strings:        [string];      // f5  实测恒为空向量
}

enum  Argument : ubyte { DRAM = 1, Scalar = 2 };   // 目前观测到的取值
struct Fixup { u64 bin_index; u64 addr_table_index; u64 storage_spec_index; }
```

**这是 64 位 FlatBuffers**：`uoffset_t`/`soffset_t` = 8 B、`voffset_t` = 2 B、
向量长度是 u64、struct 步长 8 对齐。判据不是"看起来像"：六字段 Kernel
table 的槽位偏移是 8/16/24/32/40/48、`table_size = 56`（32 位下这些会是
4/8/12/…、28），名字串的长度 u64 在字符前 8 字节。`Vector<Argument,
unsigned long>` 的 `size_type = unsigned long` 也印证了这一点。

**字段逐条证据**（全部可复现，脚本见 §7）：

| 字段 | 语义 | 证据 |
|---|---|---|
| Program.target | 目标机枚举，缺省 0 = MN-Core 2 | `--dump` 打 `target: MN-Core 2`；root vtable 的 f0 在所有样本里都**缺省**（即取默认值 0） |
| Kernel.name | 内核名 | `k`→`kk` 差分只改长度字节与第二个字符 |
| f1 arguments_type | 每实参 1 字节 | `DRAM i32*` → `01`；`i32/i16/i64/f32/f64` 标量 → `02`（见下） |
| f2 arguments | 每实参一个 union payload 表 | 见"payload 规则" |
| f3 fixups | 24 B struct 向量 | 4 个 fixup ⇒ 向量占 `8+96`，镜像长度字段正好后移 96 B；12 个 ⇒ 288 B |
| f4 instblock | 镜像本体 | 长度字段的值 = `--dump` 的 `size of instblock`，数据段逐字节 = 镜像 |
| f5 strings | 未用 | 116 个 `mncl_samples` + 全部实验核里长度恒为 0，且位置固定在 kernel table 之后 |

`Argument` 只有 1/2 两个值：把标量写成 `u8` 会被编译器挡下
（`mnir_backend.cc`：`Check failed: size == 2 || size == 4 || size == 8`），
说明该枚举描述的是"实参种类"而不是宽度——宽度在 payload 里。

**Fixup 的三个 u64**（24 B，无填充）。字段名直接取自 `libmncl.so` 里驱动器
自检字符串，语义由受控实验确定：

| 槽 | 名字（驱动器自检原文） | 实测语义 |
|---|---|---|
| 0 | `fixup.bin_index < inst_slots.size()` | 该重定位点所在指令在 **M 流水指令序列**里的序号（`nop` 填充与 `l2bmd`/`l1bmp` 等 L 流水指令不计入；`mvp`/`mvb`/`mvnop` 计入） |
| 1 | `fixup.addr_table_index < n_inout` | 实参下标（与 `mncore_write_pio32(…, 1u << mnaddr_table_index)` 的位号一致） |
| 2 | `fixup.storage_spec_index < storage_states_.size()` | 该 DRAM 实参内部的 **128 B 存储片下标** = 字节偏移 / 128 |

判定实验（每条只改一个变量）：

* 槽 1：`w1.c` 交换两个实参的使用顺序后，两个 fixup 的该槽从 `(0,0)` 变成
  `(1,0)`；`v2.c` 的 4 条 punicast 落在 `a,a+64,c,c+64` 上 → `(0,0,1,1)`。
* 槽 2：`v1.c` 的 `a+0/32/64/96`（i32，即 0/128/256/384 B）→ `0,1,2,3`；
  `v2.c` 的 `a+64`、`c+64`（256 B）→ `2`。**单位是 128 B，不是元素数。**
* 槽 0：`Ls1.c` 比 `v1.c` 多一段标量加载序言（`mvb` + `l2bmd` + `l1bmp`），
  第一条 punicast 的槽值从 `0` 变成 `6`，而 `l2bmd`/`l1bmp` 不计数时恰好
  是 6；`Ls3.c` 有两段标量序言，第一条 punicast 变成 `12`。同一文件里每条
  punicast 组（`mvp`+`mvnop`）固定 +2。

> 注意槽 0 **不是**镜像里的字节偏移，也不是 dump 的行号：`w1.c` 的两条
> `mvp` 在 dump 里是第 3/9 行、镜像里跨 256 B，而槽值是 `0,2`。

**payload 规则**（f2 的第 i 个元素）：

* `Argument::DRAM`（1）：**空 table**，`vtable {size=4, table_size=8}`，零字段。
  DRAM 地址由运行期 addr_table 给出，Program 里只声明"这个实参是 DRAM 指针"。
* `Argument::Scalar`（2）：`vtable {size=6, table_size=16}`，一个字段
  `f0 = Vector<StorageSpec>`，长度 1，struct 为
  `{kind=9, offset=0, size}`，`size ∈ {2,4,8}` 由实参宽度决定
  （`i16`→2、`i32/f32`→4、`i64/f64`→8；`u8` 被编译器拒绝）。

> **读表陷阱**：flatbuffers 的 `table_size` 把 **table 前的对齐填充**算进去，
> 所以同为"零字段空表"会看到 `{4,8}` 与 `{4,12}` 两种 vtable（差 4 B 填充）。
> 这不是两种类型。同理标量 payload 的 `{6,16}` 只是"8 B soffset + 8 B 字段"。

**布局与构建顺序**（要逐字节复现就必须照这个顺序）：

```
创建顺序（FlatBuffers 反向构建 ⇒ 早建的对象落在高地址，即文件尾部）：
  1. 每个实参的 payload table（按实参 0..n-1）
  2. arguments_type 向量（1 B/元素）
  3. arguments union 向量（8 B/元素）
  4. instblock 向量            ← 镜像，长度字段在 L−8
  5. fixups 向量（24 B/元素）
  6. 空 strings 向量
  7. Kernel table → kernels 向量 → root table
Finish 之后文件从 root 开始，于是得到 §2 的三段式布局。
```

由此 `L`、`blk`、`tail` 都是可算的，不再需要经验规则。尾部区
（`[L+blk,EOF)`）就是按上表顺序**最后建**的三样东西：`arguments` union
向量、`arguments_type` 向量、各实参 payload 表。实测 `tail` 只取
64 / 128 / 192 三种值：指针实参 0–2 个 → 64 B，3–5 个 → 128 B，6–8 个 →
192 B（gen16 的 7 个实参正好 192）；每含一个**标量**实参再加 64 B
（2 实参含 1 个 `f64` 标量 → 128 B；上表 Ls1 的 3 实参含 1 标量 → 192 B）。
它由这三样的实际大小逐项相加得到（向量 = 8 B 长度字段 + 数据，空 payload
表 8 B 或 12 B），**不必记档位**——解码时直接读 `L`/`blk` 求 `tail`。

**复现**（`tools/mnir_probe/mnprog.py`，见 §7）：

```bash
cd ~/hdrprobe && python3 -I mnprog.py --check *.bin
=== gen16.bin  size 2048
  Program  target=0 (field absent = default)  kernels=1
  Kernel   name='k'  n_args=7  n_fixup=6  n_strings=0
  image    L=320  blk=1536  tail=192  (file = L+blk+tail)
  args     [0]DRAM, [1]DRAM, [2]DRAM, [3]DRAM, [4]DRAM, [5]DRAM, [6]DRAM
  payload[0] empty vt{4,8} no fields (pointer arg)
  payload[1] empty vt{4,12} no fields (pointer arg)
  ...
  fixup[0] bin_index=0 addr_table_index=0 storage_spec_index=0
  fixup[1] bin_index=2 addr_table_index=1 storage_spec_index=0
  ...
  CHECK    OK
...
20/20 checked OK
```

`--check` 断言的是：`L+blk+tail == size`、`blk % 128 == 0`、`L`/`tail`
为 8 的倍数、`n_args == len(arguments)`、每个 tag ∈ {1,2}、标量 payload
必带 `StorageSpec` 而指针 payload 必须为空、每个 `addr_table_index < n_args`、
`storage_spec_index` 落在镜像范围内、`bin_index` 严格递增。**20/20 全过**
（覆盖 0/1/3/6/12 个 fixup、0–7 个实参、空镜像 `ke.bin`、以及此前被当成
"另一种布局"的 `gen16.bin`）。

**仍未定死的只有一处：`Target` 的非 0 取值。** `mnclc -t MNCore1/3` 直接崩溃
（栈回溯在编译器里），MNIR 路径还硬断言 `target == Target::MNCore2`，所以
除 0 以外的枚举值**没有任何可产出的样本**。这不是没查到，是这套工具链
产不出来——记在 §9，不要臆造。

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

**已进本仓库回归（本地，无需 mncl/sde64）**：

```
llvm/test/CodeGen/MNCore/Inputs/prog/{ke,s28,gen26,gen16}.bin   mnclc 产出的 Program
llvm/test/CodeGen/MNCore/scripts/mnprog.py                    解码器（parse/show/check）
llvm/test/CodeGen/MNCore/scripts/check_prog.py                逐字段钉住期望值
scripts/run_regression.sh 的 "Program container schema" 阶段  4 正 + 1 负（负例断言
                                                              错误期望必须被拒绝）
```

这一阶段是纯 Python 的，`--local` 就会跑，用来保证 §6 A.1 的 schema 结论
（`L`/`blk`/`tail`、实参 tag、fixup 三元组）在未来改动中不被悄悄改掉。
样本**不要就地重生成**：钉住的数字就是证据。

**需要 mncl 与 sde64 的样本级回归**（独立目标，不并入 `make check`）：

```
tools/mnir_probe/check_mnir.sh <kernel.c> [entry]   # 单样本
  PASS  镜像逐字节等价
  FAIL  mnclc 失败 / 无 .mnbc / 镜像不一致（打印首个差异字节）

tools/mnir_probe/sweep_mnir.sh [样本目录]            # 全样本，xargs -P 3
  结果写到 $MNIR_SWEEP_LOG（默认 ~/mnir_sweep_last.txt），末尾 PASS/FAIL/TOTAL
  注意：sweep 固定用 entry `k`，非 `k` 的核会显示为 FAIL（用 check_mnir.sh
  指定 entry 重跑即可，如 ka4.c→kname、ka5b.c→k2）

tools/mnir_probe/mnprog.py [--check] FILE...         # Program 容器解码/校验
  （与本仓库 scripts/mnprog.py 同一份；打印 target / kernel 名 / n_args /
   n_fixup / L / blk / tail / 实参种类 / payload / fixup 三元组，--check
   额外断言 §6 A.1 的不变量）
```

判据是**字节级**的，不是"能跑通"。任何未来的 Program 生成器（路线 A）
应当直接复用同一个比较函数；但它**先要过上面那条本地阶段**——
头部数字对不上时，逐字节比较是没有意义的。

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

1. **Program 头部 schema 已于 2026-11-04 完整反解**（§6 A.1）：`Program` /
   `Kernel` 六个字段、`Argument` 枚举、`Fixup` 三字段、`StorageSpec`
   布局、三段式文件布局、以及逐字节复现所需的创建顺序都已确定，
   `mnprog.py --check` 在 20/20 个样本上通过。**仍未定的只有一项**：
   `Target` 枚举除 `MNCore2`(=0) 以外的取值——`mnclc -t MNCore1/3` 崩溃、
   MNIR 路径硬断言只支持 MNCore2，这套工具链产不出样本。下一项工作是
   **写构造器**（按 A.1 的顺序生成头部），而不是继续反解。
2. **MNIR 文本解析器的两处 `ud2`**（`immu` + GRF0 目标）是供应商实现
   缺陷；不要围绕它设计任何流程，也不要假定下一版本已修。
3. **`mnrt` 的 provider 分层尚未落地**：`program` provider、异步事件、
   Program 常驻都还是设计，不是代码。
4. **`--dump-mnir` 的 `immu` 拼写** 是否在更新版本里改正，未知；本文
   所有结论基于 `mnclc` 2026-10-15 构建。
