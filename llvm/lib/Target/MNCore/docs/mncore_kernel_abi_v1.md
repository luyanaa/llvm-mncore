# MN-Core Kernel ABI v1 与 Runtime ABI 调查方案 (2026-11-03)

定位：`llvm-mncore` 是**窄而可靠的设备端 Machine Backend**；MLIR `mncore`
dialect 是唯一的设备语义与 Kernel ABI lowering 入口；PoCL/OpenMP/OpenACC/
PyTorch 各自保留 host 侧适配，但收敛到**统一 Kernel Descriptor + Device
Image + Launch API**。

---

## 1. 分层与职责

| 层 | 职责 | 归属 |
|---|---|---|
| 前端/编程模型 | Triton/TileLang/PoCL/OpenMP/OpenACC | 各项目 |
| 控制流与映射 | kernel outlining、loop 变换、work-item 映射、数据映射、predication | MLIR `mncore` legalization |
| Device ABI lowering | 参数绑定、buffer 映射、静态内存规划、DMA、资源需求 | MLIR + 共享 runtime 契约 |
| Machine Backend | intrinsics、ISel、Packetizer、Hazard、`.vsm` | `llvm-mncore` |
| Kernel Image | 指令流 + metadata（版本、参数、资源、完成语义） | assemble3 产物 + sidecar metadata |
| Host Runtime | 参数打包、传输、launch、wait、readback | `mnrt`（自建稳定 C ABI）+ providers |

后端**不**实现传统 C ABI、栈、递归、通用 `br`；但必须实现 ABI 的**物理层**
（参数落位、资源保留、入口/出口）与 `wait/fence/mask` 指令语义。

## 2. 控制流策略（后端边界）

| 类型 | 处理层 | 后端 |
|---|---|---|
| Host 控制流 | host | 不需要 |
| kernel 间循环/分支 | host runtime / command graph | 不需要 |
| kernel 内静态循环 | MLIR unroll/tile | 只接收展开后直线代码 |
| PE 条件执行 | MLIR if-conversion → mask | **需要** mask 指令语义 |
| 数据相关分支 | MLIR predication（受合法性限制） | 不需要通用 br |
| 动态循环 | 首版 host 分块；硬件机制作为后续 | 首版不支持 |
| kernel 内函数调用 | inline/specialize | 首版不支持，明确报错 |
| 跨 PE 同步 | 编译期规划 + `wait`/`fence` | **需要** |
| 递归/异常/动态栈 | 不支持 | 不需要 |

**Predication 合法性**：仅当副作用可证明等价时可 if-convert。禁止对 DMA、
同步、原子、可能越界访问、同资源并发使用盲目谓词化；MLIR verifier 必须拒绝
无法证明的 `scf.if`/`cf.cond_br`。

## 3. ABI 的四层定义

### 3.1 Framework ABI（各适配器实现）
OpenCL/OpenMP/OpenACC/PyTorch 的注册与调用方式；只依赖 `mnrt` 公共 C ABI。

### 3.2 Kernel Launch ABI（共享 `mnrt` + MLIR lowering）

```c
typedef struct mnrt_context mnrt_context;
typedef struct mnrt_program mnrt_program;
typedef struct mnrt_buffer  mnrt_buffer;
typedef struct mnrt_event   mnrt_event;

typedef struct { uint32_t struct_size, abi_major, abi_minor; uint64_t feature_bits; } mnrt_info;
typedef struct {
  uint64_t buffer_handle;   /* runtime 对象标识，非设备指针 */
  uint64_t byte_offset;     /* host 侧一律字节单位 */
  uint64_t byte_size;
  uint32_t access_flags;    /* read/write/readwrite */
  uint32_t reserved;
} mnrt_buffer_arg;

mnrt_status mnrt_get_info(mnrt_info *);
mnrt_status mnrt_create_context(const mnrt_info *, mnrt_context **);
mnrt_status mnrt_create_buffer(mnrt_context *, uint64_t size, uint32_t flags, mnrt_buffer **);
mnrt_status mnrt_load_program(mnrt_context *, const void *image, size_t size, mnrt_program **);
mnrt_status mnrt_bind_buffer(mnrt_program *, uint32_t arg_index, mnrt_buffer *, uint64_t byte_offset);
mnrt_status mnrt_bind_scalar(mnrt_program *, uint32_t arg_index, const void *data, uint32_t size);
mnrt_status mnrt_enqueue(mnrt_context *, mnrt_program *, mnrt_event **);
mnrt_status mnrt_wait(mnrt_event *);
mnrt_status mnrt_release_*();  /* context/program/buffer/event，所有权显式 */
```

原则：opaque handle；边界不传播 C++ 异常（错误码 + 查询 API）；所有权与
释放时序写进契约；**不得**把同步执行伪装成异步（provider 能力需显式上报）；
`mnrt` ABI 版本与 PFN SDK 版本分离。

### 3.3 Machine ABI（LLVM Backend 物理层）

* 专用 calling convention：`MNCore_Kernel`（实验性），入口无普通 C 参数栈。
* 入口形态（最终 CodeGen）：
  `define void @kernel() { ... llvm.mncore.* ... ret void }`
  外部参数由 launch preparation 绑定（descriptor + relocation），不暴露在
  函数签名中。
* 参数物理布局 v1（**必须冻结**）：
  - 向量参数 i → `space = 1 + i % 4`，地址 `16 * (i / 4)`（word，2 长语）；
  - 标量参数 j → GRF0 word `2 * j`（longword）；
  - 参数表由 kernel metadata 声明顺序与 kind；编译器输出实际落位。
* 当前不一致需修复：`LowerFormalArguments` 的标量落位（GRF0 `2j`）与
  Layer B 向量落位（轮转）由不同代码假定；v1 要求两者同源于 kernel
  descriptor，且由一次 lowering 生成 live-in，不得“假设已预装”。
* 资源保留：LM/L1BM/L2BM/matrix 占用、mask 寄存器可写条目、T 寄存器；
  由 metadata 报告，runtime 拒绝超限或不兼容设备。

### 3.4 Kernel Image ABI（版本化）

最小可行：**assemble3 产物 + sidecar metadata**（不立即自定义二进制容器）。

```yaml
abi: { name: mncore-kernel, version: 1 }
target: mncore2
sdk: { tool: assemble3, version: "0.9" }     # 版本入矩阵
entry: { symbol: gemm_f32, execution: static-vliw }
arguments:
  - { index: 0, name: A, kind: buffer, memory: DRAM, access: read,
      dtype: f32, rank: 2, byte_offset: 0, byte_size: 0, align: 64 }
  - { index: 1, name: B, kind: buffer, memory: DRAM, access: read }
  - { index: 2, name: C, kind: buffer, memory: DRAM, access: write }
  - { index: 3, name: n, kind: scalar, dtype: u32, slot: { space: GRF0, addr: 0 } }
launch: { policy: tiled, workgroup_mapping: compiler-defined }
resources: { grf: ..., lm: ..., l1bm: ..., l2bm: ..., matrix: ..., mask_entries: ... }
requirements: { dynamic_stack: false, device_calls: false, dynamic_branch: false }
```

* Host 侧 offset/size 用**字节**；设备侧按 word/longword 转换由 lowering 完成。
* Relocation：只在 loader 支持的措辞内绑定运行时 buffer；不直接改二进制地址。
* 兼容性：image 声明 SDK/assemble3 版本、设备代际、feature bits；不匹配拒绝。

## 4. 同步三层（不得混淆）

| 层 | 范围 | 实现 |
|---|---|---|
| VLIW Hazard | PE 数据竞争/端口 | 后端 Packetizer/Hazard |
| Kernel 内 | work-group、可见性、MV tag/wait | MLIR legalization + `wait`/`fence` |
| Kernel 间 | OpenCL event / OpenMP task / OpenACC async | host runtime（`mnrt_event`） |

`fence` 目前是“一条 NOP 步”的保守调度边界原型，**不是** barrier；
MV tag/wait 仅表示传输完成，不得未经证明等价于 work-group barrier。

## 5. Runtime ABI 调查方法论（PFN 栈）

**原则**：优先公开接口（MNCL → MLSDK → 内部库），先证明能力边界，再谈逆向；
所有观察限制在授权测试环境；原始日志不公开。

### 5.1 第一阶段：ELF/ABI inventory（被动）
对 `_torch_mncore*.so`、`_mncore_interface*.so`、`libpfvm_runtime.so*`、
`libpfcpp.so*`、`libfrontend.so*`：
`sha256sum`、`file`、`readelf -hW/-dW/--version-info`、`nm -D --defined-only
| c++filt`、`nm -D --undefined-only`。记录 SONAME/DT_NEEDED/RPATH、
GLIBCXX/CXXABI、`PyInit_*`。禁止凭 mangled 名直接用 `ctypes` 调 C++。

### 5.2 第二阶段：Python 静态追踪
在 `python_trainer/mncore` 中检索
`_mncore_interface|_torch_mncore|load_codegen|enqueue|synchronize|alloc|free`，
绘制 `Context → Program → Kernel → Buffer → Event` 所有权链（可能被隐藏/合并）。

### 5.3 第三阶段：最小动态实验（关键判定）
1. 仅创建 context；2. 加载一个已编译 kernel；3. 同一 kernel 重复执行 100 次；
4. **两个 kernel 在同一 context 中交替执行并共享 device buffer**。
最后一项是 eager 可行性的核心：若成立，障碍转向 ATen 覆盖与 launch 开销；
若不成立，记录失败边界（program loader / relocation / context ownership /
event queue / 设备独占锁），只对该边界做授权范围内的深入观察。
可先使用 MNCL 的公开 profiling（Perfetto）而非 uprobes/LD_AUDIT。

### 5.4 兼容性矩阵（必须记录）
- MNCL `clCreateProgramWithBinary` 是否接受**我们**的 assemble3 产物（裸
  `.vsm`/`.asm`/GPFNApp）——这是最优先的边界实验；
- GPFNApp 版本转换（`convert_gpfn_app_v0_to_v1`）；
- **assemble3/SDK 0.9 变更了 BAR 写入后的 hazard 检查**：后端 hazard 模型
  必须与 SDK 版本一起纳入测试矩阵（当前已知差异点）。

## 6. Eager 插件路线（PyTorch）

* 先实现 **Precompiled Operator Eager**：ATen op → 预编译 kernel registry →
  bind/enqueue；JIT operator 作为后续（模式 B）。
* 接入点：PrivateUse1 Dispatcher/DeviceGuard/Allocator + ATen kernels；
  但**先检查**既有 `_torch_mncore.so` 是否已注册 dispatcher/allocator，
  不能在另一 `.so` 中重复抢占 PrivateUse1。用隔离进程对比导入前后的
  dispatch table（`_C._dispatch_dump_table` 仅诊断用）。
* 最小 ATen 集合：`empty/empty_strided`、`copy_`、`add/mul`、`mm/bmm`、
  `relu`、`synchronize`、event；不支持的 op 必须显式报错或受控 CPU fallback，
  且 fallback 要真正完成同步。
* 版本风险：MLSDK 的 PyTorch 为特制版本；托管 ABI（ATen/CPython）与设备
  执行 ABI（`mnrt`）必须分离，前者变化不得传导到后者。

## 7. 交付物与验收（本项目）

1. **PFN Runtime ABI Inventory**：上述 ELF/Python/动态实验记录 + 能力矩阵。
2. **MNRT C ABI v1**：头文件、版本化结构、opaque handles、错误码；provider
   分离（`mnrt_mncl` / `mnrt_mlsdk` / `mnrt_emulator`）。
3. **Multi-Kernel Buffer-Reuse Regression**：同一 context 连续两个 kernel，
   第二个直接读第一个的 device buffer，数值对拍通过。
4. **ABI 兼容测试集**：Python import、native linking、create/destroy、
   multi-kernel、buffer reuse、并发 enqueue、event 生命周期、binary 兼容、
   设备代际、PyTorch 共存。
5. **非法 kernel 拒绝测试**：动态分支、越界静态资源、不可内联调用、非法
   barrier divergence、不可合法化动态寻址。

## 8. 与当前代码的衔接（按顺序）

1. 冻结 Machine ABI v1 参数布局；让 `LowerFormalArguments` 与 Layer B
   参数分配共用同一 descriptor（**当前不一致**）。
2. 增加 `MNCore_Kernel` calling convention 与入口 lowering（无参数签名 +
   显式 live-in 生成）。
3. `.vsm` 侧开始输出 kernel metadata sidecar（entry、args、resources）。
4. `mnrt` facade + emulator provider（先用 gpfn3 打通 inventory 与
   multi-kernel 实验），再接 MNCL provider。
5. 依据 §5.3 实验结果决定是否需要对 `libpfvm_runtime` 做更深的兼容性工作；
   不得先改二进制地址字段。

## 9. 风险与边界

* 同一进程内 MNCL 与 MLSDK 是否可同时占有设备、能否共享 buffer **未知**；
  无 export/import 时视为隔离内存域。
* assemble3/SDK 版本间 hazard 语义漂移（0.9 BAR 检查为已证实案例）。
* PrivateUse1 命名空间冲突与既有 PyTorch 扩展重复注册。
* GPFNApp/MNCL binary 与自产 `.vsm` 的兼容性未验证前，不得宣称可互换。
* 首版不支持的动态循环/设备分支/递归必须在 MLIR legalizer 明确拒绝。

## 10. mnrt v1 实现位置与状态 (2026-11-03)

* 位置：`ssh suzhou:~/libmnrt`（独立 git 仓库，commit `2e8b4e571`）。
* 组成：`include/mnrt.h`（ABI v1 头文件）、`src/mnrt_emulator.c`（provider）、
  `tests/selftest.c`、`tests/multikernel.c`、`Makefile`。
* 能力（`mnrt_get_info`）：abi 1.0，provider `emulator`，feature_bits=0。
* 语义：`mnrt_load_program` 接受 `.vsm` **文本**；`mnrt_enqueue` **同步**执行
  `assemble3 → gpfn3`；`mnrt_wait` 立即返回；无 C++ 异常；错误码 +
  `mnrt_last_error`。
* 已验证：selftest（imm 42 的 .vsm 完成 assemble+emulate）与 multikernel
  lifecycle（同 context 连续两个 program）均通过。
* **限制**：buffer/scalar 绑定已记录但尚未 relocation；emulator provider
  每个 program 启动新的 gpfn3 进程，**设备状态不跨 program 共享**，因此
  “multi-kernel buffer reuse” 需持久化 emulator session 或单镜像多入口；
  assemble3 以 checker 关闭方式调用（正确性验证仍由仓库回归负责）。
* **metadata sidecar v1 已实现**（行式格式，commit `b083944`）：
  `mnrt_load_program_ex` + `mnrt_program_arg_count/arg_info`；支持
  `arg <i> buffer <MEM> <access> <byte_off> <byte_size>` 与
  `arg <i> scalar <GRF0|GRF1|LM0|LM1|T> <addr> <bits>`，标量空间映射为 1..5。
  `tests/metadata.c` 验证解析、布局查询与 bind。
* MNCL 二进制兼容实验**已执行**（`suzhou:~/libmnrt/probe`）：
  - 头文件 `mncl/host/cl/cl.h` 把 `cl*` 映射到 `mncl*`；设备类型
    `CL_DEVICE_TYPE_EMU_MNCORE2`（0x40，设备名 `emu2`）。
  - 参考二进制：`mnclc` 编译的最小 kernel（256 B）→ create/build/kernel/
    enqueue/finish 全部成功。
  - **我们的 assemble3 `.asm` → `mnclCreateProgramWithBinary` 段错误**
    （无优雅错误返回）。
  - 结论：**MNCL program image ≠ assemble3 产物**。emulator provider
    （assemble3→gpfn3）是当前可运行路径；`mnrt_mncl` 只能加载 MNCL 自身
    格式（来自 mnclc），不能加载我们的 `.vsm/.asm`。若未来要让 MNCL 执行
    我们的 kernel，需要格式桥接或改用 emulator/session provider。
* 持久 emulator session（或单镜像多入口）仍待实现，以完成 buffer-reuse 回归。


## 11. MNCL loader 逆向初步 (2026-11-03, Ghidra)

**已确认**
* MNCL program image 是 **FlatBuffers `mncore::Program`**（Kernels/Arguments/
  Fixups/DebugInfo）。证据：libmncl 中 flatbuffers builder 对
  `mncore::Kernel/Argument/Fixup` 的符号；`libbackend_mnir.so` 导出
  `mncore::translator::MNIRBackend::GenerateKernel(FlatBufferBuilder*, bool)`
  @0x3c8d0。
* 指令打包器：`mncl::InstructionPacker` / `MakeInstructionPacker(mncore::Target)`
  （libinstruction_packer.so）；MNIRBackend 有 Pack/Unpack/PackBit，向
  `mncl::mnir::Builder` 发射。
* MNCL 的 emulator 执行由 **libmncl 内的 PFN runtime** 完成
  （`grape_pfn::codegen::runtime::GPFNDevice/GPFNExecuteCommand/GPFNEvent`），
  不是 gpfn3 CLI。
* 我们的 assemble3 `.asm` 不是 MNCL 镜像（loader 段错误）；mnclc 生成的
  FlatBuffers 镜像全链路通过。

**Ghidra 工作流与产物**：`suzhou:~/libmnrt/ghidra`（scripts + README +
命令模板，commit `bf85706`）。下一步反编译目标：
1. `MNIRBackend::GenerateKernel`（0x3c8d0，内部 0x3d310）→ Kernel 表字段与
   指令字节向量布局；
2. `InstructionPacker` 的 Pack 输入是 MNIR 值还是原始 `.vsm` 位流；
3. `mnclCreateProgramWithBinary` 的验证与 Program→Kernel 向量布局。

**桥接判断（已由 Ghidra 回答）**：`MNIRBackend::Pack(
std::vector<ValueHandle*>, DMode, BackendContext, bool)` 与
`GenerateKernel`（0x13c8d0，约 13.7 KB，suzhou:
~/libmnrt/ghidra/out/0013c8d0_GenerateKernel.c）表明打包输入是
**MNIR ValueHandle**；`GenerateKernel` 遍历 MNIRBackend 内部状态写
flatbuffer。因此 **`.vsm → MNCL Program` 不是简单封装**，需要 MNIR
emitter（schema + opcode map + value handles）或直接复用 MNIRBackend。
剩余逆向目标：Kernel 表的指令字节向量/fixup 字段布局，以及
InstructionPacker vtable 是否有接受原始字节的入口（用于字节级桥接）。

**第一手（gpfn3 emulator provider）状态**：mnrt v1 + metadata 已可用；
持久 emulator session / buffer-reuse 回归尚未实现（当前每 program 新进程，
设备状态不共享）。


## 12. P0 调查进展：MNCL 双 backend 与受控样本 (2026-11-03)

* `mnclc --help` 确认存在实验性 MNIR backend 选项：`--mnir`、
  `--dump-mnir`、`--dump-mnbc`、`-O size`、`--fma-disabled`（与 SDK 0.8/0.9
  更新记录一致）。**legacy/MNIR 两条构建路径在 CLI 层已确认存在。**
* 受控样本（`suzhou:~/libmnrt/mncl_samples`）：
  - `ke_legacy.bin` / `ke_mnir.bin`（空 kernel，默认 vs `--mnir`）：均 256 B，
    `cmp` 完全一致。
  - `kp7.bin` / `kp9.bin`（PE 私有数组写不同常量）：镜像相同——死存储被消除，
    暂无指令负载可供差分。
* 设备 C 的直接 DRAM/L1BM 存储被拒绝（"cannot put data to far memory"）；
  产生可观测副作用必须使用 `__builtin_l1bm_*` upload/broadcast/gather 等
  内建。因此下一步 A1/A2/A4 差分必须基于含这些内建的 kernel。
* 方法约束（采纳）：不做整文件裸 diff；先解析 FlatBuffers 结构树
  （root→vtable→field id/type/referenced object）再比较；字段结论分级
  Confirmed/Probable/Tentative。尚未找到 `.fbs/.bfbs` schema。
* Image Gate 尚未达成：需要“官方镜像 parse→rebuild→可加载”的 round-trip。


**P0 后续（同日）**：MNCL 设备侧内存模型为 LLVM address space：
PE=0（默认，普通局部量）、MAB=1、L1BM=2、L2BM=3、DRAM=5；分配器
`__builtin_{l1bm,l2bm,dram}_malloc`、`__builtin_mab_malloc_*`。构造
“计算 + upload”样本时，普通 C 数组作为 upload 源被翻译器拒绝：
部分初始化报 “Accessing blank data inside a region”，全部初始化报
`Check failed: v.is_handle() [bot, type:u64]`。说明 upload 内建要求其内部的
value-handle 形式，C 层指针猜测不足；需要先研究 `device/value.cc` 的
handle 规则或 SDK 随附的规范示例，再继续 A1–A6 差分。相关失败样本与记录见
`suzhou:~/libmnrt/mncl_samples`（该目录持续提交）。


**首个含指令镜像（同日）**：规范 upload kernel（PE 指针参数、完整初始化
8×i32、`__builtin_l1bm_malloc(16)`）以 `--allow_pe_args --mnir` 编译得到
1280 B 镜像，MNCL probe 全链路通过；同一源码用默认 backend 仅 256 B（与空
kernel 相同，说明默认/legacy 路径对该 kernel 未产生指令）。A1 差分
（V=7 vs V=9）只有 4 字节不同（偏移 264/272/279/305，0x07↔0x09 及
0x70↔0x90），暂定为两个立即数站点，**不做字段命名**，下一步用 FlatBuffers
结构树差分确认。约束：L1BM 分配 ≥16 元素；PE 源区域必须完全定义；
直接 DRAM/L1BM 存储不可用，必须走 `__builtin_l1bm_*`。


**A4/A5 与 L2-image 回归（同日）**
* A4（kernel 名 k→kname，主体不变）：镜像 1280 B，仅 5 字节不同；暂定为符号
  字符串+长度，待结构树确认。
* A5（两个 entry）：`mnclc` 多文件/换 `-e` 均只产出所选 entry（1280 B），
  多 entry 镜像需要其他生成路径——A5 仍未闭合。
* **L2-image 设备态复用回归已实现**：`tests/data/reuse.vsm`
  （`imm` → `nop` → `l1bmm@3` PE→L1BM → `nop/2` → `l1bmp` L1BM→另一 PE 文件
  → `d get`），mnrt 新增 `MNRT_DUMP`，`tests/reuse` 断言 `GREG1/v:0x39`
  通过。中间数据无 host 触碰。注意：若 `imm` 与 `l1bmm@` 同一步打包，
  gpfn3 读到旧值 0（与 §3.6.3.9 的表述存在差异），回归显式插入 nop 步；
  该同一步跨单元读写行为待进一步澄清。
* 未达成：跨 enqueue 的持久 session（gpfn3 批处理），故当前 L2 是
  “同一 run 内设备态复用（L2-image）”，不是 L2-session。


**镜像容器新发现（同日）**：MNIR 镜像不是单一 FlatBuffer，而是
**256 B FlatBuffers Program 头 + 追加的载荷段**（kb8=1280 B，空镜像=256 B）。
空镜像的 256 B 与含代码镜像的前 256 B 不同（10 个字节：113/121/185/232/
236/241-244/253），位置形态符合 flatbuffer 中的偏移/向量字段；载荷从偏移
256 开始（`0a000000` 开头），内容呈 **armeb 大端** 记录形态（0x00000070、
0x00000007 等），A1 的 0x07↔0x09 差异落在载荷内。含义：指令字节可能是
**可独立生成的追加段**，这对路线 A（Native Image Emitter）有利；但 256 B
头部如何引用载荷（偏移/长度/fixup 字段）仍待用 loader 访问器或结构树确认。
A5 仍未闭合：mnclc 只产出所选 entry。


**256 B 头部字段映射（暂定，两样本+kernel 名差分支持）**
头部按 u64 little-endian 字解释（非 FlatBuffers）：
* `@112` = file_size − 120（192 字节头+名的尾部偏移；kb8: 1280→0x488，
  k2u: 1792→0x688，空镜像 256→0x88）
* `@120` = file_size − 136（同上，空/代码一致）
* `@184` = **payload 字节数**（kb8: 0x400=1024；k2u: 0x600=1536；空: 0）
* `@152` = entry 名长度，`@160` 起为名字字符串（A4 `k`→`kname` 只改这两处）
* `@232/@240/@248` 在 1 vs 2 次 upload 下**完全相同**，暂视为固定
  trailer/描述字，不是 upload 计数
* payload 从 256 开始，A1（立即数 7→9）只改 payload 内 4 字节
  （0x07↔0x09 与相邻 0x70↔0x90），头部不变。
结论：头部可由“总长、payload 长、entry 名”推导；载荷是不透明代码段。
路线 A 的关键剩余问题：payload 是否为可由 assemble3 输出转换的指令字
（当前字节呈大端/非直观编码，需 A2/A6 样本与 loader 消费路径确认）。
A5 仍未闭合：mnclc 只产所选 entry。


**A2/A6 差分（同日）**：add/sub 与 upload lane 变体（均 2816 B）：
A2（add↔sub）仅 3 字节不同（0-based 296: 0x80↔0x84；807: 0x1f↔0x27；
1063: 0x27↔0x1f，后两者互为镜像），A6（upload0↔upload1）仅 2 字节不同
（2722: 0x30↔0xb0；2723: 0x00↔0x08）。头部 `@112=size−120`、
`@184=payload size`（0xa00）在三者一致。结论：payload 是规则指令编码，
opcode/传输切换只触碰小位域，支持路线 A 的可行性；但字边界及与
assemble3 位流的对应关系仍待映射。


**Image Gate 与决定性的 wrapper 实验（同日）**
* 256 B 头部模型已完成 parse→rebuild：空/单 upload/双 upload/A2 四个样本均
  **逐字节 round-trip 一致**，重建镜像经 MNCL probe 加载/执行成功
  （`suzhou:~/libmnrt/mncl_samples/rebuild.py`）。
* 决定性实验：把 assemble3 输出（`imm i"42"`，295 B）作为 payload 包进一个
  合法 256 B 头（`@112/@120/@184` 重算）→ loader 明确拒绝：
  **“Failed to parse kernel binary. Please check MNCL translator output.”**
* 结论：payload **不是** assemble3 输出，路线 A 需要真正的 payload 编码器
  （mnclc 打包格式），头封装不足。Instruction Gate 未达成；A5 未闭合。


**Payload 记录级解码（暂定）**
* payload 按 **16 字节记录**切分（无 store 的 kup=896 B；8 次 store 的 kb8=1024 B，
  差 128 B = 8×16 B；所有差分都落在记录边界）。
* A1（立即数 7→9）：rec0 的 byte7/15（0x70→0x90、0x07→0x09）、rec1 byte6
  （0x70→0x90）、rec3 byte0（0x07→0x09）。同一立即数以两种编码出现
  （0x07 与半字节移位 0x70），且被常量池化/重复（8 次 store 只有 3 个记录变化）。
* A2（add↔sub）：rec2 byte8（0x80→0x84，bit2）、rec34 byte7（0x1f→0x27）、
  rec50 byte7（0x27→0x1f，与 rec34 互为镜像）。
* A6（upload0↔1）：rec154 byte2..3（0x0030→0xb008）。
* 解释：记录 = 打包的指令/step 字，opcode 位在 rec2 bit2、镜像 opcode 字段在
  rec34/50、传输选择在 rec154、立即数在 rec0/1/3（半字节编码）。精确位布局与
  step 分组仍待确认；loader 对非 mnclc payload 明确拒绝。


**Payload 位级解码（暂定，多值验证）**
* 记录 16 字节；立即数/常量（C=1,7,8,16,291,4095 验证）：
  - rec0 bytes7..8 与 rec1 bytes6..7 的 16-bit LE 字段 = `(C & 0xFFF) << 4`
    （低 4 位为类型标签，这些整数为 0）；
  - 字节副本：rec0[15]=C&0xff、rec3[0]=C&0xff、rec1[0] 与 rec3[1] =
    (C>>8)&0xff。
* opcode 字节 rec2[8]（基值 0x80）：add 0x80、sub 0x84、and 0x94、
  or 0x98、xor 0x9c → 0x10 为逻辑组、0x04 为算术减法、逻辑组内
  and/or/xor = 0x04/0x08/0x0c。
* 算术镜像字段 rec34[7]/rec50[7] 仅 add↔sub 变化（0x1f↔0x27 交换）。
* L1BM 传输 lane：rec154 bytes2..3，upload0=0x0030、upload1=0xb008。
* **未解决**：参数基址/relocation 字段——在代码相同、仅 upload 源改为
  第二个 PE 参数时，payload 只差立即数（7↔8），看不到参数地址字段；
  需要“两参数各自做不同运算且都完整初始化”的合法样本继续。


**PE 参数寻址是“区域相对”的（重要）**
* 实验 1：两个 PE 参数都完整写入，仅把 upload 源从 x 换成 y —— payload 只差
  立即数（1↔2），无地址字段。
* 实验 2：`x[0..7]`+upload x 与 `x[4..11]`+upload x+4 —— payload **逐字节相同**。
* 结论（Probable）：PE 指针参数及其基址偏移**不进入 payload**；payload 只
  编码相对于已绑定区域的访问模式，真实 buffer 基址/偏移属于 kernel
  descriptor / runtime binding（与 MLSDK relocation、mnrt 的 host 绑定设计
  一致）。对桥接的含义：native payload emitter 无需知道参数地址，参数区分
  与重定位留在 descriptor 层。


**Payload = 128 字节块结构（重要）**
* 所有 payload 长度均为 128 B 的整数倍：kup 896=7、kb8 1024=8、add1 2560=20、
  add2 3200=25、a2add 2560=20。
* 每个 128 B 块（8×16 B 记录）在 rec0 byte0 有 0x0a 标记、rec1 bytes14..15
  有 0x0c00 标记；标记数恰等于块数。
* `--no-unroll` 产出逐字节相同的 payload（与循环展开无关）。
* 头部 `@232/@240/@248` 对所有非空镜像恒为 `0x700000000c / 0x3ff804700 /
  0x40000000000`（空镜像为 0）——是格式常量，不是计数。
* 块内：立即数在 rec0[7..8]/[15]、rec1[6..7]、rec3[0..1]；opcode 在 rec2[8]；
  add/sub 的镜像字段还出现在 payload 551（block4+39）与 807（block6+39），
  即后续块为其他 cycle/lane 携带副本。
* 解释：128 B 块 = 一个打包 step/指令组（4 cycle × 32 B），带块标记与
  每 cycle 的 operand/opcode 记录；step 分组与剩余位域是下一步。


**浮点常量编码（暂定）**：f32 与整数走不同路径。整数常量在 rec0/rec1 的
12-bit `(C<<4)` 字段 + rec0/3 字节副本；浮点常量这些字节为 0，改为修改
rec2 bytes7..9 的 24-bit 字段（LE）：1.0f=`FC 7F FE`、0.5f=`F8 7F FE`、
2.0f=`00 FE FE`、-1.0f=`FC 7F FF`（符号位在 byte9 bit0，FE↔FF）。
不是 IEEE 位型直接存放（1.0f=0x3F800000 不出现），位布局待定。


**浮点常量编码律（已用 18 个值验证）**：block0 rec2 bytes7..9 (b7,b8,b9)：
* `b7 = int8(floor(4*log2(|v|)) - 4)`
* `b8 = 0x7E if v==0 else (0x7F if b7<0 else 0xFE)`
* `b9 = 0xFF if v<0 else 0xFE`
验证：0、0.0625、0.125、0.25、0.5、1、1.25、1.5、1.75、2、3、4、6、8、
16、32、100、-1。例：1.0=`FC 7F FE`、0.5=`F8 7F FE`、2.0=`00 FE FE`、
-1.0=`FC 7F FF`、100.0=`16 FE FE`（floor 而非 round）。这是对数域常量格式，
非 IEEE 位。操作数/地址字段：单源平移会改变整体调度（payload 尺寸也变），
黑盒无法定位；需 loader 解码器（Ghidra xref 已启动，字符串
"Failed to parse kernel binary"@0xa7b6cb）。

---

## 13. Relocation 契约 v2（2026-11-03 冻结；编译期 2026-11-04 已实现，
runtime `mnrt` 消费端仍未实现）

§3.4 的 metadata sidecar 目前只记录 buffer 绑定，**没有 relocation**
（§10）。而 `.vsm` 的地址字段是静态立即数
（`packMVOperand`/`packPEOperand`），前端 kernel 的 buffer 指针无法直接进入
指令。本节冻结 v2 契约，使 MLIR/前端与 `mnrt` 可以并行实现。

### 13.1 两层单位

* host 侧（`mnrt_buffer_arg`、metadata）一律**字节**；
* 设备侧地址字段按操作数宽度：DRAM/PDM/L2BM 为 **longword（8 B）**，
  GRF/LM 为 **word（4 B）**。
* 转换由 lowering 完成：`addend_units = byte_offset / unit_bytes`；不整除即
  拒绝，不做隐式取整。

### 13.2 产生方式（编译期）——**已实现 (2026-11-04)**

带 buffer 基址的指令在 `.vsm` 中满足：

1. 地址字段填 **0**（占位）；
2. 该行以注释结尾：`# @mncore.fixup <site>`，`<site>` 在 kernel 内唯一；
3. sidecar 为每个 site 记录一条：

```yaml
fixups:
  - site: 7
    arg: 0                 # kernel 参数下标
    space: DRAM            # DRAM | PDM | L2BM
    unit: longword         # longword(8B) | word(4B)
    addend_units: 8        # 静态偏移 / unit
    width_bits: 32         # 地址字段宽度（避免越界写）
    opcode: mvp            # site 自检
    operand: 1             # MV: 0=params 1=src 2=dst; PE: 见下表
```

PE 操作数的 `operand` 位置按 `moncore_intrinsics_design.md` §3 的 intrinsic
签名计（例如 `alu` 的 `sx` 为 1）。

**实现路径（本仓库 → `.vsm`）**

| 层 | 表达 |
|---|---|
| `mncore` dialect | `mncore.dma` / `mncore.dma.store` 的 `src_arg` / `dr_arg`（`-1` = 绝对地址）。此时 `src_addr` / `dr_addr` 解释为**相对 buffer 的静态偏移**（longword）。verifier 拒绝 `src_arg >= 0` 且 `src_space != DRAM`，以及超范围的 arg。 |
| `convert-mncore-to-llvm` | 地址操作数改为 `mvop(..., 0, ...)`，指令 intrinsic 换成 `llvm.mncore.mv.reloc`，第 5 个参数是 `mncore::packReloc()` 打包的记录。冻结的 `llvm.mncore.mv` 4 参数 ABI 未改动。 |
| isel | `mv.reloc` 与 `mv` 走同一 opcode 选择，多出的记录作为 MI 的第 4 个操作数（MCInst 也会带上，但 asm string 不引用它，故不影响指令文本）。 |
| asm printer | 识别该操作数 → 行尾追加 ` # @mncore.fixup <site>`，site 按发射顺序从 0 递增；文件尾输出注释形式的 sidecar 块 `# @mncore.fixup.begin` … `# @mncore.fixup.end`。`assemble3` 忽略 `#` 注释，因此**同一条流**既是可汇编的 `.vsm`，也是带 sidecar 的产物。 |

记录打包（`llvm/include/llvm/Target/MNCore/MNCoreIntrinsicOps.h::
mncore::packReloc`，MLIR 与 asm printer 共用）：

```
[7:0] arg   [10:8] space   [11] unit   [19:12] width_bits
[23:20] operand   [31:24] opcode   [63:32] addend_units (signed)
```

`width_bits == 0` 表示“无 fixup”（真实地址字段至少 1 bit）。

参考消费端见 `../triton-mncore/third_party/mncore/tests/mncore_fixup.py`
（`split()` / `relocate()`）：解析 sidecar 注释块、按字节绑定换算、检查
unit 整除与 `width_bits` 越界、校验 `opcode`/`operand`/`space` 与行一致、
剥离注释后交给 assemble3。契约测试
`third_party/mncore/tests/run_reloc.py`（1 正例 + 7 负例 + 1 word-unit 正例）
与 `tests/run_e2e_add.py`（端到端用它做宿主侧重定位）；
`third_party/mncore/tests/run_all.py --remote` 把 R / E2E 一并纳入单一入口。


### 13.3 消费方式（loader）

1. 解析 sidecar，建立 `site → {arg, space, unit, addend, width, operand}`；
2. 对每个 buffer 参数取得设备基址（字节），换算为对应 unit；
3. 找到带 `# @mncore.fixup <site>` 的 `.vsm` 行，按 `opcode/operand`
   解码地址子字段，替换为 `base_units + addend_units`，并做 `width_bits`
   越界检查；
4. 可选：删除 fixup 注释后交给 assemble3。

**禁止**：在不带 site 的情况下按行号或字节偏移猜测地址位置；禁止把
`byte_offset` 直接写进设备地址字段。

### 13.4 与前端的关系

* Triton `tt.load`/`tt.store` 的指针参数、TileLang 的 `T.Buffer` 参数都只能
  通过本契约落地；
* 首版只支持 **整块 buffer base + 静态 offset**，不支持运行期 GEP
  重新指向；动态索引仍需 host 分块或展开；
* 同一 site 只允许绑定一个 arg；重复绑定、越界、未绑定 site 均应被 loader
  拒绝（fail closed）。

### 13.5 验收

* 双 kernel 共享同一 buffer：第二个 kernel 通过 site 读取第一个 kernel 写出
  的数据，数值对拍通过；——**待 `mnrt` 实现后验收**（参考消费端已就绪）。
* 负例：未绑定 site、`byte_offset % unit != 0`、地址超出 `width_bits`，
  三者都必须被拒绝。——**已由 `tests/run_reloc.py` 覆盖**，另加
  space/opcode/operand 不符、sidecar 声明但未被引用的 site 两类 fail-closed 负例。
* 端到端：`tests/run_e2e_add.py` 现在编译产出带 fixup 的 `.vsm`，由宿主侧
  `mncore_fixup.relocate()` 绑定 `arg0/1/2 → 0/128/256` longword 后交给
  assemble3；6 个被重定位操作数（2 个 load + 1 个 store，各两个 longword 平面）
  与 CPU golden 逐元素一致。


**Loader 解析器确认（Ghidra）**
* 报错字符串位于 `_createRelocatedKernelVsm`（VMA 0xaad950；Ghidra 基址
  VMA+0x100000）内的函数 FUN_00badd20；反编译 776 行，存于
  `suzhou:~/libmnrt/ghidra/out/00badd20_FUN_00badd20.c`。
* 解析事实：镜像开头为**头尺寸**（offset 0 的 ushort = 0x30），后续字段是
  相对偏移（与 `@112=size−120` 等模型一致）；payload 按**长度前缀的
  128 字节记录向量**消费（拷贝循环以 0x80 字节为块），与实测 128 B 块一致。
* 函数名 `_createRelocatedKernelVsm` 证实 relocation/buffer 绑定在此发生，
  与“PE 参数在 payload 中 region-relative、重定位在 descriptor 层”一致。
* 未解：128 字节记录的内部位布局（超出已实测的 opcode/imm 字节位置），
  以及记录是在此函数解释还是下游 GPFN runtime 解释——下一反编译目标。


**MNCL 逆向阶段状态（收束）**
已达成：256 B 头 parse→rebuild→load；payload=长度前缀的 128 B 记录向量
（loader 反编译确认 0x80 拷贝块）；整数 12-bit 立即数；浮点对数域常量律
（18 值验证）；opcode 字节；L1BM 传输 lane 字段；PE 参数 region-relative、
relocation 在 `_createRelocatedKernelVsm`。
未解：128 B 记录完整位布局（操作数/寄存器/地址）、记录到 VLIW step 的映射、
记录的下游解释位置（loader 仅拷贝；GPFN runtime 消费端未反编译）、A5 多
entry 生成路径。assemble3 作为 payload 的“头封装”路线已被 loader 明确拒绝。
下一阶段思路：反编译 128 B 记录向量的下游消费者；用 loader 作 oracle 做字段
变异；寻找官方多 entry 生产者；或接受 MNCL 为不透明 provider，继续 mnrt
emulator 路径（L2-image 已通过）。


**突破：`mnclc --dump` 是指令级 oracle**
`mnclc --allow_pe_args --mnir --dump` 打印头部 + 完整指令清单（immu/nop/
mvnop/ipassa/l1bmm@…），并给出 `size of instblock`（=payload 字节数）。
计数：kb8 24 指令/8 mvnop/1024 B；add1 60/20/2560；add2 75/25/3200；
a2add 60/20/2560。**每个 128 B 块恰含 3 条指令，最后一条恒为 mvnop**：
`[PE 或 nop][PE 或 nop][mvnop]`。这把实测块结构与官方指令流绑定。
首轮字段定位（kb8 block2 vs block3：`ipassa $r0 -> $n0v` vs `-> $n4v`）：
块内相对字节 6/13/21/39/40/41/46/47/54/59 变化，是 ipassa 目的地址/操作数
字段候选。下一步用一变量指令对（n0/n4/n8/n12、不同立即数、不同 MV）解出
位布局，再映射全部指令类型。


**首轮字段映射（dump 对照法，暂定）**
以 add1 的 block9–16 为例（ipassa 仅目的地址变化；ulpassa 源/目的同步变化）：
* ipassa `$r4 -> $sX`（X=4,5,8,9）：byte10/byte25 = `2*addr`；
  bytes17..18 LE16 = `addr<<5`；byte51 高半字节 = `addr&0xF`。
* ulpassa `$lr0->$lm0 / $ls0->$lm2 / $ls4->$lm4 / $ls8->$lm6`：
  byte7/byte22 = 源空间（0x00 GRF0 / 0x20 GRF1）；byte39 = 0x07/0x0f 镜像；
  byte9/byte24 = 源地址；byte16 = 源地址<<4；byte50 = 源地址>>1；
  byte15/byte48 = 目的 LM 地址；byte51 = 0x04 (LM0@0) / 0x05 (LM1)。
* 两个 nop 块逐字节相同（block2 vs block7 零差异），证明“块=3 指令”映射精确。
完整布局（全部指令、mvnop 槽、/1000 掩码字段）可用同一 oracle 方法继续解出。


**iadd 字段映射与方法收束**
add2 block0 `iadd $n0 $m0 -> $s0/1000` vs block4 `iadd $m4 $n4 -> $r0/1000`：
地址 4 出现在 byte6/15/21/48，`addr<<4` 在 byte7/22/55，其余
13/39/45/51/52/54/56/58 为操作数类别/空间位。结合 ipassa/ulpassa 的映射，
可看出每个 PE 操作数由**多个分量**（多种地址单位 + 空间/宽度/掩码标签）
分散编码在 128 字节块内。
**方法收束**：剩余工作已从“探索”变为“系统采集”——用 `--dump` 取指令文本，
配对仅一个操作数不同的块做字节差分，按操作数类别累积位图，最后实现
(指令,操作数)→128B 块 的编码器，并用 MNCL 加载 + `--dump` 对照验证。
廉价探索思路已用尽。


**块内分割确认**：`(nop,nop,mvnop)` 块在不同程序之间**逐字节相同**
（kb8 block1 vs add1 block2：0 差异）；所有含指令样本的变化字节都落在块内
前 60 字节，**60..127 恒定**。因此：
`128B 块 = [0..59：可变 PE 指令编码] + [60..127：nop;mvnop 固定尾部模板]`。
编译器样本中第二 PE 槽从未出现非 nop。含义：native emitter 只需编码前
60 字节，68 字节尾部可从任意合法镜像原样复制。


**修正与 Block Encoder v0（2026-11-04）**
* 修正表述：已验证的是“当前样本中仅 bytes 0..59 变化”，**未证明** packed
  块永久只有 60 B 可编码。128 B 作为完整逻辑块保留（PackMode + 两个 PE 槽 +
  MV 槽；AUTO_STRIDE 为三个 PE 槽，暂不支持）。观测到的恒定尾部
  （bytes 60..127，已验证子集 SHA-256 前 16 位 `f570986fa6c67b11`）仅作为
  带适用条件的模板，不是语义。
* Pack mode：`--dump` 显示每块 2 PE + 1 MV，符合手册 §1.2 的 **FLAT**
  （2 PE + 1 MV）；AUTO_STRIDE（3 PE + 1 MV）未观测、暂不支持。该对应关系
  是待确认假设（需 image metadata/loader 证明），不是已证模式。
* Encoder v0（`suzhou:~/libmnrt/encoder/block_encoder.py`）：PackedBlock
  （mode、PE 槽、MV 槽、trailer hash）、集中 FieldMap（int_imm/float_imm）、
  `Unsupported → fail closed`（非有限/超范围浮点、未知掩码、非 nop 第二 PE
  槽、auto-stride 全部拒绝，不产出镜像）。
* E0 结果：整数立即数 9（模板 7）与 291、浮点 2.0（模板 1.0）均与官方镜像
  **逐字节一致**；inf 被 fail-closed。
* 下一梯级：E1 官方 loader 接受自产镜像 → E2 `--dump` 语义还原 → E3 模拟器
  数值 → E4 硬件。`/1000` 掩码仍 unsupported（未编码前禁止输出）；浮点边界
  （subnormal/Inf/NaN/舍入）待采样；Relocation v2 保持语义级，EncodedFixup
  内部另加、不改已冻结契约。


**E0/E1 覆盖扩展（2026-11-04）**
* `block_encoder.py` 新增 `ipassa_dst`（目的地址）与 `ulpassa_src_grf1` /
  `ulpassa_dst_lm1`（源空间/源地址、目的地址）字段编码器；全部字段集中在
  FieldMap，未知情况一律 `Unsupported`（fail closed）。
* E0（与官方镜像逐字节一致）矩阵：
  - `immu` 整数立即数：9（模板 7）、291 —— PASS
  - `immu` 浮点立即数：2.0（模板 1.0）—— PASS；inf → fail-closed
  - `ipassa $r4 -> $sX`：X=5/8/9 —— PASS
  - `ulpassa`：（ls0→lm2）、（ls4→lm4）、（ls8→lm6）—— PASS
  - 编码规则修正：byte7/22 的源地址项是 **算术加** `0x20 + ((addr>>1)<<4)`，
    不是按位或（s4 会与 0x20 基位碰撞）。
* E1：用编码器把 kb8 的 block0 改写为 `immu i"42"` 构造自产镜像，官方 loader
  （`clCreateProgramWithBinary → build → kernel → enqueue → finish`）**接受并执行**
  （`PROBE OK`）。
* E2/E3 限制：`mnclc --dump` 只对 C 源码，不能反汇编既有镜像；MNCL native
  镜像亦不能交给 gpfn3（格式不同）。官方镜像→指令文本/数值的 oracle 目前缺失，
  E2/E3 需自写解码器或依赖未来工具。
* 浮点对数律仍按“已验证子集”处理（有限/范围内），边界（subnormal/Inf/NaN/
  舍入）未采样；不将其提升为最终编码模型。


**E0–E4 梯级状态（2026-11-04）**
* E0（逐字节一致，已验证子集）：immu 整数 9/291、immu 浮点 2.0、
  ipassa $r4->$s5/$s8/$s9、ulpassa (ls0->lm2)/(ls4->lm4)/(ls8->lm6)。
  子集外 fail-closed。
* E1：编码器改写 kb8 block0 为 `immu i"42"` 的自产镜像被官方 loader
  接受并执行（PROBE OK）。
* E2：自写解码器（字节→规范文本）与官方 `--dump` 对应行逐条一致
  （immu/ipassa/ulpassa，9 项 PASS）。注意 `--dump` 只能从 C 源生成，
  官方 image→text 转储工具缺失，E2 以官方文本作同流对照。
* E3：数值 kernel PE 填 7 → L1BM → L2BM → DRAM；镜像 `#args=1`、
  `#fixup=1`（DRAM 指针重定位点）；host 绑定 cl_mem、enqueue、readback：
  `out[0]=0x0000000700000007`，**E3 PASS**（官方 MNCL 模拟器路径）。
* E4：**本环境不可完成**——suzhou 仅有 emu2 模拟设备，无 MN-Core 2 硬件/
  驱动。需真实设备 + 官方驱动，使用同一镜像。


## 13. E0 完全覆盖与 llvm + assemble3 接入设计 (2026-11-04)

**已核实的事实**
* `assemble3` 输出是文本位串：`i`/`j` 行为 **292-bit PE 指令**（逐字符 0/1，
  与手册 PE=292 bit 一致），`m` 行为 **82-bit MV 指令**（实测）。
* MNCL payload 是 128 B 块（PE0+PE1+MV）的**重新序列化**：把同一条
  `l1bmm@0 $ln0v $lb0` 的 292-bit 位串在 MNCL 块/整个 payload 中做逐位搜索
  （MSB/LSB/反转）**均无命中** → 不能直接嵌入，必须字段级转码。

**E0 完全覆盖的操作性定义**（由 `encoder/canonical.py` 强制）
* 对后端闭环内核实际发射的每条指令，必须满足以下之一：
  1. 有 mnclc 可表达的等价指令 → **E0** 逐字节一致；
  2. mnclc 无法表达 → 用 **292-bit 解码交叉验证 + E1/E2/E3**；
  3. 属于 emulator-only 控制（`d get/set`）→ `IMAGE-EXCLUDED`，不进入镜像。
* 任何未覆盖项 **fail closed**（不产出镜像）。

**覆盖矩阵（按计划优先级）**
| 类 | 指令 | 现状 | oracle |
|---|---|---|---|
| 1 | nop / immu(int,float) / ipassa / ulpassa | E0 已过 | mnclc E0 + dump E2 |
| 1 | iadd、其余 ALU | 字段图部分 | mnclc E0 + dump E2 |
| 2 | l1bmp / l1bmm / l1bmm@ / l1bmd | 未编码 | mnclc E0 + E1/E3 |
| 3 | fvfma/fvmul/fvadd | 未编码 | mnclc E0 + E3 |
| 4 | mvp / l2bm* / wait | 未编码 | mnclc E0 + E3 |
| 5 | mwrite/mread/matvec | 未编码 | mnclc E0 + E3 |
| 6 | 双 PE 槽 / 掩码 `/(...)` / auto-stride | 不支持（fail closed） | 需 mnclc 样本或 292-bit 解码 |

**llvm + assemble3 接入选项**
* **A（推荐，长期主线）**：`llc .vsm` → Canonical IR → 128 B Block Encoder。
  不依赖 assemble3 的 GPFN 应用格式；assemble3 保留为**合法性 oracle**
  （每类编码产物仍可回灌 assemble3 校验）。
* **B（验证/替代输入）**：`assemble3` 的 `i/j/m` 位串 → 292/82-bit 字段解码
  → 同一 Canonical IR → 同一编码器。优点：以官方汇编器的位串为准；
  起点是对同一指令比较“文本派生的 canonical”与“位串解码的 canonical”。
* **C（不推荐）**：GPFN 应用 → MNCL 镜像转码。已在实测中证明裸 assemble3
  文本被 loader 拒绝，GPFN 应用格式未文档化，风险最高。

**Canonical IR（共享接口）**
```
PackedBlock { PackMode mode; PEInstr pe[2|3]; MVInstr mv; Fixup[] fixups; }
PEInstr     { opcode; prec; MaskSpec mask; Operand[] operands; }
MVInstr     { opcode; params; Operand src, dst; }
```
字段编码集中在 FieldMap；编码器与解码器共用同一表；Relocation 保持语义级
（EncodedFixup 内部扩展）。每个 golden 组同时固定 assemble3/SDK/pack mode/
设备版本（SDK 0.9 已修改 BAR hazard 检查）。

**验收**
* `encoder/canonical.py <kernel.vsm>` 对后端内核语料必须 **全部 OK**；
* 每类至少 E0 + E1 + E3 证据（mnclc 可表达时），并有 E2 dump 对照；
* E4 仍需真实设备（本环境仅 emu2）。


**E0 覆盖实测（后端语料，2026-11-04 最终重分类，post-gen57）**
`encoder/canonical.py` 对语料（122 个 `.vsm`，已固化到
`encoder/corpus/`）逐指令分类，两个指标：
* **无 UNKNOWN：122/122 文件** —— 每条指令要么是 E0（模板/精确文本/映射），
  要么是**有据记录的 `UNSUPPORTED`**（见下表）。"可达类已补齐"指这一项。
* **严格 FILE-E0：47 个文件** —— 每条指令都落在 E0 模板族内；
  其余 75 个文件的缺口全部来自下表的有据不可达类，不再是"待办"，
  而是"该 oracle 无法表达"。
* 模板表 **588 条**（gen5 → gen58：118 → 487 → 588）。488 之后新增的 102 条
  来自 60 份 dump 的全量重导（见下"模板溯源"），不是新的探针类。
* 回归：`~/libmnrt` 的 `make check` 跑三个脚本：
  `encoder/block_encoder.py`（323/323 块逐字节重建 + immu 律）、
  `encoder/regress_coverage.py`（钉住模板数、两个文件计数、每个不可达类的理由、
  fail-closed 与正负夹具、逐字节可编码率下限）、`encoder/regress_templates.py`
  （模板溯源，见下）。

**必须区分的两个"覆盖"（2026-11-04 实测，避免误读）**
语料 4728 条指令中 3724 条是 `IMAGE-EXCLUDED`（emulator-only `d get/set`），
真实指令 **1004** 条，其构成为：
| 口径 | 数量 | 含义 |
|---|---|---|
| 精确模板（逐字节可编码） | **88**（8.8%） | `block_encoder.py` 可查表直接出块 |
| 类级 E0（opcode 已验证、文本未验证） | **832** | 该类至少有一条模板过 E0，但此寄存器/立即数组合未验证；`block_encoder` 对其 **fail closed** |
| `UNSUPPORTED`（有据不可达） | **84** | 见下表 |

因此"E0 完全覆盖 / 122-122 无 UNKNOWN"是**类级**结论，**不是**语料的逐字节可编码率。
要让 llvm 产物直接可编码，还需二选一：补全文本变体（更多探针），或按类做
**字段归纳**（用同类多条模板反解寄存器/立即数位域，再用全部模板做重建校验）。
`regress_coverage.py` 现把 88 作为**下限**钉住，后续每次合并都必须让它单调上升。

**本轮（post-gen33 → post-gen57）新增模板 62 条**，覆盖
`fmin $ln0v $aluf -> $nowrite`、`fmax $ln0v $lm0v -> $nowrite; l1bmr_ffadd ...`、
`l1bmr_fmax`、`fmin/fmax $lr0 $ls0 -> $ln0/1100`（scalar 经单向分派）、
`dmin $ln0v $aluf -> $lm8v`、`dmax $ln0v $lm0v -> $ls0v`、
`lpassa $lsN -> $lrN/1000`、`l1bmm@0..7 $ln0v -> $lb<4N>`（全部 8 个变体）、
`l2bmd $lc0 -> $lb0` 与 `l2bmd $lb0 -> $lc0`（双向）、
`l1bmd+13/+14/+15`。

**对上一轮结论的更正（重要，均为探针污染所致）**
上一版把 `l2bmd`、`l1bmm@`、`lpassa`、向量 `fmin/fmax` 记为"不可达"，**这是错的**：
* `l2bmd` / `l1bmm@` 当时被判为"LM0 保留"，真实原因是探针里写了
  `out[0] = buf[0]`（PE 局部数据直存 DRAM）。mnclc 对此报
  `cannot put data to far memory 5`，该错误**连带**吞掉了整个内核的翻译，
  使我们先看到 LM0 保留的次生错误。去掉该直存后，gen53 一次捕获
  `l1bmm@0..7` 全部 8 条，gen55 一次捕获 `l2bmd` 双向。
* `lpassa`、向量 `fmin/fmax` 可达：前者来自 `__builtin_l1bm_freduce_*`
  的累加路径；后者来自 `__builtin_ffminx8/ffmaxx8/ffminx4/ffmaxx4/dfminx4`
  （注意真实名字是**双 f 前缀** `ff...`，不是 `fminx8`；此外 `__builtin_*`
  的 min/max 家族挂在 `__MNCL_DECLARE_PE_BUILTIN_1_FLOAT` 下）。
  教训：探针里任何"PE→DRAM 直存"都会污染整轮编译，必须改用
  L1BM/L2BM/DRAM 转移链或完全不加汇点。
* 上一轮（post-gen5 → post-gen33）新增的 307 条模板，覆盖
  `l1bmr_ffadd/fmax/fmin`、`l2bmr_ffadd/fmax/fmin`、`mvr_ffadd/fmax/fmin`、
  `fvadd/fvmul/fvfma`、`dvadd/dvmulu/dvmuld/dvfmau/dvfmad`、
  `hvmul/hvadd/hvfma/hvpassa`、`dmwrite/dmread/dmmulu/dmfmau`、
  `mvp/n1024i01`、`mvp/n64i01`、`nop; wait i01`、
  `l1bmm4@N`、`l2bmb@N/0`、`l2bmb2`、`l2bmr2_*`、`l2bmi@0/5`、
  `dmax/dmin`、`hbfn/8`、`hbfe/6`、`msl/msr`、`l1bmd`/`l1bmd+N`、
  `mvb2/mvb4/mvr2dfadd`、`l2bm@N`、`dbfn`、`sinc/sdec/linc/ldec`、
  及 gen14 附带的大量 `ilsl/ior/uipassa/uilsr` 寄存器初始化序列。

**已证伪的"不可达"类及证据（每一类都有可复现的关闭理由，
问题不再是"尚未尝试"，而是"该 oracle 无法表达"）**

| 类（语料命中数） | 关闭该类的证据 |
|---|---|
| `mvp/n<N>i02..i06`（43） | tag 由 mnclc 调度器分配。C 表层**无 tag 参数**（129 条 `__builtin_l2b*` 与全部 MV builtin 逐一枚举，头文件无 wait/tag/async 声明）；12 路并发转移的压力探针 gen47 仍然逐条发射 `nop; wait i01`，即 91 份 dump 中只出现 `i01`。 |
| `wait`＞i01（2） | 同上。附加证据：wait 块的 **tail 前 8 字节编码它所等待的那条转移的地址**（gen47：`...1022`/`1044`/`1066`/… 随前一 mvp 地址单调变化），因此该块**不是 PE 文本的函数**，仅靠模板无法生成。 |
| `l2bmdars`、`l2bmdarw`（12） | mnclc 自身在发射前中止：`mnir_backend.cc:3510 unimplemented`，栈顶 `MNIRBackend::L2BMDARSend()`（gen56/gen57 可复现栈回溯）。 |
| `l2bmb@<addr>/<immode≠0>`（4） | `__builtin_l2bm_unicast0..7` 只映射到 `l2bmb@N/0`；immode 仅由 `__builtin_l2bmi_*` 暴露（0..6，头文件明示 7 不提供）。 |
| `/0111`-masked（7） | 头文件只有一条 mask 相关声明 `__builtin_debug_store_current_mask`，且注明"debug helper, no assembler instruction"：C 表层无法设置指令掩码。 |
| `maskrs`、`mask`、`quit`（7） | 91 份 mnclc dump 中一次都没出现；内核终止与掩码恢复是镜像/尾块级，不在指令流内。 |
| `mvd/n64`、`mvp/n256nd4`（6） | PDM 路径：`PDM builtins are supported only by the MN-Core 2 L1IR backend`；mnclc 的 `--help`（全文 17 行，已逐行核对）没有后端选择开关。 |

以上全部由 `canonical.py` 的 `UNREACHABLE` 表逐条钉住，`regress_coverage.py`
对每一类跑负向夹具（必须报 `UNSUPPORTED` 且带理由），并用一条虚构 opcode
（`zzq_fake`）做 fail-closed 正控——它必须报 `UNKNOWN`，防止分类器"一律放行"。
* **VSM→MNIR 映射**：canonical.py 新增 `vsm_to_mnir_text()`，
  处理 `l1bmm@N $src $dst` → `l1bmm@N $src -> $dst`、
  `mvp/nNNNiNN $src $dst` → `mvp/nNNNiNN $src -> $dst`、
  `l2bmb@[a,b]` → `l2bmb2`、`l2bmb@N/M` → `l2bmb@N/M`、
  `l2bm@N` → `l2bm@N`，
  及 `VSM_TO_MNIR` opcode 表（reduce/dv/hv/dm/fv/l1bmm4/l2bmi/dmax/hbfn 家族）。
* **分类器归一化漂移修复**：`_template_texts()` 原先用一张手写的前缀折叠表，
  而 `opcode_of()` 用另一套规则，两边不一致导致 `l2bmd`(29) 与 `l1bmm@`(21)
  在**已有精确模板**的情况下仍被判 UNKNOWN（`l2bmd` 被折叠成 `l2bm`，
  `l1bmm@` 被折叠成 `l1bmm`）。现改为
  `{opcode_of(t) for t in TEMPLATES}`，两侧由同一个归一化函数导出，
  结构上不可能再漂移。仅此一处修复就让"无 UNKNOWN"文件数从 37 跳到 47。

**模板溯源与错位回滚（2026-11-04 收尾，重要）**
模板表此前是"逐轮合并"的累积物，没有任何机制能证明它与 dump 一致。收尾轮做了
一次全量重导，发现并修正了一类**结构性错误**：

* **镜像头长度不是常量**，必须由算术导出：`hdr = len(image) - 128*nblocks`。
  实测 60 份镜像里 `256`（55 份）、`320`（gen6 / gen10 / gen26）、`512`（gen47）。
  早期提取器把 256 写死，对 320 头的镜像**每个块提前 64 B**，切出的"头"实际跨越
  了相邻块边界。
* **错位可机械判别**：正常块在块内偏移 **104** 有固定尾标 `0c00000070000000004780ff03`
  （= `TAIL_BODY` 的固定段）；wait/nop 类块在同一 24 B 常量处偏移 **30**。
  提前 64 B 的块会把这个标记推到偏移 **40**——一个合法块永远不会出现的位置。
* 据此逐条核对 487 条模板，命中 **18 条**由 gen6/gen10/gen26 错位提取而来的条目
  （含 `nop; wait i01`、`mvp/n64i01 …`、`mvb2/mvb4 …`、`l1bmr_*`/`l2bmr_*`、
  `mvr_*`、以及一条 `zero -> $ln0v [first-block]` 伪条目）。全部按**对齐样本**
  重建（每条都有唯一对齐来源，无歧义），伪条目删除。
* 同时把 60 份 dump 全部重导并合并：**+102 条**此前从未合并的真实模板，
  与已有模板**零冲突**（这本身就是一次跨批次一致性校验）。
* 新增 `encoder/oracle/`（60 份 dump，2253 块，916 KB）与
  `encoder/regress_templates.py`，把"模板表必须能由 oracle 重导"变成回归不变量：
  S1 除末块外每块尾标在 104/30、每镜像至多一个无标记块且必为末块；
  S2 同一文本 → 同一 60 B 头（60 份 dump 全量，无例外）；
  S3 纯 nop 头唯一且等于 `TEMPLATES['nop']`；
  S4/S5 oracle 与模板双向一致；S6 计数钉死（588）。
  这一类错误今后不可能再静默发生。
* **复现性已验证**：`kb8 / gen30 / gen14 / gen39` 用同一 mnclc
  （`/opt/pfn/pfcomp/hpc/mncl/bin/mnclc`，2026-10-15）与同一组 flag 重编，
  **逐字节一致**；且默认（L1IR）后端输出与两者都不同（2304 B vs 7936 B），
  故 oracle 全集确为 `--mnir` 产物。

**编码器边界的新发现（供下一阶段 E1 使用）**
块级布局：`[0..59]` 指令头 + `[60..127]` 尾区；正常块尾区 = 44 B 零 +
`0c00000070000000004780ff030000000000000000040000`（24 B 固定段，其起点即块内偏移
104）。`block_encoder.py` 目前一律产出 `模板头 + TAIL_BODY`，两个边界**尚未覆盖**：

1. **带 wait 的块**：尾区前 8 B 编码被等待转移的地址，随上下文变化
   （gen47 的 wait 块各不相同），所以 `nop; wait i01` 虽有模板头，
   尾部不能取常量 `TAIL_BODY`。
2. **末块尾区只有实测的三态，且与"是否纯填充"无关**（此前的表述已被证伪）：
   60 份镜像中末块尾区为
   * **33 份**：固定的非零终止记录
     `…0001 000000 18000000 00000000 01000000 00000000 01000000 04080004…`
     （其中 32 份是无尾标标记的末块，另一份是 gen6——末块为 wait 块但尾区取此记录）；
   * **20 份**：全零（`TAIL_FINAL`）；
   * **4 份**（gen10 / gen26 / gen28 / gen47）：末块是普通 wait 块，尾区为
     `…0002 000000 30… 18…` 的等待描述符。
   末块**可以载真实指令而尾区全零**（`l1bmm@0`、`fftoi`、`l1bmm4@3`、`uipassa`…），
   也**可以载同一指令而尾区非零**：kb8 与 gen30 的末块 PE 文本与 60 B 头
   **逐字节相同**，仅尾区不同。因此"末块尾标由末块文本决定"不成立。
   选择函数**尚未归因**。已排除的候选（均有受控实验）：
   * 末块是否为转移：6 个新内核都以一次 L1BM upload 结尾 → 全零，但 gen30
     （末块 `l1bmm@0`）非零；
   * 指令家族：`lzerox4+lincx4` 无 upload 时非零，加一次 upload 后全零；
     而 `izero+iinc+idec` + 2 次 upload（gen30 去 i64 变体）仍非零；
   * 镜像规模/块数：全零组最大 39 块，但非零组含 1..3 块的 gen12/gen32/gen38。
   已纪实的结论只有：**尾区取值是镜像级上下文，不是块文本的函数**，E1 之前必须
   由 oracle 匹配或由调度器语义决定，不能取常量。
   另外：**55 份镜像中恰有一个无标记块，且它在 55/55 里都是末块**；其余 5 份
   （以 wait 块结尾）没有无标记块。这一点已由 `regress_templates.py` 的 S1 钉住。

现有 selftest 的 323 个样本恰好都不含这两类，所以 323/323 仍成立；
它们在 E1（loader 接受）之前必须按样本补齐，属于**已知未闭合项**，
不得据此宣称镜像已可装载。

早期按 6 测试组的手工清单（alu/mau/l1bml2bm/mv/mask/imm-fixed）已由模板表驱动
的自动分类取代。**“E0 完全覆盖”验收标准不变**：对后端语料
（排除 emulator-only `d get/set`）`canonical.py` 不得报告任何 `UNKNOWN`
——该条件现已在 122/122 文件上达成。


**assemble3 292-bit 解码启动（2026-11-04）**
`encoder/bits292.py`：用 assemble3 的单操作数差分确定并验证了首个指令类的
292-bit 字段切片（全部 8 个样本 PASS）：
* opcode：bits [216:218]（b216 为 MSB），`l1bmm@`=1、`l1bmm4@`=2；
* 目的 PE 地址：地址位 k → bit `153-k`（k=0..4）；
* MAB 地址：地址位 k → bit `240-k`（k=0..3；bit 241 为常量 1）；
* L1BM 地址：地址位 k → bit `233-k`（k=0..12）。
该结果证明 **assemble3 的 `i/j` 位串可按字段解码**，接入路径 B（292-bit
解码 → Canonical IR → 128 B Block Encoder）可行；下一步按同一差分法扩展
opcode 字段宽度与其余操作数类别（ALU/MAU/MV/L2BM/掩码），并与 E0 的
mnclc 侧字段图在 Canonical IR 汇合。


## 14. MNIR 接缝与 Program 容器（2026-11-04）

§13 记录的"镜像头长 256 / 320 / 512 三态"不是随机噪声，而是
**FlatBuffers 头部随 fixup 表增长**。用
`hdr = len(image) − size of instblock` 对同一批样本重算：

| 样本 | fixup | args | instblock | 文件 | 头长 |
|---|---|---|---|---|---|
| gen28 | 1 | 2 | 256 | 512 | 256 |
| gen6 | 3 | 1 | 2176 | 2496 | 320 |
| gen26 | 3 | 2 | 768 | 1088 | 320 |
| gen10 | 4 | 2 | 1024 | 1344 | 320 |
| gen47 | 12 | 2 | 3072 | 3584 | 512 |

因此 `mnclc -o` 的真实产物是 **`mncore::Program` = 变长 FlatBuffers 头 +
N×128 B 指令块**；`block_encoder.py` 产的是后段，头部（含重定位/参数表）
仍是缺口。这也说明 **relocation 属于 Program 头，不属于指令流**，
与 §13.3 的 loader 契约一致。

头长跟着 **fixup 数**走，不跟 arg 数走：同一 kernel 1 arg → 2 args 时
文件与头长都不变（49664 B / 256 B），只有偏移 112、120 处的两个 8 字节
字段移动。头部结构本身的初步反解（root table / vtable / 内核名字段）
见 [`mncore_mnir_offload.md`](mncore_mnir_offload.md) §6 A.1。

供应商 MNIR 的两个可用/不可用入口、`.mnbc` 与 `.mnir` 的可回灌性差异、
以及面向动态 offload 的 `mnrt` provider 分层设计，见
[`mncore_mnir_offload.md`](mncore_mnir_offload.md)。一句话：
**MNIR 是 Program 的生产者之一，不是运行时格式；LLVM 与 MNIR 应在
Program 层汇合。**
