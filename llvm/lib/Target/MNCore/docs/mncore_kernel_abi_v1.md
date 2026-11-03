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

## 13. Relocation 契约 v2（2026-11-03，待 runtime 实现）

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

### 13.2 产生方式（编译期）

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
  的数据，数值对拍通过；
* 负例：未绑定 site、`byte_offset % unit != 0`、地址超出 `width_bits`，
  三者都必须被拒绝。


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
