# MN-Core 前端接入设计（Triton / TileLang）— 草案 (2026-11-03)

本文描述 `../triton-mncore` 与 `../tilelang-mncore` 如何接入本仓库的
MLIR 设备层与 LLVM 后端。设备运行时（模拟器/真机）的接入由另一条工作线
负责；本文只定义**编译产物边界**。

---

## 1. 唯一契约

前端不直接生成 `.vsm`，也不直接调用 `assemble3`。统一路径：

```
前端 IR
  │  (前端专用 lowering)
  ▼
mncore dialect  (mlir/Dialect/MNCore, 本仓库)
  │  convert-mncore-to-llvm
  ▼
LLVM IR: llvm.mncore.* intrinsics  (已冻结 ABI)
  │  llc -mtriple=mncore2
  ▼
.vsm 文本 + metadata sidecar
```

两条硬边界：

1. **`mncore` dialect 是设备语义的唯一入口**（`docs/mncore_mlir_dialect.md`、
   `mlir/include/mlir/Dialect/MNCore/IR/`）。
2. **`llvm.mncore.*` 是已冻结的机器层 ABI**
   （`llvm/include/llvm/IR/IntrinsicsMNCore.td`、
   `llvm/include/llvm/Target/MNCore/MNCoreIntrinsicOps.h`）。任何前端不得绕过
   `convert-mncore-to-llvm` 自行拼装 Layer A token。

### 1.1 安装产物

前端需要一份**带 MNCore target 的 LLVM/MLIR 安装**：

```sh
cmake -G Ninja -S llvm -B build-mlir \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD= \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=MNCore \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF
cmake --install build-mlir --prefix /opt/llvm-mncore
```

安装目录必须同时包含：

* `bin/llvm-config`、`bin/llc`、`bin/mlir-opt`、`bin/mlir-translate`；
* `lib/libMLIRMNCoreDialect.a`、`lib/libMLIRMNCoreToLLVM.a`；
* `include/mlir/Dialect/MNCore/...`、`include/mlir/Conversion/MNCoreToLLVM/...`；
* `include/llvm/Target/MNCore/MNCoreIntrinsicOps.h`（已迁移为公共头）。

前端构建时通过 `LLVM_SYSPATH`（Triton）或 `USE_LLVM`（TVM）指向该前缀。

---

## 2. Triton 3.9 接入（`../triton-mncore`）

现状（2026-11-03）：上游 `triton-lang/triton` main，版本 `3.9.0`；LLVM 由
`LLVM_SYSPATH` 提供，不再内置 LLVM。后端接口为
`BaseBackend.add_stages`（`python/triton/backends/compiler.py`）。

### 2.1 目录

```
third_party/mncore/
  CMakeLists.txt              # add_triton_library / add_triton_plugin
  include/mncore/Conversion/  # TritonGPU -> mncore 的 pass 声明
  lib/Conversion/             # TritonGPUToMNCore.cpp
  backend/compiler.py         # MncoreBackend(BaseBackend)
  backend/driver.py           # 启动/加载 .vsm（与另一条运行时工作线的接口）
```

构建：`-DTRITON_CODEGEN_BACKENDS=mncore -DLLVM_SYSPATH=/opt/llvm-mncore`。

### 2.2 stage 划分

`add_stages` 提供：

| stage | 内容 |
|---|---|
| `ttir` | 复用 Triton 的 TTIR pass 链 |
| `ttgir` | `convert_to_ttgpuir` 使用 mncore 的 layout |
| `mnir` | **新增**：`tritongpu-to-mncore`（TTGIR → `mncore` dialect） |
| `llir` | `convert-mncore-to-llvm` + LLVM dialect 翻译为 `.ll` 文本 |
| `vsm` | `llc -mtriple=mncore2` → `.vsm` 文本；附 metadata |

最终 stage 返回 bytes；driver 只认 `.vsm` + metadata。

实现要点（已核对 Triton 3.9 源码）：

* `llir` stage 返回 **LLVM IR 文本**（`str(llvm_mod)`），不是模块对象；
* `vsm` stage 调 `llvm.translate_to_asm(src, "mncore2", ...)`，内部用
  `TargetRegistry::lookupTarget` + `addPassesToEmitFile` 直接跑本仓库的
  `MNCoreAsmPrinter`，无需子进程；
* `llvm.attach_datalayout` 同样按 triple 创建 TargetMachine，要求该 LLVM
  构建包含 MNCore target；
* Triton 内置的 `llvm.init_targets()` 只初始化 NVPTX/AMDGPU，插件必须自行调用
  `LLVMInitializeMNCoreTargetInfo/Target/TargetMC/AsmPrinter`。

### 2.3 TTGIR → mncore 映射

| TTGIR | mncore |
|---|---|
| `tt.load` / `tt.store`（DRAM） | `mncore.mv` + `mncore.l2b` + `mncore.l1b`（`mncore.dma`，M2） |
| `ttg.local_alloc` | LM0/LM1（`mncore.peop` space 3/4） |
| `ttg.local_load/store` | `mncore.l1b` / PE 直接寻址 |
| `tt.dot` | `mncore.mat.write` + `mncore.mat.vec` + `mncore.mat.read` |
| elementwise | `mncore.vec.fma/mul/add/max/min` |
| `ttg.convert_layout` | `mncore.l1b`（broadcast/distribute/gather） |
| mask/select | `mncore.maskop` + `zmask`/`wmask` |
| barrier | `mncore.fence` + `mncore.wait`（tag 分配） |

**执行模型与 layout 契约**（决定 pass 的形状）：

* MN-Core 的一条 VLIW 表达式由**所有 PE 同时执行**；每个 PE 每步处理一个
  2 长语（f32×4 / f64×2 / f16×8）。因此一个 `mncore.vec.*` op 代表
  “所有 PE 的同一表达式”，其 `vector<4xf32>` 是**单个 PE 的切片**，不是
  一个 tensor。
* 首版要求 TTGIR 的 1-D blocked layout 为
  `sizePerThread = [4]`（f32）/`[2]`（f64）/`[8]`（f16）、`order = [0]`，
  即 lane *i* 独占连续的 4 个 f32。其他 layout 直接拒绝并给出错误，不做
  静默重排。
* `ttg.local_load`（L1BM/共享内存 → 寄存器）**不能固定 lowering 为
  `l1bmp`**。依据 MN-Core 2 Software Developer Manual §3.6.8，L1BM→PE 有四种
  分配方式（§3.6.8.6/.8/.14/.20）：

  | 指令 | 每 L1BM 每周期读取 | 64 PE 映射 |
  |---|---|---|
  | `l1bmp` | 1 longword | 同值广播到全部 64 PE |
  | `l1bmm` | 4 longwords | 4 个 PE 各取不同值，16 个 MAB 重复该组（MAB 维广播） |
  | `l1bmm4` | 16 longwords | 4 组不同数据，每组由 4 个 MAB 共享 |
  | `l1bmd` | 64 longwords | 64 个 PE 各取不同值（唯一真正的逐 PE 分发） |

  关键结论：

  1. `l1bmm`/`l1bmm4` 仍是 **MAB 维广播**，不是逐 PE 独立加载；
  2. 逐 PE 一对一分发只有 `l1bmd`，且手册明确 **`l1bmd` 没有 double-longword
     分发形式**；
  3. 因此 4×f32/PE（2 longword）的分布式 tensor 不能用一条两长语 `l1bmd`
     完成，需要按 longword 平面拆成两条 `l1bmd`，并要求 shared/L1BM 布局
     把同一平面（低/高 longword）连续排放，使两次读取的地址各自连续。

  实现要求：`mnir` pass 必须根据 **源 memdesc 的 shared layout 与目标
  distributed layout** 计算 PE 映射，再选择 `l1bmp`/`l1bmm`/`l1bmm4`/`l1bmd`。
  `mncore.vec.load` 因此携带显式 `mode`，当前只实现并验证了
  `Broadcast`（`l1bmp`）；其余三种在布局模型落地前由验证器显式拒绝。

  额外可利用的手册性质：**同一传输类型的读/写配对保持数据布局**，适合做
  `local_load → elementwise → local_store` 的 round-trip 测试。
* `tt.load`（DRAM）需要 MV → L2BM → L1BM 的 DMA 序列（M2，`mncore.dma`）；
  首版切片可先用 `ttg.local_load` 验证计算路径。
* `tt.store` 需要 PE → L1BM 的 `l1b` 方向；该方向目前**未接线**
  （isel 的 `set` 限定符仍拒绝），因此首版以 `mncore.mat.write` 观测结果。

**首版限制**：程序必须能完全展开为直线代码（后端拒绝
`BR/BRCOND/BR_CC/BR_JT/BRIND`，`MNCoreISelDAGToDAG.cpp:70-74`）。Triton 的
`scf.for`/`scf.if` 必须在 `mnir` stage 之前完全 unroll 或 if-convert 成掩码；
无法证明等价的副作用（DMA、同步）不得谓词化。

### 2.4 指针参数与重定位（关键）

`.vsm` 的地址字段是静态的（`packPEOperand`/`packMVOperand` 只接受立即数），
Triton 的运行时指针无法直接进入指令。因此：

* `mncore` 层把 buffer 参数表达为 `symbol + static offset`；
* `convert-mncore-to-llvm` 生成占位地址，并在模块 metadata 中记录
  `{arg_index, byte_offset, access}` fixup；
* loader（`mnrt`）在 `mnrt_load_program_ex` 时做重定位。

当前 `mnrt` 只记录 binding、**尚未实现 relocation**（见
`mncore_kernel_abi_v1.md` §10）。这是与运行时工作线的**交叉契约**，必须先冻结
fixup 记录格式。

**合法性门禁**：Triton pass 不得把动态 `tt.load` 指针无条件折叠为静态地址。
只有当寻址能证明是 `buffer base + static offset`（且 offset 按 §13 的 unit
整除）时才允许生成 `mncore.dma`；否则必须在合法化阶段报错，不得留到
assemble3 或 runtime。当前原型 pass 对 `tt.load` 一律拒绝。

### 2.5 版本风险

Triton 3.9 的上游 LLVM 版本与 LLVM 24 的 API 兼容性尚未验证。若不兼容，
优先级为：

1. 把 Triton 的 MLIR/LLVM 调用点适配到 24；
2. 若成本过高，改用**进程外契约**：Triton 后端只产出 `mncore` dialect 文本
   （stdout/文件），由外部 `mlir-opt --convert-mncore-to-llvm | mlir-translate | llc`
   完成余下步骤。该路径牺牲 JIT 时延，但保证 ABI 与版本解耦。

---

## 3. TileLang 接入（`../tilelang-mncore`）

现状：TVM 子模块已初始化（`3rdparty/tvm`，commit `4211874e`）；TileLang 有
自建 backend 的模板 `src/ascend/`（`TVM_REGISTER_TARGET_KIND("ascend", ...)` +
`CodeGenTileLangAscend : CodeGenTileLangCBase`）。Ascend 是 C 源码后端；MN-Core
需要 LLVM IR，因此走 **TVM 的 LLVM codegen** 路径，而不是复制 Ascend 的 C
codegen。

### 3.1 推荐路径：TVM `CodeGenLLVM` 子类

1. 用本仓库的 LLVM 安装构建 TVM：
   `USE_LLVM=/opt/llvm-mncore/bin/llvm-config`（必须包含 MNCore target）。
2. 在 `src/mncore/target.cc` 注册 target kind：
   ```cpp
   TVM_REGISTER_TARGET_KIND("mncore2", kDLExtDev)
       .add_attr_option<ffi::String>("mcpu")
       .set_default_keys({"mncore2"});
   ```
   并注册 `target.build.mncore2`（复用 `LLVMModuleNode`/`target.build.llvm` 的
   实现）。
3. `CodeGenLLVM::Create` 按 TargetMachine 名查找
   `tvm.codegen.llvm.target_mncore2`（`codegen_llvm.cc:110`）。注册工厂，
   返回 `CodeGenMNCore : CodeGenLLVM`：
   * `VisitExpr_(CallNode*)` / `FLowerIntrinsic` 规则发射
     `llvm.mncore.*` intrinsic；
   * 约定 TIR `thread_binding` → PE lane 映射；
   * 非法结构（动态循环、设备函数调用、越界静态资源）直接报错。
4. 产物：`LLVMModuleNode` 持有 `LLVMTarget`，可调
   `TargetMachine::addPassesToEmitFile(..., CGFT_AssemblyFile)`，由本仓库的
   `MNCoreAsmPrinter` 直接产出 `.vsm` 文本（无需子进程，且 LLVM 版本天然一致）。
   `CodeGenLLVM::Create` 按 TargetMachine 名查找
   `tvm.codegen.llvm.target_mncore2`，因此工厂名必须是 `target_mncore2`
   （MNCore 的 `RegisterTarget<Triple::mncore2>` 名字）。

### 3.2 TIR 映射前提

* TileLang 的 `T.Kernel` / 并行轴必须映射为 PE lane + 静态展开；TVM 的
  `ForKind::kParallel` 与 `thread_binding` 不能留给 LLVM 生成真实分支。
* `T.alloc_shared` → LM0/LM1；`T.copy` → MV/L2B/L1B 序列。
* GEMM tile → 矩阵寄存器（`mat.write` / `mat.vec` / `mat.read`）。
* 动态循环首版由 TileLang 在 host 侧分块或完全展开；后端不接收 CFG。

### 3.3 备选路径

若 TVM 与 LLVM 24 的 API 兼容性代价过大：

* **文本桥**：`CodeGenMNCore` 用 TVM 自带 LLVM 发射名为 `mncore.*` 的普通
  函数调用（非 `llvm.` 前缀，避免 TVM 侧 verifier 拒绝），在交给本仓库
  `llc` 前做一次受控文本重写为 `llvm.mncore.*`。该桥已在设计上可行，
  但必须由 golden 测试锁定，不能作为长期形态。
* 不建议在 TileLang 侧重写一套 LLVM IR emitter。

---

## 4. 共享缺口（按优先级）

1. **重定位/fixup + `mnrt` 支持**：前端 buffer 参数的前提（§2.4）。
2. **`mncore.dma` / buffer 描述符**：把 MV→L2B→L1B 序列、tag 分配、静态内存
   规划收敛到一个 op，避免前端各自拼 token。
3. **控制流策略**：MLIR 侧 unroll/if-conversion 的合法性规则；后端只接收
   直线代码。
4. **tag 分配与 double buffering**：MV tag/wait 的编译期分配。
5. **数值 golden**：当前只有 `d get`/`d set` 的最小闭环
   （`golden.mir`/`golden_mem.mir`）；前端端到端对拍需要模拟器工作线的
   `--dump` 支持。

---

## 5. 待办（本仓库侧）

* ~~把枚举头迁到公共 include~~ **已完成**：`llvm/include/llvm/Target/MNCore/`。
* ~~枚举同步检查~~ **已完成**：`MNCoreOps.cpp` 对 33 个值做 `static_assert`
  （MLIR TableGen 枚举 ↔ `MNCoreIntrinsicOps.h`），任何漂移都是编译错误。
* `mlir-translate` 端到端脚本：`.mlir → .ll → llc → .vsm`，并与现有
  `intrinsics.ll`/`layerb*.ll` 文本对拍。
* 在 `mncore` dialect 增加 `func` 级的 kernel metadata 输出
  （entry、参数、资源），供 loader 消费。

---

## 6. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| F0 | 本仓库 MLIR dialect + conversion 可用 | `mlir-opt --convert-mncore-to-llvm` lit 通过 |
| F1 | 安装前缀；Triton/TVM 能链接 `MLIRMNCoreDialect` | 两个前端各自最小 C++ 链接测试 |
| F2 | Triton `mnir` stage：单个 elementwise kernel | `.vsm` 可被 assemble3 接受 |
| F3 | TileLang target kind + `CodeGenMNCore` | 单个 copy kernel 产出 `.vsm` |
| F4 | buffer 参数 + fixup + `mnrt` 重定位 | 多 kernel buffer reuse 数值对拍 |
| F5 | dot / GEMM 路径 | 与 CPU golden 对拍 |

---

## 7. `mncore.dma` 设计 (2026-11-03)

目标：把 Triton `tt.load` / TileLang `T.copy` 的 DRAM→共享内存路径收敛到一个
op，避免前端各自拼 Layer A token。

### 7.1 语义

```
mncore.dma src_space, src_addr, l2_addr, l1_addr, size, tag, mv_flags
```

* 地址单位一律 **longword（8 B）**；
* 语义：`DRAM/PDM[src_addr .. +size) → L2BM[l2_addr ..) → L1BM[l1_addr ..)`；
* `tag` 供后续 `mncore.wait` 使用；`size` 为 longword 数；
* 只做搬运，不做同步；等待由显式的 `mncore.wait` 表达。

### 7.2 Lowering（全部是已验证的 Layer A 指令）

```llvm
%s = call i64 @llvm.mncore.mvop(i64 <src_space>, i64 <src_addr>, i64 <mv_flags>)
%d = call i64 @llvm.mncore.mvop(i64 2, i64 <l2_addr>, i64 <mv_flags>)
%p = call i64 @llvm.mncore.mvparams(i64 <size>, i64 <tag>, i64 -1, i64 -1)
call void @llvm.mncore.mv(i64 0, i64 %p, i64 %s, i64 %d)
%c = call i64 @llvm.mncore.lbop(i64 1, i64 <l2_addr>, i64 1, i64 0)
%b = call i64 @llvm.mncore.lbop(i64 0, i64 <l1_addr>, i64 1, i64 0)
call void @llvm.mncore.l2b(i64 0, i64 0, i64 %c, i64 %b)
```

`mvp` 的源/目的 kind 决定方向（`1=DRAM`, `2=L2BM`, `0=PDM`），因此
DRAM→L2BM 与 PDM→L2BM 共用同一 opcode；`l2bmb` 的 `set=0` 表示全部 L1B
（与 `intrinsics.ll` 的 `l2b1` 一致）。hazard/packetizer 会在 `mv` 与
`l2b`、`l2b` 与后续 `l1b` 之间插入必要的 nop。

### 7.3 未闭合

* **buffer 基址**仍走 §13 的 fixup 契约（地址字段静态）；
* 首版只支持**单块连续**搬运；多 tile / 跨 group / 广播（`mvb2/mvb4/mvd`）
  尚未暴露；
* L1B 子集 `set` 固定 0；多 L1B 分配需要额外的资源模型；
* `wait` 的 tag 由前端或 tag 分配 pass 负责；当前不自动插入。

---

## 8. Triton 3.9 ↔ LLVM 24 bring-up 记录 (2026-11-04)

上游 Triton 3.9 假定的是更老的 LLVM/MLIR；用本仓库的 LLVM 24 构建时，实际
需要以下补丁（均已应用到 `../triton-mncore`）：

| 文件 | 问题 | 处理 |
|---|---|---|
| `lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp` | `LoopAnnotationAttr::get` 少 2 个参数 | LLVM 24 新增 `align`、`parallelAccesses`；按新签名显式占位 |
| `python/src/llvm.cc` | `OptPassGate::shouldRunPass` 多 `funcName` 参数 | 覆盖函数改为 3 参 |
| `python/src/llvm.cc` | `PassBuilder` 构造函数新增 `LLVMContext&` 首参 | 传入 `mod->getContext()` |
| `python/src/llvm.cc` | `LLVMInitializeNVPTX*/AMDGPU*` 未声明 | 本仓库 LLVM 必须同时构建 `NVPTX;AMDGPU` target（Triton 核心无条件初始化它们） |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` | `arith::populateCeilFloorDivExpandOpsPatterns` 已移除 | 该展开已并入 `populateArithToLLVMConversionPatterns`，删除显式调用 |
| `python/src/ir.cc` 及 `third_party/{nvidia/hopper,amd}` | Release 下 `dump()` 被 `LLVM_DUMP_METHOD` 裁掉 | 以 `-DLLVM_ENABLE_DUMP=ON` 构建 LLVM（上游假定 assertions 或 dump 可用） |

构建配置（已验证 configure 通过）：

```sh
# LLVM/MLIR
cmake -G Ninja -S llvm -B build-mlir \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="NVPTX;AMDGPU" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=MNCore \
  -DLLVM_ENABLE_DUMP=ON -DCMAKE_BUILD_TYPE=Release

# Triton（nanobind 需要装在独立 venv 中）
cmake -G Ninja -S . -B ../triton-mncore-build \
  -DLLVM_SYSPATH=/Users/yanlu/Documents/llvm-mncore/build-mlir \
  -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
  -DTRITON_CACHE_PATH=... -DTRITON_WHEEL_DIR=... \
  -DTRITON_BUILD_PYTHON_MODULE=ON -DTRITON_BUILD_UT=OFF \
  -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_NVIDIA_GSAN_RUNTIME=OFF \
  -DPython3_EXECUTABLE=<venv>/bin/python -Dnanobind_ROOT=<venv>/.../nanobind/cmake
```

注意：

* `LLVM_SYSPATH` 指向 `build-mlir` 即可：`MLIRConfig.cmake` 会把源码
  `mlir/include` 与生成头目录都导出，不需要 `cmake --install`；
* `TRITON_CODEGEN_BACKENDS` 不能为空：核心的 `TritonInstrumentToLLVM`
  依赖 NVIDIA 方言的生成头；
* AMD 方言被 `python/src/gluon_ir.cc` 无条件 include，因此 `amd` 也必须启用；
* Triton 的 TTGIR layout 默认按 GPU warp 选择；mncore backend 的
  `make_ttgir` 目前只是占位，真正的 mncore layout 在 `mnir` pass 中定义。

---

## 9. Triton 接入验收门禁 (2026-11-04)

| 阶段 | 测试 | 验收目标 | 当前状态 |
|---|---|---|---|
| T0 | plugin import / discovery | `triton.backends` 含 `mncore`，LLVM/MLIR 无重复链接冲突 | 待构建完成 |
| T1 | 最小 TTGIR elementwise | 合法生成 `mncore` dialect | pass 已写，待编译 |
| T2 | `local_alloc → local_load` | memdesc 物理存储域一致（首版统一 L1BM） | 已定契约；load 待布局映射 |
| T3 | 不同 PE 值的 distributed tensor | 数值级 layout 映射正确（四种模式） | **关键新增门禁**；`Distributed` 双 longword 暂被拒绝 |
| T4 | `local_load → fma → writeback` | 寄存器生命周期与 Hazard | 计算链已验证；writeback 未接线 |
| T5 | `DMA → load → compute → store` | 完整数据闭环 | `dma` 已验证，store 待做 |
| T6 | 小尺寸 GEMM | 端到端数值 | 未开始 |

T3 的数值判据见 `mncore_mlir_dialect.md` §8.6（`A[m,p] = 100m + p` 的四种
模式期望表）。在 T3 通过前，任何 Triton kernel 都不得宣称语义正确。
