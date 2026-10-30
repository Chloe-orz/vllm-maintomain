# 边云 LWD prefill_only 结构化输出恢复 设计文档

> 状态：**已实现**（提交 `dd41cc3dd`，5 个文件 +95/-24；`vllm-ascend` 零改动）。
> 背景：`2fdd04ac8`（load only the parameters each edge/cloud role runs）把 prefill_only 云侧的
> tokenizer 加载与切片 tokenizer 文件裁掉后，结构化输出不可用；本文档记录恢复方案、
> 修改点与验证方式。
> 涉及部署侧变化：云侧模型目录需补 tokenizer 文件（见 §6）。

---

## 1. 结论摘要

1. **背景**：`2fdd04ac8`（2026-10-12）在 `VllmConfig.__post_init__` 强制 prefill_only 云侧 `skip_tokenizer_init=True`，配套 `0336599ba`（切分脚本）让云侧切片不拷 tokenizer 文件——两者合起来即"云上 tokenizer 和词表加载被去掉"，结构化输出因此不可用。lm_head（unembedding 词表权重）从未被删。
2. **原理**：结构化输出的文法编译与采样 bitmask 都发生在采样所在的一侧——LWD 是 rank-replay 架构（云侧采样+秩编码，边侧按秩重放），执行点在云侧，云侧 tokenizer 必需；bitmask 施加后约束 token 经秩传到边侧，边侧恢复逻辑零改动。与原生集中式的基本原理完全一致，只是执行位置在云侧进程（对照见 §4 末）。
3. **实现**：恢复云侧 tokenizer 加载（多模态 processor/visual tower 裁剪保留，见 §3）；约束随边→云请求预告上 wire，云侧重建 Request 后按原生路径编译、调度、施加 bitmask。除一处必要的 picker 过滤（正确性修复，见 §4）外，不新增任何校验、守卫、配置开关。

---

## 2. 恢复内容与处置

| 项 | 来源提交 | 内容 | 处置 |
|---|---|---|---|
| 运行时强制 | `2fdd04ac8` | 云侧强制 `skip_tokenizer_init=True` | 删该强制行；`skip_mm_profiling/mm_encoder_remote` 保留 |
| 部署裁剪 | `0336599ba` | 切分脚本云侧切片不拷 tokenizer 文件 | 云侧规则补 tokenizer 条目 |
| renderer/reasoning guard | `c397e7f6d` / `2fdd04ac8` | tokenizer=None 的适配 guard | 保留（tokenizer 恢复后 inert，无害） |
| 模块裁剪/权重过滤 | `2fdd04ac8` 其余 hunks | skips_module / PPMissingLayer / lwd_weight_filter | 不动（独立显存优化，与本目标无关） |

历史从未存在、本次补齐的三处接缝：边侧拒绝块（`cd295b1b8` 起）、wire 缺约束字段（`LwdRequestNotify` 从未携带）、云侧 promote 路径不触发 `grammar_init`。

---

## 3. 为什么只恢复文本 tokenizer、不重建多模态 processor

多模态模型（如 Qwen3-VL）在 vLLM 里有三组独立的东西：

| 组件 | 位置/形态 | 加载物 | LWD 云侧是否需要 |
|---|---|---|---|
| **多模态 processor**（HF，如 Qwen2VLProcessor） | CPU 组件，引擎启动时 `MultiModalRegistry` 构建 | `preprocessor_config.json` 等 processor 配置（其内部还会加载一份 tokenizer） | **不需要**：云侧收到的是边侧算好的 embedding，预处理在边侧完成 |
| **visual tower**（VisionTransformer） | NPU 权重 | 权重张量 | **不需要**：模块裁剪已换成 PPMissingLayer（本方案不动） |
| **文本 tokenizer** | `cached_tokenizer_from_config` | `tokenizer.json / tokenizer_config.json / vocab*` 等 | **需要**：`StructuredOutputManager → XgrammarBackend` 编译结构化输出文法要用 |

因此 `skip_tokenizer_init` 不再强制 True，`mm_encoder_remote / skip_mm_profiling` 保留；云侧模型目录只需 tokenizer 文件、不需要 processor 文件——切分脚本也只补 tokenizer 条目。

---

## 4. 修改点明细（5 文件，+95/-24）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `vllm/config/vllm.py` | `VllmConfig.__post_init__` 云侧强制块：删 `skip_tokenizer_init = True` 一行，mm 强制保留；注释与启动日志同步 |
| 2 | `scripts/lwd/split_lwd_checkpoint.py` | `METADATA_RULES["cloud"]` 追加 tokenizer 条目（tokenizer.json / tokenizer_config.json / special_tokens_map.json / added_tokens.json / vocab.json / merges.txt / vocab.* / tokenizer.model）；docstring 同步 |
| 3 | `vllm/v1/lwd_control/control_communication/lwd_notify.py` | `LWD_WIRE_VERSION` 4→5（版本史注释补记）；`LwdRequestNotify` 新增 `structured_outputs: str = ""`——边侧 Processor 已校验的 StructuredOutputsParams 的 JSON 序列化，空串 = 普通请求 |
| 4 | `vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py` | ① `_lwd_validate_request` 删结构化输出拒绝块（其余校验原样保留）；② `lwd_edge_notify_request` 组包时把 `sp.structured_outputs` 经 `dataclasses.asdict` + `json.dumps` 序列化上 wire（含 `_backend`）；③ picker 过滤（见下） |
| 5 | `vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py` | ① 模块级 `_lwd_decode_structured_outputs()`：JSON 反序列化，`_backend/_backend_was_auto` 为 `init=False` 私有字段、不进构造单独回填，空串返回 None；② `_lwd_build_request` 构造 `SamplingParams(structured_outputs=...)`，Request 构造即原生派生 `structured_output_request`；③ `_lwd_promote` 在 ADD 入队前显式调 `structured_output_manager.grammar_init(request)`（promote 绕过 IO 线程 `preprocess_add_request`——那是原生唯一自动触发编译的入口） |

**picker 过滤（正确性修复，非守卫）**：结构化请求初始状态为 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`（`skipped_waiting` 队列，正是 picker 候选源）。边调度器 picker 的选中集必须被原生调度逐请求复现（步后 chunk 核验 `got != want` 即 RuntimeError 炸引擎），而 grammar 异步编译期间原生对 blocked 请求是 defer 语义——不过滤则 picker 选中、原生跳过 → 核验失守崩溃。修复：`_lwd_pick_prefill_batch` mixed 路径候选扫描加 `_lwd_grammar_ready(c)`（只读 grammar property 解析 future，不迁状态；状态提升仍由原生 `_try_promote_blocked_waiting_request` 完成）。phase 分支（逃生通道）未动——废弃分支且无核验契约。云侧无需同款：mixed 的 W2 核验天然容忍零排、下步重试。

**与原生集中式的对照（各环节归属）**：

| 环节 | 原生集中式 | LWD 边云（本次后） |
|---|---|---|
| 前端校验 | Processor + tokenizer | 同一套，在边侧（原生代码） |
| 文法编译 grammar_init | IO 线程 `preprocess_add_request` 自动触发 | 边侧同原生自动触发（解锁边调度门）；云侧 promote 显式调用（改动 5③），因请求经 wire 重建 |
| 调度门 | `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` → 原生状态提升 | 同一套原生机制，两侧调度器继承 |
| bitmask 生成 | `EngineCore.step` 的 `get_grammar_bitmask`（core.py:456/535） | 同一套，云引擎走原生 step |
| bitmask 施加 | `apply_grammar_bitmask`（sampler 前） | 同一套，ascend `model_runner_v1.py:2432`，云侧 `LwdCloudModelRunner.sample_tokens(grammar_output)` 继承 |
| 约束生效点 | 采样选 token | 同一处（云侧采样）；约束 token 经秩编码 DOWN、边侧按秩重放（数据面零改动） |
| 请求对象 | 同一进程内 Request 直通 | **跨进程重建**——约束须随 wire 携带（改动 3/4②/5①②，LWD 架构固有，非新机制） |

原生继承零改动的其他验证点：`AsyncScheduler._update_after_schedule` 置 `has_structured_output_requests/pending_structured_output_tokens`；ascend NPU bitmask kernel（`worker/v2/structured_outputs.py`）与 `patch_structured_output.py` 的 grammar_init 包装；lm_head 权重加载（两侧 head/tail 都建）；边引擎 IO 线程 `preprocess_add_request → grammar_init`。

**非结构化请求的性能影响**：热路径全部早退（`get_grammar_bitmask` 首行 `has_structured_output_requests` 判断、`grammar_output is None` 跳过 bitmask 施加、解码函数空串判空、picker 过滤状态比较、编译线程池 submit 时才懒建线程、bitmask 张量懒分配）——纯非结构化负载稳态性能零变化；差异仅在云侧启动多加载 tokenizer（一次性秒级）与每进程 CPU 内存 +几十 MB（NPU HBM 零变化）。注意**混批**（同一步既有结构化又有非结构化请求）时 bitmask 施加里 `logits.to("cpu")` 的往返是批级的，批内非结构化请求会分摊（与原生集中式行为一致）。

---

## 5. 验证

**本机已验证（无 torch/NPU，逻辑级）**：
- 5 个文件 `py_compile` 全过；
- wire 序列化/反序列化 round-trip：6 种约束（json/regex/choice/grammar/json_object/structural_tag）+ 空串 + 无 backend 路径逐值一致（`_backend_was_auto` 为前端簿记位，云侧不读，不上 wire 属正确语义）；
- msgspec 字段增量双向兼容：新云收旧边（缺字段→默认空串=普通请求）、旧云收新边（未知字段被容忍）；
- 既有测试核查：`test_lwd_module_scope.py` / `test_lwd_weight_filter.py` 无 skip_tokenizer_init 断言，LWD 测试无结构化输出引用，无需改测试。

**待测试环境执行**：
1. 云侧拉起：出现 "[LWD] cloud: tokenizer is initialized (structured output)..." 日志，`StructuredOutputManager.tokenizer` 非 None；
2. 端到端：OpenAI `response_format={"type":"json_schema"}` 与 `guided_json`/`guided_regex`，输出过 schema 校验；边云 token 一致性对拍（复用 `log_analyze_tools/lwd_token_diff.py`）；
3. 混跑回归：普通请求输出与改前逐 token 一致（bitmask None 时零路径变化）；
4. MTP + 结构化输出组合（原生 v1 支持 spec 场景 bitmask，云 mixed scheduler 已填 `scheduled_spec_decode_tokens`）；
5. 多 DP 云下结构化请求打到非 0 号 engine。

---

## 6. 部署约束

- **云侧模型目录须包含 tokenizer 文件**：重切分（新规则自动带）或手工补拷（tokenizer.json / tokenizer_config.json / special_tokens_map.json / added_tokens.json / vocab.json / merges.txt / tokenizer.model）；processor 文件不需要；
- 边云同版本部署（wire_version 5）：旧云 + 新边时结构化约束会被静默丢弃（msgspec 容忍未知字段），靠部署纪律保证，与 mrope/prefix 的"两侧必须同版本部署"约定同源；
- 云侧不要显式传 `--skip-tokenizer-init`（传了 tokenizer 不加载，结构化请求会在 grammar_init 处报 AttributeError——错误配置，不在代码里拦）。

---

## 7. 风险与取舍

| # | 风险 | 等级 | 说明 |
|---|---|---|---|
| R1 | 云侧启动重新加载 tokenizer（内存/启动时间小幅回退） | 低 | 纯文本模型无感 |
| R2 | 边+云双侧各编译一次 grammar（CPU、异步线程池） | 低 | 不占关键路径；边侧编译用于解锁调度门 |
| R3 | grammar 编译窗口内请求 defer 排队 | 低 | 原生语义，picker 过滤已对齐；复杂 schema 编译耗时体现为 TTFT 增加 |
| R4 | MTP + 结构化组合 | 中 | 原生已支持，必测项（§5.4） |
| R5 | 版本号 4→5 使 mrope/prefix/多 DP 的既有版本检查要求对端 ≥5 | 低 | 同版本部署下无感，与既有纪律一致 |
