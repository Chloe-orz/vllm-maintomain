# LWD 模型文件拆分工具使用说明

本目录下的 `split-lwd-checkpoint.jar` 用于把一份**完整的 HF safetensors
checkpoint** 按 LWD（layerwise disaggregated）角色拆成两个子集：边侧只保留
embed / unembed / 视觉塔所需的权重，云侧只保留 transformer 层、norm、embed、
lm_head 与 MTP，从而让每一侧只持有一部分模型文件。

本工具是纯字节级实现（safetensors 布局 = `[8 字节小端头长度][JSON 头][原始张量数据]`），
**不依赖 Python、safetensors 库或任何第三方 jar**，运行时只需要一个 JRE。

---

## 1. 运行环境

```bash
java -version          # 需要 Java 8 或更高版本(JRE 即可,不需要 JDK)
```

如果 `java` / `jar` 不在 `PATH` 中，用 JDK/JRE 的 `bin` 目录全路径调用，例如：

```bash
"/c/Program Files/Java/jdk-21.0.12.1/bin/java" -jar split-lwd-checkpoint.jar --help
```

---

## 2. 快速开始

```bash
# 0) 先 dry-run:只读,不写任何文件,用来核对张量数量与体积
java -jar split-lwd-checkpoint.jar --src /path/to/Qwen3.6-27B --role cloud --dry-run
java -jar split-lwd-checkpoint.jar --src /path/to/Qwen3.6-27B --role edge  --dry-run

# 1) 真跑(注意 --dst 要与 --src 在不同目录;建议放在空间足够的盘上)
java -jar split-lwd-checkpoint.jar \
    --src /path/to/Qwen3.6-27B --dst /data/models/Qwen3.6-27B-cloud \
    --role cloud --max-shard-gb 8

java -jar split-lwd-checkpoint.jar \
    --src /path/to/Qwen3.6-27B --dst /data/models/Qwen3.6-27B-edge \
    --role edge --max-shard-gb 8

# 2) 用拆分出来的目录启动对应角色
#    边侧:需要 config.json + tokenizer/processor 配置 + 边侧权重
#    云侧:只需 config.json + 云侧权重(依赖 vLLM 侧的 LWD 补丁,见第 6 节)
```

---

## 3. 命令行参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--src <目录>` | 是 | 完整模型目录（含 `*.safetensors`）。支持相对路径与 `~`；相对路径按当前工作目录解析 |
| `--dst <目录>` | 非 dry-run 必填 | 输出目录。不能等于或包含 `--src`（安全校验），不存在会自动创建 |
| `--role {edge,cloud}` | 是 | 拆给哪一侧，决定保留哪些张量（见第 4 节） |
| `--max-shard-gb <GB>` | 否 | 输出分片大小上限，默认 `5`。目标盘够大时给 `8` 或更大可减少分片数量 |
| `--keep-prefix <前缀>` | 否 | 追加保留前缀，可重复，优先级在角色规则之前 |
| `--metadata {auto,all}` | 否 | 非权重文件策略，默认 `auto`（见第 6 节） |
| `--dry-run` | 否 | 只打印将保留/丢弃的清单与体积，不写文件 |
| `--help`, `-h` | 否 | 打印用法 |

---

## 4. 角色保留规则

保留判定是**按名字前缀 `startswith` 匹配**（`--keep-prefix` 追加的前缀排在最前）：

| 角色 | 保留前缀 | 说明 |
| --- | --- | --- |
| `edge` | `lm_head.` | 边侧 rank-replay 还原 token |
| | `model.language_model.embed_tokens.` | VL 布局：文本嵌入 |
| | `model.embed_tokens.` | 纯文本布局：文本嵌入 |
| | `model.visual.` | 视觉塔（多模态必须，边侧负责编码） |
| `cloud` | `lm_head.` | 云侧采样 + 算秩 |
| | `model.language_model.` | VL 布局：embed / layers / norm |
| | `model.layers.` | 纯文本布局：transformer 层 |
| | `model.norm.` | 纯文本布局：final norm |
| | `model.embed_tokens.` | 纯文本布局：文本嵌入（decode 行本地嵌入） |
| | `mtp.` / `model.mtp.` | 独立 MTP 权重（开启投机解码时） |

**被丢弃的张量**：

- 边侧：全部 `…layers.*`、`…norm.*`、`mtp.*`（边侧不跑 transformer）；
- 云侧：全部 `model.visual.*`（云侧不跑视觉塔）。

`dry-run` 会按“名字前 3 段”分组打印丢弃清单，便于快速判断是否拆错。

---

## 5. 输出内容

- 权重分片命名为 `model-00001-of-000NN.safetensors`；
- 分片数 `> 1` 时写入新的 `model.safetensors.index.json`（`weight_map` 指向新分片）；
  分片数 `== 1` 时**删除**同目录下同名旧索引，避免旧索引把自建分片过滤掉；
- 头部按新打包顺序重算 `data_offsets`，并写 `__metadata__ = {"format": "pt"}`；
- 数据段以 8 MiB 缓冲**流式拷贝**，因此内存占用与分片大小无关（不需要大堆内存）；
- **空间预检**：写盘前检查目标盘可用空间，低于「将写出体积 + 256 MiB」时直接以退出码 `5`
  中止并提示换盘，不会写到一半失败；
- **失败自动清理**：写出过程中出错会删除本次生成的分片，只保留已拷贝的非权重文件，可直接重跑。

---

## 6. 非权重文件策略（`--metadata`）

| 取值 | 边侧 | 云侧 |
| --- | --- | --- |
| `auto`（默认） | 全拷：`config.json`、`tokenizer*.json`、`vocab.json`、`merges.txt`、`chat_template.jinja`、`preprocessor_config.json`、`video_preprocessor_config.json`、`generation_config.json` 等 | **只拷 `config.json`** |
| `all` | 全拷 | 全拷 |

两侧都会跳过：`.git*`、`README.md`、`LICENSE`、`.gitattributes`、`model.safetensors.index.json`
（索引由本工具按需重建）。

> **`auto` 模式对云侧的前提**：云侧运行的 vLLM 必须包含 LWD 侧补丁
> （云侧强制 `skip_tokenizer_init` 与 `mm_encoder_remote`，即 `feat(lwd): load only the
> parameters each edge/cloud role runs` 这次提交）。补丁生效后云侧不再构建 tokenizer 与
> 多模态 processor，因此只需 `config.json`。
>
> 如果云侧仍运行**未打补丁**的 vLLM，请使用 `--metadata all`；否则云侧会在启动时报
> `OSError: Can't load image processor for ... containing a preprocessor_config.json`。

---

## 7. 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功（dry-run 走完也算成功） |
| `1` | 其它失败（IO 错误、JSON 解析失败等，会打印 `失败: …`） |
| `2` | 参数或路径非法（缺 `--src`、`--role` 取值错、非 dry-run 未给 `--dst`、`--dst` 等于/包含 `--src`、源目录不存在） |
| `3` | 没有任何张量命中保留规则（通常是把 `--role` 选错，或模型布局与规则不匹配） |
| `4` | 写完后复核失败（输出分片的键集合与计划不一致） |
| `5` | 目标盘空间不足 |

---

## 8. 输出日志说明

```text
[in ] 1199 tensor(s), 51.75 GiB on disk
[rule] keep prefixes: [lm_head., model.language_model., ...]

  matched tensors per prefix:
    lm_head.                                 1 tensor(s)
    model.language_model.                    863 tensor(s)
    model.mtp.                               0 tensor(s)  <-- 没命中,检查命名布局/是否该保留

  => keep 864 tensor(s) of 1199; drop 335

  dropped tensors grouped by leading components (top 15):
    model.visual.patch_embed                 3
    model.visual.blocks                      330

  [size] 将写出 50.31 GiB / 源 51.75 GiB (97.2%)

[meta] 将拷贝 1 个非权重文件: [config.json]
[meta] 主动跳过 9 个: [chat_template.jinja, configuration.json, ...]
[meta] 云侧最小化依赖 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote);若云侧仍跑未打补丁的 vllm,请加 --metadata all

[space] 目标 /data/models 可用 851.28 GiB,需要 50.56 GiB

[copy] 1 metadata file(s): [config.json]
[write] 1 shard(s): [model-00001-of-00001.safetensors]
[verify] OK: 864 tensor(s), 50.31 GiB in /data/models/Qwen3.6-27B-cloud
[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量
```

关键行：

- `[size]`：将要写出的体积（只读 safetensors 头部统计，不读权重数据），用来判断盘是否够；
- `[meta]`：将拷贝 / 主动跳过的非权重文件；
- `[space]`：目标盘可用空间与需求对比；
- `[verify] OK`：写完后回读复核通过（张量数与体积）。

---

## 9. 常见问题

### 9.1 `No space left on device (os error 28)`

目标盘空间不足。处理方式：

- 把 `--dst` 换到另一块有空间的盘/挂载点（**不要**靠调小 `--max-shard-gb`，它只改变分片粒度，不减少总量）；
- 或者只拆一侧：云侧直接用原始完整目录（vLLM 侧补丁会在读盘前跳过用不到的权重），只给边侧腾出空间。

当前版本在写盘前会做空间预检（退出码 `5`），失败时也会自动清理本次产生的分片；若你用的是
更早的版本，请手动删除目标目录里被截断的 `*.safetensors` 后再重跑。

### 9.2 云侧启动报 `Can't load image processor ... preprocessor_config.json`

云侧 vLLM 还没有 LWD 补丁（`skip_tokenizer_init` + `mm_encoder_remote`）。两种处理：

- 临时：重新用 `--metadata all` 生成云侧目录（把 `preprocessor_config.json` 等拷回去）；
- 正式：把 LWD 补丁同步到云侧代码后，再用默认的 `--metadata auto`。

### 9.3 启动报 `Following weights were not initialized from checkpoint: [...]`

按报错列出的张量名把对应分片补回来即可。常见原因：

- `--role` 选错（例如把 edge 目录给了云侧启动）；
- 手工删过文件，删掉了该角色需要的张量；
- 云侧开启了 MTP 但目录里没有 `mtp.*` 权重。

### 9.4 日志里没有 `LWD weight filter: skipped N tensor(s)`

这条日志来自 vLLM 侧的权重过滤器（`logger.debug`），要同时满足：

1. 日志级别放开：`VLLM_LOGGING_LEVEL=DEBUG`（vLLM 默认 `INFO`）；
2. 模型目录**确实包含**该角色不构建的模块张量（用完整 checkpoint 才会命中；用本工具拆出来的
   目录本来就没有这些张量，因此 `skipped` 为 0、不打日志，这是正常现象）；
3. vLLM 代码包含该过滤器（`model_loader/default_loader.py` 中的 `_skip_unbuilt_weights`）。

---

## 10. 校验拆分结果

```bash
python - <<'PY'
import json

for role, path in (("edge", "./Qwen3.6-27B-edge"), ("cloud", "./Qwen3.6-27B-cloud")):
    wm = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
    layers = sum(k.startswith(("model.language_model.layers.", "model.layers.")) for k in wm)
    norm = sum(k.endswith("norm.weight") for k in wm)
    visual = sum(".visual." in k or k.startswith("visual.") for k in wm)
    print(f"{role:5s} tensors={len(wm):5d} layers={layers:4d} norm={norm:2d} visual={visual:4d}")
# 期望:edge 的 layers=0;cloud 的 visual=0
PY
```

单分片输出（`--max-shard-gb` 足够大）不会生成索引文件，此时可改为直接读分片头部的键集合。

---

## 11. 重新构建（可选）

本工具是单文件 Java 实现（无第三方依赖）。若需要重新编译：

```bash
javac -d build SplitLwdCheckpoint.java
jar cfe split-lwd-checkpoint.jar SplitLwdCheckpoint -C build .
rm -rf build
```

只运行 jar 的机器不需要 JDK，有 JRE 即可。
