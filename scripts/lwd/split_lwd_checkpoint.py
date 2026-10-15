#!/usr/bin/env python3
"""把一个完整的 HF safetensors checkpoint 按 LWD 角色(edge/cloud)切成两个子集。

背景:边云 LWD 的每一侧只运行一部分模块(配套 model_load_update 分支里的
``LwdConfig.skips_module`` + ``lwd_skips_module()``(模型侧不构建本角色永不
执行的模块)与 ``lwd_weight_filter``(装载期丢弃占位模块的权重)),所以每侧
的模型目录只需要放自己会用到的张量。但 HF 的分片是**按张量交错打包**的,
不能"整片给某一侧",必须按张量名拆分。

用法(先 dry-run 看清单,再真拆):

    # 只看规则会保留什么(不写文件)
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --role edge --dry-run

    # 真正拆分到 /home/extra/Qwen3.6-27B-edge
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --dst /home/extra/Qwen3.6-27B-edge --role edge

    # 云侧(几乎不省:云侧仍需要 embed/norm/lm_head + 全部 transformer 层)
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --dst /home/extra/Qwen3.6-27B-cloud --role cloud

规则(可用 --keep-prefix / --keep-name-regex 追加)。默认同时覆盖三种命名布局:
VL 包装(model.language_model.*)、model.* 布局(Qwen3.5 等)、以及 DeepSeek 原生
布局(layers./norm./embed./head./hc_head/mtp.,无 model. 前缀)。

  edge  : head.* / lm_head.* / model.head.*        (unembed:rank-replay)
          + embed.* / embed_tokens.* / model.embed*.(embed:PP first rank)
          + hc_head*(V4 顶层无条件创建的真参数,边侧也要)
          + model.visual.*(多模态塔)
          + 按名匹配 <*>.layers.<i>.mlp|ffn.gate.tid2eid
            (DeepSeek-V4 的 Hash MoE 路由表:边侧不构建层,但会直接从
             checkpoint 分片里读这些表,漏了会报 "Missing tid2eid tables")
          (边侧只做 embed / unembed;transformer 层与 final norm 都不建)
  cloud : layers.* / model.layers.* / model.language_model.*(全部层)
          + norm.* + embed* + head.*/lm_head.* + hc_head*
          + mtp.* / model.mtp.*(开启投机解码时)
          (云侧不建视觉塔 visual)

非权重文件按角色拷贝(--metadata auto,默认;规则支持 * 通配)。目录默认不拷,
只有规则里显式点名(如 "yaml")或 --metadata all 才整目录拷贝。
边侧要对外服务并自己跑视觉塔,所以 tokenizer/processor/chat_template 等全拷;
云侧配套 vllm 的 LWD 改动(强制 skip_tokenizer_init + mm_encoder_remote,云侧
不再建 tokenizer 与多模态 processor),因此只拷 config.json/configuration.json
与量化描述(*quant*description*.json)——**后者必须带**,ModelSlim 量化靠它判定
(见 vllm-ascend 的 quantization/utils.py),缺了会被当成 float 加载。若云侧仍跑
未打补丁的 vllm,用 --metadata all 回到"全拷"。两侧都跳过 .git*、README.md、
LICENSE、.gitattributes。

(若不用 --metadata auto 而想让云侧也具备完整配置:--metadata all。)

输出:按 --max-shard-gb 切成若干 model-XXXXX-of-YYYYY.safetensors;多于一个
分片时写新的 model.safetensors.index.json,只有一个分片则不写(避免留下会
过滤掉自建分片的旧索引)。

写盘期间显示进度(百分比 / 已写 GiB / 速率 / ETA):终端上原地刷新一行,重定向到
日志文件时改为每 10 秒一行;用 --no-progress 关闭。进度只写 stderr,stdout 仍然
只有报告本身,方便 grep 与重定向。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

INDEX_NAME = "model.safetensors.index.json"

# 每侧必须保留的 checkpoint 名前缀。默认同时覆盖三种常见命名布局:
#   1) VL 包装布局      model.language_model.*      (Qwen3-VL 等)
#   2) model.* 布局     model.layers./norm./embed_tokens. (Qwen3.5 等)
#   3) DeepSeek 原生布局 layers./norm./embed./head./hc_head/mtp.
#      (见 vllm 的 _make_deepseek_v4_weights_mapper:
#       layers.->model.layers.、embed.->model.embed.、norm.->model.norm.、
#       head.weight->lm_head.weight、hc_head->model.hc_head、mtp.->model.mtp.)
ROLE_RULES: dict[str, tuple[str, ...]] = {
    "edge": (
        # --- 文本嵌入:边侧是 PP first rank,必须自己建 ---
        "model.language_model.embed_tokens.",
        "model.embed_tokens.",
        "model.embed.",
        "embed_tokens.",
        "embed.",  # DeepSeek 原生:embed.weight -> embed_tokens.weight
        # --- 输出头:边侧 rank-replay 还原 token,必须自己建 ---
        "lm_head.",
        "model.head.",
        "head.",  # DeepSeek 原生:head.weight -> lm_head.weight
        # --- DeepSeek-V4 顶层的 hc_head 参数:模型里**无条件**创建
        # (vllm/models/deepseek_v4/*/model.py 与 vllm_ascend/models/deepseek_v4.py),
        # 边侧也必须拿到,否则报缺权重 ---
        "hc_head",
        "model.hc_head",
        # --- 视觉塔:多模态必须(边侧自己跑编码) ---
        "model.visual.",
        # 其它命名的塔(vision_tower / vision_model / mm_projector 等)用
        # --keep-prefix 追加
    ),
    "cloud": (
        # --- 全部 transformer 层 + 尾部 norm ---
        "model.language_model.",  # VL 包装布局(含 embed/layers/norm)
        "model.layers.",
        "layers.",  # DeepSeek 原生布局
        "model.norm.",
        "norm.",
        # --- 文本嵌入:decode 行需要本地嵌入 ---
        "model.embed_tokens.",
        "model.embed.",
        "embed_tokens.",
        "embed.",
        # --- 采样头 + hc_head ---
        "lm_head.",  # 云侧采样 + 算秩
        "model.head.",
        "head.",
        "hc_head",
        "model.hc_head",
        # --- MTP(若开投机解码) ---
        "mtp.",  # DeepSeek 原生布局
        "model.mtp.",
    ),
}

# DeepSeek-V4 的 Hash MoE 路由表。边侧虽然不构建任何层,但会直接从 checkpoint
# 分片里读这些表(vllm-ascend 的 worker/lwd_edge_worker.py:load_model ->
# worker/lwd_hash/lwd_hash_tables.py),因此必须随边侧一起保留,否则边侧启动会报
# "Missing tid2eid tables for Hash MoE layers [...]"。
# 云侧不需要该特例:它构建全部层,这些表就是普通参数,由 layers./model.layers. 覆盖。
_TID2EID_PATTERN = re.compile(
    r"^(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.tid2eid(?:\.weight)?$"
)

# 前缀规则表达不了的"按整个张量名匹配"的保留项。
ROLE_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "edge": (_TID2EID_PATTERN,),
    "cloud": (),
}

SKIP_FILE_EXACT = {"README.md", "LICENSE", ".gitattributes"}
SKIP_FILE_PREFIX = (".git", INDEX_NAME)

# 每侧需要拷贝的非权重文件("auto" 策略),支持 * 通配(用 fnmatch 匹配);
# **目录默认不拷**:规则里显式点名某个目录(例如 "yaml")时才整目录递归拷贝,
# 或者用 --metadata all 全拷。
# 边侧对外服务、并且自己跑视觉塔编码 -> 全部配置/分词文件都要。
# 云侧配套 vllm 侧 LWD 改动(强制 skip_tokenizer_init + mm_encoder_remote):
# 不建 tokenizer、不建多模态 processor、不对外服务 -> 只需要 config.json;
# 但**量化描述必须带上**:ModelSlim(W8A8/W4A8...)靠目录里的
# quant_model_description.json 判定量化(见 vllm-ascend 的
# quantization/utils.py 与 quantization/modelslim_config.py),缺了会被当成
# float 加载,权重直接错。
METADATA_RULES: dict[str, tuple[str, ...]] = {
    "edge": ("*",),
    "cloud": (
        "config.json",
        "configuration.json",
        "*quant*description*.json",
        # 仅为兼容启动脚本里 "--chat-template <模型目录>/chat_template.jinja" 的写法;
        # 云侧不渲染模板(请求都在边侧),启动命令里没传 --chat-template 时可以不拷。
        "chat_template.jinja",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="完整模型目录(含 *.safetensors)")
    parser.add_argument("--dst", help="输出目录(与 --src 不同;--dry-run 时可省略)")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_RULES))
    parser.add_argument("--max-shard-gb", type=float, default=5.0, help="输出分片上限(默认 5GB;内存紧张就调小)")
    parser.add_argument("--keep-prefix", action="append", default=[], help="追加保留前缀(可重复)")
    parser.add_argument(
        "--keep-name-regex",
        action="append",
        default=[],
        help="追加按张量名匹配的正则(可重复;前缀规则表达不了时用,如仅保留某些层内参数)",
    )
    parser.add_argument(
        "--metadata",
        choices=("auto", "all"),
        default="auto",
        help="非权重文件策略:auto=按角色(云侧只拷 config.json/量化描述等必需文件);"
        "all=全拷(云侧仍跑未打补丁的 vllm 时用)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭写盘进度显示(默认:终端原地刷新一行;重定向到文件时每 10 秒输出一行)",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将保留/丢弃的清单,不写文件")
    return parser.parse_args()


def load_weight_map(src: str) -> dict[str, str]:
    index_path = os.path.join(src, INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    # 无索引:单分片或自建分片场景,退化为只按文件名列表
    weight_map: dict[str, str] = {}
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(os.path.join(src, name), framework="pt") as handle:
                for key in handle.keys():
                    weight_map[key] = name
    return weight_map


def select_keys(
    weight_map: dict[str, str],
    prefixes: tuple[str, ...],
    patterns: tuple[tuple[str, "re.Pattern[str]"], ...] = (),
) -> tuple[dict[str, list[str]], int]:
    """归组要保留的张量;返回 {规则标签: [张量名]} 与命中总数。

    先按前缀匹配,未命中的再用正则按整个张量名匹配(用于 DeepSeek-V4 的
    Hash MoE 路由表这类"藏在 layers.<i> 之下但边侧必须保留"的参数)。
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    matched = 0
    for key in weight_map:
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix].append(key)
                matched += 1
                break
        else:
            for label, pattern in patterns:
                if pattern.match(key):
                    grouped[label].append(key)
                    matched += 1
                    break
    return grouped, matched


def shard_tensor_bytes(path: str) -> dict[str, int]:
    """tensor 名 -> 字节数;只读 safetensors 头部,不读权重数据。

    safetensors 文件布局 = [8 字节小端 u64: 头长度 N][N 字节 JSON 头][数据],
    头里每个张量带 data_offsets,所以字节数可以精确算出(纯 stdlib,不依赖
    safetensors 版本)。
    """
    import struct

    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: 不是合法的 safetensors(文件太短)")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    sizes: dict[str, int] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        sizes[name] = end - start
    return sizes


def kept_bytes(weight_map: dict[str, str], src: str, keep: set[str]) -> tuple[int, int]:
    """返回 (保留张量的字节数, 缺失分片数)。"""
    want_per_shard: dict[str, set[str]] = defaultdict(set)
    for key in keep:
        want_per_shard[weight_map[key]].add(key)
    total = 0
    missing = 0
    for shard, keys in want_per_shard.items():
        path = os.path.join(src, shard)
        if not os.path.isfile(path):
            missing += 1
            continue
        sizes = shard_tensor_bytes(path)
        total += sum(sizes.get(key, 0) for key in keys)
    return total, missing


def report(
    weight_map: dict[str, str],
    src: str,
    grouped: dict[str, list[str]],
    prefixes: tuple[str, ...],
    patterns: tuple[tuple[str, "re.Pattern[str]"], ...] = (),
) -> int | None:
    """打印保留/丢弃清单与体积;返回保留字节数(统计失败时为 None)。"""
    per_shard_bytes = {name: os.path.getsize(os.path.join(src, name)) for name in set(weight_map.values())}
    total_bytes = sum(per_shard_bytes.values())

    labels = tuple(prefixes) + tuple(label for label, _ in patterns)
    width = max([40, *(len(label) for label in labels)])
    print(f"[in ] {len(weight_map)} tensor(s), {gib(total_bytes):.2f} GiB on disk")
    print(f"[rule] keep prefixes: {list(prefixes)}")
    if patterns:
        print(f"[rule] keep name patterns: {[label for label, _ in patterns]}")
    print("\n  matched tensors per rule:")
    kept = 0
    for label in labels:
        keys = grouped.get(label, [])
        if not keys:
            print(f"    {label:{width}s} 0 tensor(s)  <-- 没命中,检查命名布局/是否该保留")
            continue
        kept += len(keys)
        print(f"    {label:{width}s} {len(keys)} tensor(s)")
    print(f"\n  => keep {kept} tensor(s) of {len(weight_map)}; drop {len(weight_map) - kept}")

    # 被丢弃的张量按"前 3 段"聚一下,便于确认丢的都是层权重
    dropped_groups: dict[str, int] = defaultdict(int)
    keep = {key for keys in grouped.values() for key in keys}
    for key in weight_map:
        if key in keep:
            continue
        dropped_groups[".".join(key.split(".")[:3])] += 1
    print("\n  dropped tensors grouped by leading components (top 15):")
    for name, count in sorted(dropped_groups.items(), key=lambda item: -item[1])[:15]:
        print(f"    {name:48s} {count}")

    # 精确的保留体积(dry-run 时就能判断目标盘放不放得下)
    try:
        keep_size, missing_shards = kept_bytes(weight_map, src, keep)
    except (OSError, ValueError) as exc:
        print(f"\n  [size] 无法统计保留体积: {exc}")
        return None
    note = f"(有 {missing_shards} 个分片文件缺失,统计不完整)" if missing_shards else ""
    print(
        f"\n  [size] 将写出 {gib(keep_size):.2f} GiB / 源 {gib(total_bytes):.2f} GiB "
        f"({keep_size / max(total_bytes, 1) * 100:.1f}%) {note}"
    )
    return keep_size


def _metadata_candidates(src: str) -> tuple[list[str], list[str]]:
    """候选非权重条目 -> (文件, 目录),已排除 *.safetensors / index / .git* / README / LICENSE。

    目录单独返回:默认**不**拷贝(避免把导出附带的 ``yaml/`` 之类塞进模型目录),
    只有角色规则里显式点名(如 ``"yaml"``)或 ``--metadata all`` 时才拷。
    """
    files, dirs = [], []
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors") or name.startswith(SKIP_FILE_PREFIX):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        (dirs if os.path.isdir(os.path.join(src, name)) else files).append(name)
    return files, dirs


def plan_metadata(src: str, role: str, mode: str) -> tuple[list[str], list[str]]:
    """返回 (要拷贝的条目, 主动跳过的条目)。

    文件按角色规则(支持 ``*`` 通配)拷贝;目录默认不拷,除非规则里**显式点名**它
    (写 ``"*"`` 不算点名,免得 ``--metadata auto`` 顺手带走整个目录)或 mode 是 ``all``。
    """
    allowed = ("*",) if mode == "all" else METADATA_RULES[role]
    files, dirs = _metadata_candidates(src)

    def keep_file(name: str) -> bool:
        return any(fnmatch.fnmatch(name, rule) for rule in allowed)

    def keep_dir(name: str) -> bool:
        if mode == "all":
            return True
        return any(rule != "*" and fnmatch.fnmatch(name, rule) for rule in allowed)

    wanted = [name for name in files if keep_file(name)]
    wanted += [name for name in dirs if keep_dir(name)]
    skipped = [name for name in files + dirs if name not in wanted]
    return wanted, skipped


def copy_metadata(src: str, dst: str, names: list[str]) -> list[str]:
    """拷贝非权重条目;目录会整目录递归拷贝(例如导出附带的 yaml/)。"""
    copied = []
    for name in names:
        source_path = os.path.join(src, name)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, os.path.join(dst, name), dirs_exist_ok=True)
            copied.append(name + "/")
            continue
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dst, name))
        copied.append(name)
    return copied


def gib(num_bytes: int) -> float:
    """字节 -> GiB。"""
    return num_bytes / 2**30


def fmt_duration(seconds: float) -> str:
    """秒 -> "m:ss" / "h:mm:ss"。"""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Progress:
    """写盘进度显示(百分比 + 已写体积 + 速率 + ETA)。

    模式:
      bar   -- 终端:回车原地刷新一行(在 tty 上默认用它)
      lines -- 非 tty(重定向/日志):每 interval 秒输出一行,避免把日志刷满
      off   -- 不输出

    进度只写 stderr,stdout 依旧只有报告,方便 grep / 重定向。
    """

    def __init__(
        self,
        total_bytes: int,
        label: str,
        mode: str = "auto",
        stream=None,
        interval: float | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        if mode == "auto":
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
            mode = "bar" if is_tty else "lines"
        self.mode = mode
        self.total = max(int(total_bytes), 1)
        self.label = label
        self.interval = interval if interval is not None else (0.3 if mode == "bar" else 10.0)
        self.start = time.monotonic()
        self.last_update = 0.0
        self.done = 0
        self.extra = ""
        self.printed = False

    def update(self, delta_bytes: int, **extra: object) -> None:
        self.done += max(int(delta_bytes), 0)
        if extra:
            self.extra = " ".join(f"{key}={value}" for key, value in extra.items())
        if self.mode == "off":
            return
        now = time.monotonic()
        finished = self.done >= self.total
        if not finished and now - self.last_update < self.interval:
            return
        self.last_update = now
        elapsed = max(now - self.start, 1e-6)
        ratio = min(self.done / self.total, 1.0)
        speed = self.done / elapsed
        eta = (self.total - self.done) / speed if speed > 0 else 0.0
        text = (
            f"{self.label} {ratio * 100:5.1f}% "
            f"({gib(self.done):.1f}/{gib(self.total):.1f} GiB, "
            f"{gib(speed):.2f} GiB/s, ETA {fmt_duration(eta)})"
        )
        if self.extra:
            text += f" {self.extra}"
        if self.mode == "bar":
            print(f"\r{text}", end="\n" if finished else "", file=self.stream, flush=True)
        else:
            print(text, file=self.stream, flush=True)
        self.printed = True

    def close(self) -> None:
        """收尾(bar 模式下补一个换行,免得后续输出接在同一行)。"""
        if self.mode == "bar" and self.printed and self.done < self.total:
            print(file=self.stream, flush=True)
        self.printed = False


def write_slices(
    src: str,
    dst: str,
    weight_map: dict[str, str],
    keep: set[str],
    max_shard_bytes: int,
    progress: "Progress | None" = None,
) -> list[str]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    # 按源分片归组,顺序读取以控制内存峰值
    want_per_shard: dict[str, list[str]] = defaultdict(list)
    for key in keep:
        want_per_shard[weight_map[key]].append(key)

    written: list[str] = []
    buffer: dict[str, object] = {}
    buffer_bytes = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes
        if not buffer:
            return
        name = f"model-{len(written) + 1:05d}-of-XXXXX.safetensors"
        target = os.path.join(dst, name)
        try:
            save_file(buffer, target, metadata={"format": "pt"})
        except BaseException:
            # 半成品必须删掉:名字还带 -of-XXXXX,留着会被误当成有效分片
            if os.path.isfile(target):
                os.remove(target)
            raise
        written.append(name)
        buffer = {}
        buffer_bytes = 0

    try:
        seen = 0
        for shard in sorted(want_per_shard):
            with safe_open(os.path.join(src, shard), framework="pt") as handle:
                for key in want_per_shard[shard]:
                    tensor = handle.get_tensor(key)
                    buffer[key] = tensor
                    size = tensor.numel() * tensor.element_size()
                    buffer_bytes += size
                    seen += 1
                    del tensor
                    if progress is not None:
                        progress.update(size, shards=len(written), tensors=seen)
                    if buffer_bytes >= max_shard_bytes:
                        flush()
        flush()
    except BaseException:
        if progress is not None:
            progress.close()
        # 本次已写出的分片不完整,整套都不可用;清理掉,只保留 metadata 拷贝
        for name in written:
            path = os.path.join(dst, name)
            if os.path.isfile(path):
                os.remove(path)
        print(
            f"[clean] 写出失败,已删除本次生成的 {len(written)} 个分片"
            f"(目标目录只留 metadata 文件,可直接重跑)",
            file=sys.stderr,
        )
        raise
    if progress is not None:
        progress.close()

    # 分片名要带总数,先按最终数量重命名
    total = len(written)
    final_names = []
    for index, name in enumerate(written, start=1):
        final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(dst, name), os.path.join(dst, final_name))
        final_names.append(final_name)
    return final_names


def write_index(dst: str, weight_map: dict[str, str], keep: set[str], shard_names: list[str]) -> None:
    index_path = os.path.join(dst, INDEX_NAME)
    if len(shard_names) <= 1:
        # 单分片不需要索引;留着旧索引反而会把自建分片过滤掉
        if os.path.isfile(index_path):
            os.remove(index_path)
        return
    # 重新计算每个 key 落在哪个新分片
    new_weight_map: dict[str, str] = {}
    for name in shard_names:
        from safetensors import safe_open

        with safe_open(os.path.join(dst, name), framework="pt") as handle:
            for key in handle.keys():
                new_weight_map[key] = name
    total_size = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, handle, indent=2)
    missing = keep - set(new_weight_map)
    if missing:
        raise RuntimeError(f"{len(missing)} 个张量没有写进新分片,例如 {sorted(missing)[:5]}")


def main() -> int:
    args = parse_args()
    prefixes = tuple(args.keep_prefix) + ROLE_RULES[args.role]
    patterns: list[tuple[str, "re.Pattern[str]"]] = [
        (f"re:{pattern.pattern}", pattern) for pattern in ROLE_NAME_PATTERNS[args.role]
    ]
    for raw in args.keep_name_regex:
        try:
            patterns.append((f"re:{raw}", re.compile(raw)))
        except re.error as exc:
            print(f"--keep-name-regex 非法: {raw} ({exc})", file=sys.stderr)
            return 2
    patterns_tuple = tuple(patterns)
    # 相对路径按当前工作目录解析;顺带展开 ~,并消掉符号链接,
    # 这样下面"dst 不能等于/包含 src"的校验不会被软链接绕过。
    src = os.path.realpath(os.path.expanduser(args.src))
    if not os.path.isdir(src):
        print(f"源目录不存在: {src}", file=sys.stderr)
        return 2

    weight_map = load_weight_map(src)
    grouped, matched = select_keys(weight_map, prefixes, patterns_tuple)
    keep_size = report(weight_map, src, grouped, prefixes, patterns_tuple)
    if matched == 0:
        print("\n没有任何张量命中保留规则,请用 --keep-prefix/--keep-name-regex 指定正确规则。", file=sys.stderr)
        return 3

    meta_copy, meta_skip = plan_metadata(src, args.role, args.metadata)
    print(f"\n[meta] 将拷贝 {len(meta_copy)} 个非权重文件: {meta_copy}")
    if meta_skip:
        print(f"[meta] 主动跳过 {len(meta_skip)} 个: {meta_skip}")
    if args.metadata == "auto" and args.role == "cloud":
        print("[meta] 云侧最小化依赖 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote);"
              "若云侧仍跑未打补丁的 vllm,请加 --metadata all")
    if args.dry_run:
        return 0

    if not args.dst:
        print("非 dry-run 必须给 --dst", file=sys.stderr)
        return 2
    dst = os.path.realpath(os.path.expanduser(args.dst))
    if dst == src or src.startswith(dst + os.sep):
        print("为安全起见,--dst 不能等于/包含 --src", file=sys.stderr)
        return 2

    # 空间预检:宁可在这里退出,也不要在写第 N 个分片时把盘写满、留下半成品
    if keep_size:
        probe = dst
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        free = shutil.disk_usage(probe).free
        need = keep_size + 256 * 2**20  # 留 256MiB 余量
        print(
            f"\n[space] 目标 {probe} 可用 {free / 2**30:.2f} GiB,"
            f"需要 {need / 2**30:.2f} GiB",
        )
        if free < need:
            print(
                "空间不足:换一个有空间的 --dst(另一块盘/挂载点),或先清理目标盘。",
                file=sys.stderr,
            )
            return 5

    os.makedirs(dst, exist_ok=True)

    copied = copy_metadata(src, dst, meta_copy)
    print(f"\n[copy] {len(copied)} metadata file(s): {copied}")

    keep = {key for keys in grouped.values() for key in keys}
    # 进度:体积来自上面的精确统计;统计不到(分片缺失)或用户 --no-progress 时关闭
    progress = Progress(
        keep_size or 0,
        "[write]",
        mode="off" if (args.no_progress or not keep_size) else "auto",
    )
    shard_names = write_slices(
        src, dst, weight_map, keep, int(args.max_shard_gb * 2**30), progress=progress
    )
    print(f"[write] {len(shard_names)} shard(s): {shard_names}")
    write_index(dst, weight_map, keep, shard_names)

    # 复核:输出目录的键集合必须与计划一致
    from safetensors import safe_open

    written_keys: set[str] = set()
    for name in shard_names:
        with safe_open(os.path.join(dst, name), framework="pt") as handle:
            written_keys |= set(handle.keys())
    if written_keys != keep:
        print(f"[verify] FAIL: 计划 {len(keep)} / 实际 {len(written_keys)}", file=sys.stderr)
        return 4
    total = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    print(f"[verify] OK: {len(written_keys)} tensor(s), {total / 2**30:.2f} GiB in {dst}")
    print("[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
