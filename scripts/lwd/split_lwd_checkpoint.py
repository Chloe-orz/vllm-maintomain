#!/usr/bin/env python3
"""把一个完整的 HF safetensors checkpoint 按 LWD 角色(edge/cloud)切成两个子集。

背景:边云 LWD 的每一侧只运行一部分模块(配套 model_load_update 分支
里的 ``patch_lwd_module_scope`` + ``lwd_weight_filter``),所以每侧的模型
目录只需要放自己会用到的张量。但 HF 的 15 个分片是**按张量交错打包**的,
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

规则(可用 --keep-prefix 覆盖/追加,默认同时兼容 VL 与纯文本两种命名):

  edge  : 顶层 lm_head.* + model.language_model.embed_tokens.* + model.visual.*
          (边侧只做 embed / unembed;transformer 层与 final norm 都不建)
  cloud : 顶层 lm_head.* + model.language_model.*(全部:embed/layers/norm)
          + mtp.* / model.mtp.*(开启投机解码时)
          (云侧不建视觉塔 visual)

非权重文件按角色拷贝(--metadata auto,默认):边侧要对外服务并自己跑视觉塔,
所以 tokenizer/processor/chat_template 等全拷;云侧配套 vllm 的 LWD 改动
(强制 skip_tokenizer_init + mm_encoder_remote,云侧不再建 tokenizer 与多模态
processor),因此**只拷 config.json**。若云侧仍跑未打补丁的 vllm,用
--metadata all 回到"全拷"。两侧都跳过 .git*、README.md、LICENSE、.gitattributes。

(若不用 --metadata auto 而想让云侧也具备完整配置:--metadata all。)

输出:按 --max-shard-gb 切成若干 model-XXXXX-of-YYYYY.safetensors;多于一个
分片时写新的 model.safetensors.index.json,只有一个分片则不写(避免留下会
过滤掉自建分片的旧索引)。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

INDEX_NAME = "model.safetensors.index.json"

# 每侧必须保留的 checkpoint 名前缀。
ROLE_RULES: dict[str, tuple[str, ...]] = {
    "edge": (
        "lm_head.",  # 边侧 rank-replay 还原 token
        "model.language_model.embed_tokens.",  # VL 布局:文本嵌入
        "model.embed_tokens.",  # 纯文本布局:文本嵌入
        "model.visual.",  # 视觉塔(多模态必须)
    ),
    "cloud": (
        "lm_head.",  # 云侧采样 + 算秩
        "model.language_model.",  # VL 布局:embed/layers/norm
        "model.layers.",  # 纯文本布局:transformer 层
        "model.norm.",  # 纯文本布局:final norm
        "model.embed_tokens.",  # 纯文本布局:文本嵌入(decode 行本地嵌入)
        "mtp.",  # 独立 MTP 权重(若开投机解码)
        "model.mtp.",
    ),
}

SKIP_FILE_EXACT = {"README.md", "LICENSE", ".gitattributes"}
SKIP_FILE_PREFIX = (".git", INDEX_NAME)

# 每侧需要拷贝的非权重文件("auto" 策略)。
# 边侧对外服务、并且自己跑视觉塔编码 -> 全部配置/分词文件都要。
# 云侧配套 vllm 侧 LWD 改动(强制 skip_tokenizer_init + mm_encoder_remote):
# 不建 tokenizer、不建多模态 processor、不对外服务 -> 只需要 config.json。
METADATA_RULES: dict[str, tuple[str, ...]] = {
    "edge": ("*",),
    "cloud": ("config.json",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="完整模型目录(含 *.safetensors)")
    parser.add_argument("--dst", help="输出目录(与 --src 不同;--dry-run 时可省略)")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_RULES))
    parser.add_argument("--max-shard-gb", type=float, default=5.0, help="输出分片上限(默认 5GB;内存紧张就调小)")
    parser.add_argument("--keep-prefix", action="append", default=[], help="追加保留前缀(可重复)")
    parser.add_argument(
        "--metadata",
        choices=("auto", "all"),
        default="auto",
        help="非权重文件策略:auto=按角色(云侧只拷 config.json);all=全拷(云侧仍跑未打补丁的 vllm 时用)",
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


def select_keys(weight_map: dict[str, str], prefixes: tuple[str, ...]) -> tuple[dict[str, list[str]], int]:
    """按前缀归组要保留的张量;返回 {前缀: [张量名]} 与命中总数。"""
    grouped: dict[str, list[str]] = defaultdict(list)
    matched = 0
    for key in weight_map:
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix].append(key)
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
    weight_map: dict[str, str], src: str, grouped: dict[str, list[str]], prefixes: tuple[str, ...]
) -> int | None:
    """打印保留/丢弃清单与体积;返回保留字节数(统计失败时为 None)。"""
    def gib(num_bytes: int) -> float:
        return num_bytes / 2**30

    per_shard_bytes = {name: os.path.getsize(os.path.join(src, name)) for name in set(weight_map.values())}
    total_bytes = sum(per_shard_bytes.values())

    print(f"[in ] {len(weight_map)} tensor(s), {gib(total_bytes):.2f} GiB on disk")
    print(f"[rule] keep prefixes: {list(prefixes)}")
    print("\n  matched tensors per prefix:")
    kept = 0
    for prefix in prefixes:
        keys = grouped.get(prefix, [])
        if not keys:
            print(f"    {prefix:40s} 0 tensor(s)  <-- 没命中,检查命名布局/是否该保留")
            continue
        kept += len(keys)
        print(f"    {prefix:40s} {len(keys)} tensor(s)")
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


def _metadata_candidates(src: str) -> list[str]:
    """候选非权重文件(已排除 *.safetensors / index / .git* / README / LICENSE)。"""
    names = []
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors") or name.startswith(SKIP_FILE_PREFIX):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        if not os.path.isfile(os.path.join(src, name)):
            continue
        names.append(name)
    return names


def plan_metadata(src: str, role: str, mode: str) -> tuple[list[str], list[str]]:
    """返回 (要拷贝的文件, 主动跳过的文件)。"""
    allowed = ("*",) if mode == "all" else METADATA_RULES[role]
    wanted = [name for name in _metadata_candidates(src) if "*" in allowed or name in allowed]
    skipped = [name for name in _metadata_candidates(src) if name not in wanted]
    return wanted, skipped


def copy_metadata(src: str, dst: str, names: list[str]) -> list[str]:
    copied = []
    for name in names:
        source_path = os.path.join(src, name)
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dst, name))
        copied.append(name)
    return copied


def write_slices(src: str, dst: str, weight_map: dict[str, str], keep: set[str], max_shard_bytes: int) -> list[str]:
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
        for shard in sorted(want_per_shard):
            with safe_open(os.path.join(src, shard), framework="pt") as handle:
                for key in want_per_shard[shard]:
                    tensor = handle.get_tensor(key)
                    buffer[key] = tensor
                    buffer_bytes += tensor.numel() * tensor.element_size()
                    del tensor
                    if buffer_bytes >= max_shard_bytes:
                        flush()
        flush()
    except BaseException:
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
    # 相对路径按当前工作目录解析;顺带展开 ~,并消掉符号链接,
    # 这样下面"dst 不能等于/包含 src"的校验不会被软链接绕过。
    src = os.path.realpath(os.path.expanduser(args.src))
    if not os.path.isdir(src):
        print(f"源目录不存在: {src}", file=sys.stderr)
        return 2

    weight_map = load_weight_map(src)
    grouped, matched = select_keys(weight_map, prefixes)
    keep_size = report(weight_map, src, grouped, prefixes)
    if matched == 0:
        print("\n没有任何张量命中保留规则,请用 --keep-prefix 指定正确前缀。", file=sys.stderr)
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
    shard_names = write_slices(src, dst, weight_map, keep, int(args.max_shard_gb * 2**30))
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
