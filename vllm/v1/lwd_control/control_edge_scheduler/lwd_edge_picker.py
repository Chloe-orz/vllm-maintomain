# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""边侧 prefill 组批纯函数:续传优先 + FCFS 前缀 + 末位截断(PD_mix
唯一形态;PD 分离的单请求逃生通道已随相位机删除)。

chunk 语义依据 lwd_chunked_prefill_design.md §3.1,由 LwdEdgeScheduler
适配层组装候选后调用;输入输出全部 plain 值,不含调度器状态,表驱动
可测。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class LwdPrefillCandidate(NamedTuple):
    """组批候选:调度器在调用点把 Request 抽成本三字段。"""

    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int


def lwd_pick_prefill_batch(
    running: Sequence[LwdPrefillCandidate],
    waiting: Sequence[LwdPrefillCandidate],
    *,
    cap: int,
    free_seq: int,
    chunk_unit: int,
    single_prefill_min_tokens: int = 0,
) -> tuple[list[str], dict[str, int]]:
    """选下一步 embed 批,返回 (req_ids, expected)。

    running 为续传候选(保持 running 序,先收尾再开新);waiting 为开新
    候选(调用方拼好 skipped 优先 + waiting 的 FCFS 序)。chunk_unit 即
    chunk 单元 C,每块自限 c = min(剩余, C)。cap < 1 返回空批;新增
    请求数受 free_seq 钳制;末位截断:批内最后一个条目允许被剩余额度
    (cap − used)截尾,截断即收批。

    single_prefill_min_tokens:>0 时,总 prompt 长度达到该值的请求独占
    prefill 批(含续传,批内该请求前后不再装别的条目);0 = 关闭
    (DeepSeek V4 传 8000,其余模型传 0)。

    """
    picked: list[str] = []
    expected: dict[str, int] = {}
    if cap < 1:
        return picked, expected
    used, single_prefill = _pick_resumes(
        running, picked, expected, cap, chunk_unit, single_prefill_min_tokens
    )
    _pick_new_requests(
        waiting, picked, expected, used, cap, free_seq,
        chunk_unit, single_prefill_min_tokens, single_prefill,
    )
    return picked, expected


def _is_single_prefill(single_prefill_min_tokens: int, num_prompt_tokens: int) -> bool:
    """该请求是否触发独占规则(总 prompt 长度达标)。"""
    return (
        single_prefill_min_tokens > 0
        and num_prompt_tokens >= single_prefill_min_tokens
    )


def _append_chunk(
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    request_id: str,
    chunk: int,
) -> tuple[int, bool]:
    """往批里装一个 chunk;used + chunk 超预算走末位截断
    (剩余额度 c = cap − used,c ≥ 1 才入批,截断块同样占预算),
    截断即收批。返回 (新 used, 批是否已收口)。"""
    if used + chunk > cap:
        c = cap - used
        if c >= 1:
            picked.append(request_id)
            expected[request_id] = c
            used += c
        return used, True
    picked.append(request_id)
    expected[request_id] = chunk
    return used + chunk, False


def _pick_resumes(
    running: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    cap: int,
    chunk_unit: int,
    single_prefill_min_tokens: int,
) -> tuple[int, bool]:
    """续传循环:running 序逐请求取 c = min(剩余, C)。

    独占规则:达标请求置 single_prefill 标志,批内已有条目即当场收批
    (长请求不与短请求合批);标志跨循环传递(本批收口,开新循环不再
    装)。返回 (used, single_prefill 标志)。"""
    used = 0
    single_prefill = False
    for cand in running:
        if single_prefill and picked:
            break
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        # 独占判定用总 prompt 长度而非本 chunk 剩余:长请求后到达时先
        # 收掉当前批,留给下一批独占,不用 waiting 的新开绕过它
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        used, batch_closed = _append_chunk(
            picked, expected, used, cap,
            cand.request_id, min(remaining, chunk_unit),
        )
        if batch_closed:
            break
    return used, single_prefill


def _pick_new_requests(
    waiting: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    free_seq: int,
    chunk_unit: int,
    single_prefill_min_tokens: int,
    single_prefill: bool,
) -> int:
    """开新循环:waiting 序取 FCFS 前缀,c = min(prompt, C),新增数
    ≤ free_seq;不跳过队首(保 FCFS 与 seqno 链次序)。single_prefill
    由续传循环传入(True 即本批已收口)。返回 used。"""
    for new_count, cand in enumerate(waiting):
        if single_prefill and picked:
            break
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        chunk = min(cand.num_prompt_tokens, chunk_unit)
        if new_count >= free_seq:
            break
        used, batch_closed = _append_chunk(
            picked, expected, used, cap, cand.request_id, chunk
        )
        if batch_closed:
            break
    return used
