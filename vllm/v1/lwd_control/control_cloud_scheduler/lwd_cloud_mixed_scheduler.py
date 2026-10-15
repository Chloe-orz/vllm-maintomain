"""云侧混合调度器:prefill/decode 同批下发(设计:lwd_mixed_batch_design.md
+ lwd_chunked_prefill_design.md + docs/design/lwd_host_staging.md)。

chunk 混排语义:边侧按 c = min(剩余 prompt, chunk 单元) 切块组批
(续传优先 + FCFS 前缀),云侧每步至多消费 prefill_notify_queue 队首
一条批量 RangeNotify,与全部 decode 态请求混排一个批下发:

  - **可见集谓词**:`recv_rows − num_computed <= 0` 的请求本步不进 batch
    (数据面没到就算 = 读零占位、静默错算;落点是既有的临时可见集机制,
    原生 scheduler.py 一行不改);
  - 放行先做 offset 对账(item.offset == 云侧 num_computed,chunk
    断流/重复/乱序即 fail-fast);
  - **按请求下发**:每请求本步行数 = `min(已收 − 已算, 剩余预算)`,靠
    `max_num_scheduled_tokens = decode_exact + Σ该值` 交给原生 —— 原生没有
    per-request 钩子,这个总预算配合 FIFO 前缀序精确等于逐请求取小;
  - **下发前 W2 核验**:逐请求 `num_scheduled ≤ recv_rows − consumed`。
    超了不会报错,只会算出一批零 embedding —— 本方案唯一的"静默错算"型
    风险,必须当场炸;
  - 出队判据是 `live_rows`(唯一权威 = 原生水位,不是第二本台账)。

注:`long_prefill_token_threshold` **不再被覆写**。v4 用它复现边侧 chunk
边界;v7 里云侧按"已收 + 预算"下发,与边侧怎么切块无关,所以阈值复现
(以及 `_lwd_preflight_ok` / `_lwd_verify_full_admission` / 块对齐)整套删除,
按"本方案就是唯一方案"的原则不留兼容分支。

前置约束:spec decode(MTP)预算已按 1+k 系数对齐(边 cap 与本类
preflight),但端到端未经真机验证(eagle 纯度判据失真风险,台账
P3-3/设计 M-8);
不兼容云侧抢占重算(占位 embeds 无法本地重 prefill)与 prefix caching
(命中使 num_computed 起点 >0,触发 offset 对账 fail-fast;且被命中
chunk 的 UP send 无人消费 = 通道打洞,chunk 化后硬前提,设计 §8-C3)
——部署须保证 KV 充足并关闭缓存,本类的核验只负责把违背变成当场
报错而非静默错算。
"""

from __future__ import annotations

import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_WIRE_VERSION,
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)

logger = init_logger(__name__)


class LwdCloudMixedScheduler(LwdBaseScheduler):
    """notify 门控的 prefill/decode 混排调度器(相位机的继任者)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # prefill 通知队列:边侧批量范围预告(RangeNotify)逐条入队,每步
        # 至多消费队首一条;预告自带 seqno 即本批 UP 链配对号
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        # [Lwd][sched] 调度批日志步计数(饿死分析:RangeNotify 到达 ->
        # 混排步消费的间隔与中间插入的 decode-only 步数)
        self._lwd_sched_step = 0
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 占预算与 KV);k = num_speculative_tokens(配置解析期已求值,
        # 缺省 None 按 0),无 spec 配置即 1
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        num_spec = (
            getattr(spec_config, "num_speculative_tokens", None) or 0
        )
        self._lwd_spec_factor: int = 1 + num_spec
        logger.info(
            "[Lwd] cloud mixed scheduler: chunked-prefill batches mixed "
            "with decode (one notify consumed per step at most, "
            "spec_factor=%d)",
            self._lwd_spec_factor,
        )
        # spec 接受率累计器(引擎进程内直出):LWD 云前端没有注册请求,
        # 前端 stats 链(output_processor 丢弃未知请求输出 + lifespan
        # 周期任务)在云侧不可靠;接受率数据的源头本来就在调度器里,
        # 此处按原生口径(scheduler.py:1414-1424)累计并周期直出,
        # 与集中式的 Prometheus 计数器/前端日志同公式可直接对比
        self._lwd_spec_drafts = 0
        self._lwd_spec_draft_tokens = 0
        self._lwd_spec_accepted = 0
        self._lwd_spec_last_log = time.monotonic()

        # ------------------------------------------------------------------ #
        # host 暂存调度(design v7 §5)                                        #
        # ------------------------------------------------------------------ #
        # 注:**不再有"已消费行数"台账**。v4 那本账与原生 num_computed_tokens
        # 是同一件事的第二份拷贝,两份就可能分歧,而分歧的表现是静默切错窗口。
        # v7 的权威只有一个:原生水位。已收行数则在引擎侧(lwd_recv_rows),
        # 由 worker 捎带上报。
        # 已 abort、但剩余行永远不会被算完的请求(从"活跃"里摘掉)。
        self._lwd_dropped: set[str] = set()
        # 本步待通知 worker 的 abort 名单(§6.4)。
        # 摘除台账只活在调度器进程里,worker 的暂存池看不到,所以必须随批
        # 捎带过去;每步排空(通知是幂等的,重复送达只会重复一个 no-op)。
        self._lwd_abort_pending: list[str] = []
        # 窗口预算基数:原生 token_budget 用的就是这个实例属性(§3.2.1 常量口径)。
        self._lwd_kv_block_size: int = max(int(getattr(self, "block_size", 1)), 1)
        logger.info(
            "[Lwd][host-stage] windowed consume armed: BS=%d base=%d",
            self._lwd_kv_block_size, self.max_num_scheduled_tokens,
        )

    # ------------------------------------------------------------------ #
    # KV 只读探针(照抄原生口径,不做任何分配)                             #
    # ------------------------------------------------------------------ #
    def _lwd_blocks_needed(self, request, num_tokens: int) -> int:
        """该请求算到 ``num_tokens`` 为止需要**新分配**多少 KV 块。

        直接复用原生的 ``coordinator.get_num_blocks_to_allocate`` —— **不**
        接着调 ``allocate_new_blocks``,所以是纯读探针(不占块、不改状态)。
        这是本方案唯一一处读原生内部函数的地方,口径必须与原生一致:

        * ``apply_admission_cap=False``:该参数是"全序列准入闸门"
          (``full_sequence_must_fit=True``)专用的;逐步分配路径必须保持
          False,否则预测值与 ``allocate_new_blocks`` 不一致
          (``kv_cache_coordinator`` 里写明的约束)。
        * ``new_computed_blocks=EMPTY``(首阶段不算前缀命中,§10.2):需求被
          高估 ⇒ 窗口偏小 ⇒ 只少算、不越界。running 请求在该函数里走快路径
          并断言无新命中,所以对 decode 是**精确**的。
        """
        coordinator = self.kv_cache_manager.coordinator
        empty = self.kv_cache_manager.empty_kv_cache_blocks.blocks
        total_computed = min(
            request.num_computed_tokens, self.max_model_len
        )
        num_tokens_main_model = total_computed + num_tokens
        return coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                num_tokens_main_model + self.num_lookahead_tokens,
                self.max_model_len,
            ),
            new_computed_blocks=empty,
            num_encoder_tokens=0,
            total_computed_tokens=total_computed,
            num_tokens_main_model=num_tokens_main_model,
            apply_admission_cap=False,
        )

    def _lwd_verify_w2(self, out, notify, windowed, windows, decode_exact) -> None:
        """下发前核验:**排程量不得超过已到达量**(design v7 §7 的 W2)。

        这是本方案唯一"静默错算"型风险的兜底。``num_scheduled_tokens``
        大于 ``recv_rows − consumed`` 时不会报错,只会让 runner 从 host 上
        读一段**还没被写入**的零值区,算出一批零 embedding —— 表现是"输出
        莫名其妙"而不是崩溃,极难定位。所以这里逐请求核验并把差异报全:

        1. 该请求本步是否真的被排了(排了才有下面的量);
        2. ``scheduled ≤ recv_rows − consumed``;
        3. 该请求是否**根本没到数据却被排**(``recv_rows == 0``)—— 那说明
           可见集谓词(§5.3.1b)失效,这是最严重的一档。
        """
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        for item, w in zip(windowed, windows):
            rid = item.request_id
            got = int(out.num_scheduled_tokens.get(rid, 0))
            if got == 0:
                continue
            request = self.requests.get(rid)
            if request is None:
                continue
            recv_rows = int(ledger.get(rid, 0))
            consumed = request.num_computed_tokens - item.offset
            avail = recv_rows - consumed
            if recv_rows <= 0:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid} seqno="
                    f"{notify.seqno}: scheduled {got} tokens but the data "
                    f"plane reported recv_rows=0 (该请求不该进 batch —— 可见集"
                    f"谓词 `recv_rows > 0` 失效)"
                )
            if got > avail:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid} seqno="
                    f"{notify.seqno}: native scheduled {got} rows but only "
                    f"{avail} have arrived (recv_rows={recv_rows} "
                    f"consumed={consumed} window={w} prompt={item.num_tokens} "
                    f"offset={item.offset} decode_exact={decode_exact})"
                )

    def _lwd_kv_rows_capacity(self, decode_blocks: int) -> int:
        """扣掉 decode 之后 prefill 还能用多少行(decode 优先,§3.2.1)。"""
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return max(free - decode_blocks, 0) * self._lwd_kv_block_size

    def update_from_output(self, scheduler_output, model_runner_output):
        """先累计 spec 接受率(镜像原生 update_from_output 的统计分支),
        再走原生输出处理。"""
        self._lwd_observe_spec_acceptance(scheduler_output, model_runner_output)
        return super().update_from_output(scheduler_output, model_runner_output)

    _LWD_SPEC_LOG_INTERVAL_S = 10.0

    def _lwd_observe_spec_acceptance(
        self, scheduler_output, model_runner_output
    ) -> None:
        """逐请求累计草稿/接受数并周期打运行总计。

        口径与原生完全一致:num_draft_tokens = len(scheduled_spec_token_ids),
        num_accepted = len(generated_token_ids) - 1(含 bonus,减一剔除);
        跳过已终结/未知请求(与原生 :1403-1412 同款过滤)。"""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            return
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index
        for req_id, spec_ids in spec_tokens.items():
            if not spec_ids:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = sampled[req_index] if sampled else []
            if not generated:
                continue
            self._lwd_spec_draft_tokens += len(spec_ids)
            self._lwd_spec_accepted += len(generated) - 1
            self._lwd_spec_drafts += 1
        now = time.monotonic()
        if now - self._lwd_spec_last_log < self._LWD_SPEC_LOG_INTERVAL_S:
            return
        self._lwd_spec_last_log = now
        if not self._lwd_spec_draft_tokens:
            return
        rate = self._lwd_spec_accepted / self._lwd_spec_draft_tokens * 100
        mean_accept_len = 1 + self._lwd_spec_accepted / max(
            self._lwd_spec_drafts, 1
        )
        logger.info(
            "[Lwd][spec] cumulative: rate=%.1f%% mean_accept_len=%.2f "
            "accepted=%d draft_tokens=%d drafts=%d",
            rate, mean_accept_len, self._lwd_spec_accepted,
            self._lwd_spec_draft_tokens, self._lwd_spec_drafts,
        )

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样)。chunk 化后 running 可含
        prefill 半途请求(num_computed 爬坡中)——它们不进 decode 集,
        留在隐藏集等自己的下一条 chunk 预告驱动续排。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _lwd_collect_decode_requests(self) -> list[str]:
        """收集所有 decode 态(prompt 已算完)请求的 req_id。

        按 running/waiting/skipped 顺序遍历三队列,输出可直接作为
        _lwd_schedule_for_visible_reqs 的入参。"""
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._lwd_is_decode(req)
        ]

    # ------------------------------------------------------------------ #
    # notify 条目分类与 preflight                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lwd_notify_items(notify: LwdRangeNotify) -> list[LwdRangeItem]:
        """取批条目:优先 items(mixed 组批);空则回退旧版单请求语义。"""
        if notify.items:
            return list(notify.items)
        return [
            LwdRangeItem(
                request_id=notify.request_id,
                offset=notify.offset,
                num_tokens=notify.num_tokens,
            )
        ]

    def _lwd_classify_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """条目三分:ready(待 prefill)/ stale(已完结或重复预告,剔除)
        / unknown(ADD 尚未落地)。返回 (ready, 是否有 unknown)。

        unknown 存在即整批 defer(all-or-nothing:一次 UP recv 对应整批
        张量,不能只放行子集);abort 清扫(引擎侧)负责把永不落地的
        条目摘除,队首不会被永久堵住。"""
        ready: list[LwdRangeItem] = []
        unknown = False
        for item in items:
            request = self.requests.get(item.request_id)
            if request is None:
                unknown = True
                continue
            if self._lwd_is_decode(request):
                logger.warning(
                    "[Lwd][cloud-sched] drop stale notify item req=%s "
                    "num=%d (already prefilled; duplicate notify?)",
                    item.request_id, item.num_tokens,
                )
                continue
            ready.append(item)
        return ready, unknown

    def _lwd_pre_recv_ready(self, notify: LwdRangeNotify) -> bool:
        """提前收闸门:该批(本 chunk 批)的 UP 数据收完了没。

        收完才允许下发这个 prefill 批 —— 此时 embeds(主帧)与 mrope(aux 帧)
        都已在云 worker 的接收 buffer 里,注入侧 future.wait() 立刻返回;没收完
        就本步只下 decode(notify 留队首,下步再判),不把等待塞进 worker 前向。

        判据是引擎维护的每通道水位(worker 通信线程上报收完的 seqno):完成序
        = seqno 序(通道 FIFO),所以 seqno <= 水位 即精确判据。一条 seqno 覆盖
        主帧 + aux 帧(``LwdCommRequest`` 的 aux 约定),故水位天然覆盖两帧。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            is_irecv_complete,
            watermark,
        )
        from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

        ready = is_irecv_complete(LwdChannelType.UP, notify.seqno)
        # [Lwd][pre-recv] 闸门读数:与 HINT/SUBMIT/DONE/REPORT/POST 用同一单调钟,
        # 可把"预告到达->worker挂recv->worker收完上报->引擎放行"四段直接相减。
        logger.info(
            "[Lwd][pre-recv] gate seqno=%d watermark=%d ready=%s ts=%.3f",
            notify.seqno,
            watermark(LwdChannelType.UP),
            ready,
            time.monotonic(),
        )
        return ready

    # 注:``_lwd_preflight_ok``(整批三检)与 ``_lwd_verify_full_admission``
    # ("禁止部分准入")已按 §3.3.2 删除。它们的三项检查各有新落点:
    #   * token 预算 -> 并入 ``_lwd_windowed_admission`` 的 budget 公式;
    #   * KV 块数   -> 行级逐 item 算,且用原生口径(``_lwd_blocks_needed``);
    #   * 并发名额  -> 请求级,由原生准入在 item 首次进 running 时查一次;
    #   * 步后核验  -> "排程量 == 本步窗口"(§3.4 W2,在 _schedule_mixed 内)。

    # ------------------------------------------------------------------ #
    # 调度主体(见 docs/design/lwd_host_staging.md §5)                     #
    # ------------------------------------------------------------------ #
    def _schedule_impl(self) -> SchedulerOutput:
        # (2) 收集 decode 集(判据:prompt 已算完 = 可采样)。窗口化后 running
        #     里会多出"chunk 算到一半"的请求,判据天然把它们排除,它们由预告
        #     队列驱动续排 —— 与改造前一致。
        decode_ids = self._lwd_collect_decode_requests()
        # (3) 取队首 notify 并分类
        notify = (
            self.prefill_notify_queue[0] if self.prefill_notify_queue else None
        )
        if notify is not None:
            items, unknown = self._lwd_classify_items(
                self._lwd_notify_items(notify)
            )
            # ---- 可见集谓词:数据面没到行的请求**不进本步 batch** ----
            # (design v7 §5.3.1b,提问方指定)。落点是"构造可见集时多一个
            # 条件"——复用既有的 _lwd_schedule_for_visible_reqs 机制,
            # **原生 scheduler.py 一行不改**。
            #
            # 为什么必须先挡住:云侧的 prompt_embeds 是 torch.zeros 占位,
            # 一个 recv_rows==0 的请求若被组进 batch,原生会照常给它排行,
            # 而 runner 只会读到零值区 ⇒ **算出一批零 embedding 且不报错**。
            # 只挡"完全没数据"这一档就够了:到了但不够预算,由下发规则
            # (min(avail, budget))处理。
            ledger = getattr(self, "lwd_recv_rows", None) or {}
            gated: list = []
            for item in items:
                rid = item.request_id
                recv_rows = int(ledger.get(rid, 0))
                request = self.requests.get(rid)
                if request is None:
                    gated.append(item)
                    continue
                if recv_rows - request.num_computed_tokens <= 0:
                    # recv_rows == 0(完全没到)= 等待;已到但算完 =
                    # 该项其实已交付,下一处 all_done 逻辑会消费掉队首。
                    continue
                gated.append(item)
            if len(gated) != len(items):
                logger.info(
                    "[Lwd][host-stage] GATE seqno=%d visible=%d/%d "
                    "(recv_rows>0 谓词挡掉 %d 项)",
                    notify.seqno, len(gated), len(items),
                    len(items) - len(gated),
                )
            items = gated
            live = self._lwd_live_rows(notify) if items or unknown else 0
            if not items and not unknown:
                # 全部条目失效(完结/重复):弹队,本步纯 decode。
                # 按对象身份弹(abort 清扫可能整体替换过 deque,见
                # _schedule_mixed 同款处理)
                queue = self.prefill_notify_queue
                if queue and queue[0] is notify:
                    queue.popleft()
                notify = None
            elif live == 0:
                # (§3.6 情形 D)条目还在但**剩余活行为 0** —— 全部被 abort
                # 或已消费完:notify 的这一批没有任何东西可算了,必须弹队,
                # 否则它作为队首会把后面所有 notify 永久钉死(今天是同形态
                # 死锁的真机实录)。
                queue = self.prefill_notify_queue
                if queue and queue[0] is notify:
                    queue.popleft()
                logger.info(
                    "[Lwd][host-stage] notify seqno=%d released: live_rows=0 "
                    "(all items consumed or aborted)",
                    notify.seqno,
                )
                notify = None
            elif unknown:
                # 请求 ADD 未落地(input_queue 次序滞后):预告留队首,
                # 本步先 decode;abort 清扫兜底防永久堵队首
                logger.info(
                    "[Lwd][cloud-sched] notify seqno=%d deferred: "
                    "request(s) not added yet",
                    notify.seqno,
                )
                notify = None
            elif not self._lwd_pre_recv_ready(notify):
                # 提前收闸门:UP 张量还没收完 -> 本步不下发这个 prefill 批
                # (notify 留队首,下步再判),本步只下 decode。收完的批优先,
                # 没收完的批一直让 decode 自转,不把等待塞进 worker 前向。
                notify = None
            else:
                # (4)(5) decode 的 token/块账 -> budget -> 逐 item 求窗口。
                # 取代原来的 _lwd_preflight_ok 整批三检:三检里的"token 预算"
                # 并入 budget 公式,"KV 块数"改成行级逐 item 算,"并发名额"
                # 是请求级的、由原生准入在首次进 running 时查一次。
                windows, windowed, decode_exact, _budget, reason = (
                    self._lwd_windowed_admission(notify, items, decode_ids)
                )
                if not windows:
                    # 窗口全空(预算/KV 都不够):预告留队首,decode 泄压后
                    # 重试。这条必须打日志 —— 它是"数据收完了、闸门也放行了,
                    # prefill 却一直不下发"的唯一静默出口。
                    logger.info(
                        "[Lwd][cloud-sched] notify seqno=%d deferred: no "
                        "window (reqs=%d tokens=%d decode=%d pending=%d "
                        "reason=%s)",
                        notify.seqno, len(items),
                        sum(item.num_tokens for item in items),
                        len(decode_ids), len(self.prefill_notify_queue),
                        reason,
                    )
                    notify = None
                else:
                    # (6)(7)(8) 一起在 _schedule_mixed 里完成:覆写预算调原生、
                    # 步后核验、提交台账并写 LwdEmbedBatch。
                    self._lwd_note_window_step(notify)
                    out = self._schedule_mixed(
                        notify, windowed, decode_ids, windows, decode_exact
                    )
                    self._lwd_attach_abort_notice(
                        out, self._lwd_take_abort_pending()
                    )
                    return out
        out = self._lwd_schedule_for_visible_reqs(decode_ids)
        self._lwd_attach_abort_notice(out, self._lwd_take_abort_pending())
        return out

    def _lwd_decode_tokens(self, decode_ids: list[str]) -> tuple[int, dict]:
        """本步 decode 逐请求的 token 消耗(原推导逐字保留,只改成"留账")。

        预算钳制的精确性依赖逐字复刻原生 running 段(scheduler.py:381-416)
        的 num_new 推导——1+len(spec) 不是准确式:
        ① placeholder(spec 拒绝遗留的非零值)计入排程窗口;
        ② max_tokens 守卫(占位>0 且距 max_tokens 不足 2)与
          next_decode_eligible_step 会让请求整步跳过、消耗 0。
        不逐字复刻就会留下余量,使末位截断块被超排(真机实录:
        decode 请求被守卫跳过消耗 0,末位块 604 被排成 608)。

        返回 ``(decode_exact, {req_id: d_i})``。**同一份 d_i 同时喂给
        token 账与 KV 块账**(§6.1 第 4 步)——这是"两个口径逐请求一致"
        的落地方式:不存在"扣了 token 没扣块"的漏算。
        """
        per_req: dict[str, int] = {}
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue  # 原生 max_tokens 守卫:本步跳过,消耗 0
            if self.current_step < req.next_decode_eligible_step:
                continue  # V2+PP 节拍守卫:本步跳过,消耗 0
            d_i = max(
                0,
                min(
                    req.num_tokens_with_spec
                    + placeholders
                    - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
            if d_i <= 0:
                continue
            per_req[rid] = d_i
            decode_exact += d_i
        return decode_exact, per_req

    def _lwd_windowed_admission(self, notify: LwdRangeNotify, items, decode_ids):
        """第 (4)(5) 步:算 decode 的 token/块账,再逐 item 求**本步下发量**。

        下发规则(design v7 §5.3,提问方指定):

            avail_i = recv_rows[i] − consumed_i          # 数据面到了多少
            n_i     = min(avail_i, 剩余预算)              # 与预算取小

        ``recv_rows`` 是 worker 捎带上报的**已收行数**(§5.2),
        ``consumed`` 直接取原生水位 ``request.num_computed_tokens −
        item.offset``(唯一权威,不再维护第二本账)。

        **为什么不按 ``item.num_tokens``(chunk 剩余)封顶**:那样预算就依赖
        "原生批处理次序与我们一致"。只按 ``avail`` 封顶时,
        ``budget = decode_exact + Σavail_i`` 与原生 FIFO 前缀序组合出的结果
        **精确**等于逐请求取小;万一序不一致,原生最多排到 ``Σavail``,
        而 W2 会在下发后当场炸 —— 不会静默算脏数据(§7)。

        返回 ``(windows, windowed_items, decode_exact, budget, reason)``;
        ``windows`` 为空表示本步不排 prefill(纯 decode)。
        """
        BS = self._lwd_kv_block_size
        decode_exact, per_req = self._lwd_decode_tokens(decode_ids)

        # decode 先占 KV(§5.6):块账用原生函数逐请求算,不用粗估。
        decode_blocks = 0
        for rid, d_i in per_req.items():
            req = self.requests[rid]
            decode_blocks += self._lwd_blocks_needed(req, d_i)

        # 本步可用的 KV 行数;预算同时受"单步 token 上限 − decode"约束
        kv_rows = self._lwd_kv_rows_capacity(decode_blocks)
        cfg_room = self.max_num_scheduled_tokens - decode_exact
        room = min(cfg_room, kv_rows)
        if room < 1:
            logger.info(
                "[Lwd][host-stage] WINDOW seqno=%d empty: room=%d "
                "(decode_exact=%d decode_blocks=%d free=%d)",
                notify.seqno, room, decode_exact, decode_blocks,
                self.kv_cache_manager.block_pool.get_num_free_blocks(),
            )
            return [], [], decode_exact, 0, "budget"

        ledger = getattr(self, "lwd_recv_rows", None) or {}
        windows: list[int] = []
        windowed: list[LwdRangeItem] = []
        capped_by_budget = False
        for item in items:
            rid = item.request_id
            if rid in self._lwd_dropped:
                continue
            request = self.requests.get(rid)
            if request is None:
                continue
            consumed = request.num_computed_tokens - item.offset
            recv_rows = int(ledger.get(rid, 0))
            avail = recv_rows - consumed
            if avail <= 0:
                # 数据面还没到(或这一段已算完):**本步不给它任何行**。
                # avail == 0 的请求根本不该进 batch —— 那由可见集谓词挡掉
                # (§5.3.1b);这里挡的是"到了但又算完了"。
                continue
            n = min(avail, room)
            if n < avail:
                capped_by_budget = True
            if n < 1:
                break  # 保序:后面的一并留到下步
            windows.append(n)
            windowed.append(item)
            room -= n
            if room < 1:
                break
        reason = (
            "none" if not windows
            else ("kv" if kv_rows <= cfg_room else "cfg")
        )
        if capped_by_budget:
            reason = f"{reason}+budget"
        logger.info(
            "[Lwd][host-stage] WINDOW seqno=%d windows=%s avail=%s "
            "room_left=%d decode_exact=%d decode_blocks=%d free=%d reason=%s",
            notify.seqno, windows,
            [
                int(ledger.get(it.request_id, 0))
                - (self.requests[it.request_id].num_computed_tokens - it.offset)
                for it in windowed
            ],
            room, decode_exact, decode_blocks,
            self.kv_cache_manager.block_pool.get_num_free_blocks(), reason,
        )
        return windows, windowed, decode_exact, room, reason

    def _schedule_mixed(
        self,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
        decode_ids: list[str],
        windows: list[int],
        decode_exact: int,
    ) -> SchedulerOutput:
        """混排步:decode 集 + 本批**本步可取的行**同批下发。

        offset 对账:item.offset 必须 == 云侧当前 num_computed(async
        关闭前提下调度时刻即真实水位)——不等即 chunk 断流(>)/
        重复(<)/乱序,协议级事故 fail-fast,不当场炸就会把
        embeddings 注入错位窗口静默错算。

        **按请求下发(design v7 §5.3/§5.4)**:本步每个请求排多少行由调用方
        按"已收多少 + 预算"算好(``windows``),这里只负责

          1. 把 ``max_num_scheduled_tokens`` 压到
             ``decode_exact + Σwindows`` 后交给原生 —— 这是本方案
             **唯一**的逐请求上界手段(原生没有 per-request 钩子),配合
             原生 FIFO 前缀序精确等于逐请求取小;
          2. **下发前 W2 核验**:逐请求
             ``num_scheduled_tokens ≤ recv_rows − consumed`` —— 超了就是
             读 host 上的零值区、静默算出一批零 embedding,必须当场炸;
          3. 并按 ``live_rows``(唯一权威 = 原生水位)决定 notify 是否出队。
        """
        logger.info(
            "[Lwd][cloud-sched] mixed notify seqno=%d reqs=%s tokens=%d "
            "windows=%s",
            notify.seqno, [item.request_id for item in items],
            sum(item.num_tokens for item in items), windows,
        )
        for item in items:
            if item.has_mrope and not item.mrope_positions:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] mrope payload missing "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"has_mrope=True 但控制面未携带 positions 字节——边侧"
                    f"仍按数据面 aux 帧协议运行(wire_version<"
                    f"{LWD_WIRE_VERSION}),两侧必须同版本部署"
                )
            request = self.requests[item.request_id]
            if request.num_computed_tokens != item.offset:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] chunk offset mismatch "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"notify offset={item.offset} != cloud computed="
                    f"{request.num_computed_tokens} "
                    f"(chunk 断流/重复/乱序)"
                )
        req_ids = decode_ids + [item.request_id for item in items]
        saved_budget = self.max_num_scheduled_tokens
        # 把单步 token 预算压到 decode 的**精确**需求 + 本步 prefill 可取的行数。
        # 这是本方案唯一的"逐请求上界"手段:原生没有 per-request 钩子
        # (num_new_tokens = request.num_tokens − num_computed,只被
        # long_prefill_token_threshold 与 token_budget 削),但
        # `Σmin(avail_i, room) + decode_exact` 与原生 FIFO 前缀序组合出的
        # 结果**精确**等于逐请求取小 —— 只要原生不会"跳过前面的请求去排后面的"。
        # 万一它真那样排,下面的 W2 会当场炸,而不是静默算脏数据。
        self.max_num_scheduled_tokens = decode_exact + sum(windows)
        try:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        finally:
            self.max_num_scheduled_tokens = saved_budget
        # ---- W2(design v7 §7):本方案唯一"静默错算"型风险的兜底 ----
        # 下发的行数绝不能超过**已到达**的行数:超了就会读 host 上还没写过的
        # 零值区,算出一批零 embedding 而**不报任何错**。这里逐请求核验
        #     native_scheduled ≤ recv_rows − consumed
        # 并把"原生 prefill 总量 == Σwindows"一并核验(多排即越界)。
        self._lwd_verify_w2(out, notify, windowed, windows, decode_exact)
        prefill_ids = [item.request_id for item in items]
        native_prefill = sum(
            int(out.num_scheduled_tokens.get(rid, 0)) for rid in prefill_ids
        )
        if native_prefill != sum(windows):
            raise RuntimeError(
                f"[Lwd][cloud-sched] W2 violated seqno={notify.seqno}: "
                f"native prefill={native_prefill} Σwindows={sum(windows)}"
            )
        # 消费队首:abort 清扫(IO 线程)可能整体替换过 deque,按对象
        # 身份弹队首,避免错弹替换后的新队首。
        # 判据是"该批所有请求都算完了"(v7:不再有窗口台账,唯一权威是原生
        # 水位 —— 数据没到的行本步根本不会被下发)。
        all_done = all(
            self.requests[item.request_id].num_computed_tokens
            >= item.offset + item.num_tokens
            for item in items
            if item.request_id in self.requests
        )
        if all_done:
            queue = self.prefill_notify_queue
            if queue and queue[0] is notify:
                queue.popleft()
            for item in items:
                self._lwd_dropped.discard(item.request_id)
        else:
            logger.info(
                "[Lwd][host-stage] WINDOW partial seqno=%d consumed=%s "
                "steps=%d",
                notify.seqno,
                {
                    it.request_id: (
                        self.requests[it.request_id].num_computed_tokens
                        - it.offset
                    )
                    for it in items if it.request_id in self.requests
                },
                self._lwd_window_steps(notify),
            )
        # UP 链 seqno 随批下发云 worker:批配对号直接取预告自带 seqno(与
        # 边侧 EMBED 批派发号同源同值)。batch_meta 承载 worker 的 recv
        # 尺寸与注入切行信息:req_ids 取批内逐请求,token_ids 为占位
        # 列表——长度必须等于边侧实际发送的逐请求 token 数,recv numel
        # 才能与边侧 isend 严格相等(HCCL P2P 要求两端 numel 匹配)。
        # prompt_offsets/has_mrope/mrope_positions 逐 item 透传:runner
        # 对 MM 请求把 positions 字节 frombuffer 直写 req_state(文本
        # 请求空 bytes、零路径变化);数据面 aux 帧已退役。
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in items],
                token_ids=[[0] * item.num_tokens for item in items],
                prompt_offsets=[item.offset for item in items],
                has_mrope=[item.has_mrope for item in items],
                mrope_positions=[item.mrope_positions for item in items],
            ),
        )
        return out

    # ------------------------------------------------------------------ #
    # 出队判据(§6.2:被 abort 的请求剩余行永远不会被消费)                    #
    # ------------------------------------------------------------------ #
    def _lwd_live_rows(self, notify: LwdRangeNotify) -> int:
        """该 notify 里还没算完、且未被 abort 的行数。

        **唯一权威是原生水位** ``request.num_computed_tokens``(design v7):
        v4 那本"已消费行数"台账已删除 —— 它与原生水位是同一件事的第二份
        拷贝,两份就可能分歧,而分歧的表现是静默切错窗口。这里改成直接问
        原生:``剩余 = offset + num_tokens − num_computed_tokens``。

        被 abort(进了 ``_lwd_dropped``)的请求其剩余行**永远不会**被算完,
        所以不计入 —— 这正是"用 consumed==total 判出队会把队首钉死"的那个
        坑(§6.2)。已算完的行不回滚:那是历史事实。
        """
        total = 0
        for item in self._lwd_notify_items(notify):
            rid = item.request_id
            if rid in self._lwd_dropped:
                continue
            request = self.requests.get(rid)
            if request is None:
                continue
            total += max(
                item.offset + item.num_tokens - request.num_computed_tokens,
                0,
            )
        return total

    def _lwd_window_steps(self, notify: LwdRangeNotify) -> int:
        """该 notify 已被下发过几步(诊断:steps 大 = KV 长期紧)。"""
        return int(getattr(notify, "_lwd_window_steps", 0))

    def _lwd_note_window_step(self, notify: LwdRangeNotify) -> None:
        notify._lwd_window_steps = self._lwd_window_steps(notify) + 1

    def _lwd_drop_request_from_chunks(self, request_id: str) -> int:
        """把一个被 abort 的请求从逐请求台账里摘成"永不消费"。

        返回它被丢弃的剩余行数。已算完的行是历史事实(KV 已算、embeddings
        已注入),不回滚;只把剩余行冻结,``live_rows`` 才能归零(§6.2)。

        同时把它排进"待通知 worker"名单:worker 的暂存池是按请求管存储的,
        这些剩余行在 host 上还占着 pinned,必须让 worker 也冻结 —— abort 时
        worker 只冻结,真正的释放在请求结束那条路上做(design v7 §6.1/§6.3)。
        """
        already = request_id in self._lwd_dropped
        self._lwd_dropped.add(request_id)
        remaining = 0
        for notify in self.prefill_notify_queue or ():
            for item in self._lwd_notify_items(notify):
                if item.request_id != request_id:
                    continue
                request = self.requests.get(request_id)
                if request is not None:
                    remaining += max(
                        item.offset + item.num_tokens
                        - request.num_computed_tokens,
                        0,
                    )
        if remaining:
            logger.info(
                "[Lwd][host-stage] ABORT-PARTIAL req=%s dropped_rows=%d",
                request_id, remaining,
            )
        if not already:
            # 只在首次摘除时排队通知(重复送达也只是幂等 no-op,但名单里
            # 每步只该出现新摘除的请求)
            self._lwd_abort_pending.append(request_id)
        return remaining

    def _lwd_take_abort_pending(self) -> list[str]:
        """取出本步要通知 worker 的 abort 名单(取走即清,幂等靠 worker 侧)。"""
        pending = self._lwd_abort_pending
        if not pending:
            return []
        self._lwd_abort_pending = []
        return pending

    def _lwd_attach_abort_notice(self, out: SchedulerOutput, aborted: list[str]) -> None:
        """把 abort 名单挂到本步输出上(§3.6.1)。

        挂在哪取决于本步有没有 EMBED 批:
          * 有批 -> 挂在该批的 ``batch_meta.aborted_req_ids``(搭已有通道,
            不新增协议);
          * 没批(纯 decode / 空步)-> 退化挂一个**只带 abort 名单**的
            EMBED 批:``req_ids`` 为空,所以 worker 的 recv 路径与 runner 的
            注入路径都不会碰它(两边都以 ``req_ids`` 为空即返回),
            只有 abort 处理会读它。**这一步是必需的** —— 否则"请求被 abort
            且当时没有批要下发"时,pinned 就没人回收了。

        **名单绝不丢弃**:挂不上(已有非 EMBED 批)时放回待发队列,下一步再送
        —— pinned 回收是滞后容忍的,漏送则不是。
        """
        if not aborted:
            return
        batch = getattr(out, "lwd_batch", None)
        if batch is None:
            out.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=-1,
                batch_meta=LwdEmbedBatch(aborted_req_ids=list(aborted)),
            )
            return
        if (
            batch.batch_type is LwdBatchType.LWD_EMBED
            and getattr(batch, "batch_meta", None) is not None
        ):
            # 与已有名单合并(同一步可能既有新摘除又有积压)
            existing = list(batch.batch_meta.aborted_req_ids or ())
            batch.batch_meta.aborted_req_ids = existing + [
                r for r in aborted if r not in existing
            ]
            return
        # 已有非 EMBED 批:不动它,名单放回队列等下一步
        # (取走时已清空,这里补回;顺序无所谓,通知是幂等的)
        self._lwd_abort_pending = list(aborted) + self._lwd_abort_pending

    def schedule(self) -> SchedulerOutput:
        # [Lwd][perf] 云侧每步 LWD 税分段之一:调度(容器交换)时长
        _t = time.monotonic()
        self._lwd_settle_deferred_aborts()
        out = self._schedule_impl()
        self._lwd_sched_step += 1
        self._lwd_log_sched_batch(out)
        logger.info(
            "[Lwd][perf] cloud-sched dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        return out

    def _lwd_log_sched_batch(self, out: SchedulerOutput) -> None:
        """[Lwd][sched] 每步调度批结构化日志(lwd_backlog_probe ⑤ 段锚点)。

        phase 判定:lwd_batch=EMBED 且批内还有 decode 请求即 MIXED;EMBED
        独占为 PREFILL;有排程 token 为 DECODE;否则 EMPTY。pending_notify
        给出待吃量——EMPTY/DECODE 连跑且 pending_notify>0 = 通知已到但
        preflight 不过或 ADD 未落地。reqs 超 12 个截断,防大 decode 批
        刷屏。"""
        batch = getattr(out, "lwd_batch", None)
        if batch is not None and batch.batch_type == LwdBatchType.LWD_EMBED:
            n_prefill = len(batch.batch_meta.req_ids)
            n_reqs = len(out.num_scheduled_tokens)
            phase = "MIXED" if n_reqs > n_prefill else "PREFILL"
            seqno: object = batch.seqno
        else:
            phase = "DECODE" if out.total_num_scheduled_tokens else "EMPTY"
            seqno = ""
        reqs = list(out.num_scheduled_tokens)
        reqs_str = ",".join(reqs[:12]) + (
            f",+{len(reqs) - 12}more" if len(reqs) > 12 else ""
        )
        logger.info(
            "[Lwd][sched] cloud step=%d phase=%s seqno=%s reqs=[%s] tokens=%d "
            "pending_notify=%d decode_ready=%d",
            self._lwd_sched_step, phase, seqno, reqs_str,
            out.total_num_scheduled_tokens, len(self.prefill_notify_queue),
            len(self._lwd_collect_decode_requests()),
        )
