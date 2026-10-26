"""云侧混合调度器:prefill/decode 同批下发,**arrival 驱动**(design v8)。

边侧按 c = min(剩余 prompt, chunk 单元) 切块组批发 RangeNotify,但
云侧调度**不再消费预告队列** —— 预告只驱动数据面提前收(引擎下发
recv hint);调度器每步对"已收未算完"的请求直接开窗:

  - **可见集谓词**:`recv_rows − num_computed <= 0` 的请求本步不进 batch
    (数据面没到就算 = 读零占位、静默错算);recv_rows 是 worker 在分段
    D2H 完成后随水位捎带上报的逐请求累计高水位;
  - **按请求下发**:每请求本步行数 = `min(已收 − 已算, 剩余预算)`,通过
    挂在请求对象上的 `lwd_window_cap` 属性交给原生 —— 原生 running/
    waiting 循环算 `num_new_tokens` 时对它取 min(总预算
    `max_num_scheduled_tokens = decode_exact + Σ该值` 压缩保留为总量
    上界);chunk 化从此只是边侧的流量整形;
  - **攒满批**:数据未收完且窗口不足满窗(数据限速的残余窗)时不开
    窗,攒到与下一块合并 —— 消除"碎窗独占一步"的固定开销尾巴;
    末段(已收 == prompt 全长)不受此限,保证请求必然算完;
  - **下发前 W2 核验**:逐请求 `start + got ≤ recv_rows` 且 `got ≤ 窗口`
    (有 cap 在恒真,纯看门狗);KV/预算截断导致的**少排**是合法行为,
    只告警不核验,下一步按原生水位重算;
  - KV/并发名额/块对齐全部交还原生。

前置约束:spec decode(MTP)的 decode 预算按 decode_exact 逐请求精确
记账(草稿 token 同样占预算),但端到端未经真机验证(eagle 纯度
判据失真风险,台账 P3-3/设计 M-8);
云侧抢占重算:已支持本地重 prefill —— pinned host 载荷活到请求结束
(design v7 §6.1),抢占把 num_computed 归零后 recv_rows 台账不动,
可见集谓词自然对该请求重开整条 prompt 的窗口,runner 在 harvest 前
把 stager 读指针回退到调度水位(``_lwd_rewind_staged_reads``),注入
几何与首算逐字相同;被抢请求拼回 waiting 队首(原生语义)。调度侧
每步多付的只是一次 re-prefill 的预算。
prefix caching(wire v4 起支持):命中边界经协商定死——边侧预告真实
内容的块哈希链,云侧建请求时 reserve(钉住命中块 + fail-closed 快照)
并回 READY,边侧自 num_computed=hit 起发送剩余段;原生 allocate_slots
经 kv_cache_manager 的 LWD 钩子把命中块挂进请求 KV 链,可见性谓词中
被命中的 [0, H) 零值行因 num_computed 起点 = H 而永不进窗。"零占位
token ids 哈希跨请求同值"的风险由两点消除:hasher 优先采用 wire 预告
链(云引擎),链长不符的协商请求当场回 error 拒绝准入。
抢占重准入(prefix caching 开启时)走统一路径(设计 §5.3):可见集
组装前对被抢占请求只读重查 H',H'≥暂存地板 S 直接按快照续算
(H'>S 为 self-hit),H'<S 发 LwdRefillNotify 由边侧补发缺失行并挂
refill 闸门(``_lwd_refill_gate``,可见性谓词前短路),补发完成
(``lwd_refill_done``)后摘闸准入。"""

from __future__ import annotations

import itertools
import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    prefix_start,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)

logger = init_logger(__name__)


class LwdCloudMixedScheduler(LwdBaseScheduler):
    """arrival 驱动的 prefill/decode 混排调度器(相位机的继任者)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 保留一个恒空的预告队列占位:基类的 abort 延迟终结机制会读它
        # (恒空 ⇒ abort 立即终结);引擎自 wire_version 3 起不再入队
        # (调度按 arrival 驱动,RangeNotify 只驱动数据面提前收)。
        self.prefill_notify_queue: deque = deque()
        # [Lwd][sched] 调度批日志步计数
        self._lwd_sched_step = 0
        logger.info(
            "[Lwd] cloud mixed scheduler: arrival-driven prefill mixed "
            "with decode (no notify queue)"
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
        # 前缀协商(wire v4):开关读配置级,与 HELLO 通告同源;预留表由
        # 引擎 init_lwd_prefix_cache 装配(scheduler.lwd_prefix_reservations)。
        self._lwd_prefix_enabled = bool(
            self.vllm_config.cache_config.enable_prefix_caching
        )
        # refill 闸门(设计 §5.3):抢占重准入重查 H' < 暂存地板 S 时挂起,
        # 补发完成(lwd_refill_done)摘除;门内请求可见性谓词前短路(不开窗)。
        self._lwd_refill_gate: set[str] = set()
        # rid -> 钉住的重查命中 H'(refill 完成时把暂存地板推进到 H')
        self._lwd_refill_pending: dict[str, int] = {}
        logger.info(
            "[Lwd][host-stage] windowed consume armed: base=%d",
            self.max_num_scheduled_tokens,
        )

    def _lwd_verify_w2(
        self, out, notify, windowed, windows, decode_exact,
        eff_starts=None,
    ) -> None:
        """下发前核验:**排程量不得超过已到达量**(design v7 §7 的 W2)。

        这是本方案唯一"静默错算"型风险的兜底。``num_scheduled_tokens``
        超出已到达量时不会报错,只会让 runner 从 host 上读一段**还没被写入**
        的零值区,算出一批零 embedding —— 表现是"输出莫名其妙"而不是崩溃,
        极难定位。所以这里逐请求核验并把差异报全:

        1. 该请求是否**根本没到数据却被排**(``recv_rows == 0``)—— 那说明
           可见集谓词(§5.3.1b)失效,这是最严重的一档;
        2. 本步排程量是否落在**已到达**的行数之内;
        3. 本步排程量是否越过**窗口**(``got > w``)—— 即 ``lwd_window_cap``
           钩子失守。越过窗口 = 越过本 chunk 边界,runner 的注入几何
           (mrope 按 chunk_rows 切窗)会静默错乱,而逐请求"已到达"核验
           不一定拦得住(后续 chunk 可能已随累计水位到达,真机实录
           seqno=101)。窗口判据在 cap 机制正常时恒真,是纯看门狗。

        **判据里绝不能再出现"已消费行数"**(真机踩过):
        ``consumed`` 是 **worker 侧的存储读指针**,由 ``_lwd_stage_requests_to_device``
        在同一步的 harvest 阶段推进,而 harvest 跑在 ``_prepare_inputs`` 内、
        **早于**本核验。拿它做减法,调度器看到的必然是"已经搬完了"的
        值 ⇒ ``avail == 0`` ⇒ 第一批就被误判成越界(真机实录:
        ``recv_rows=15 consumed=15`` 却报 "only 0 have arrived")。

        "取数区间必须落在已写入范围内"是**另一件事**,由 worker 的 stager
        几何校验负责(``LwdReqEmbed.read_range``,见
        ``_pre_recv_host_stage_check.py`` §5)。两处判据分开,各自看自己
        能权威看到的量 —— 这也是 §7 表格里的分工。
        """
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        for k, (item, w) in enumerate(zip(windowed, windows)):
            rid = item.request_id
            got = int(out.num_scheduled_tokens.get(rid, 0))
            if got == 0:
                continue
            if got > w:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid} seqno="
                    f"{notify.seqno}: native scheduled {got} rows beyond "
                    f"the window w={w} (lwd_window_cap 钩子失守 —— 越过本 "
                    f"chunk 边界会破坏 runner 的注入几何,宁可当场炸)"
                )
            recv_rows = int(ledger.get(rid, 0))
            if recv_rows <= 0:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid} seqno="
                    f"{notify.seqno}: scheduled {got} tokens but the data "
                    f"plane reported recv_rows=0 (该请求不该进 batch —— 可见集"
                    f"谓词 `recv_rows > 0` 失效)"
                )
            # 该请求已累计收到的最高行号(recv_rows 按 prompt 起点计数,
            # 不是相对本 chunk)。本步要算到 ``start + got``,不能越过它。
            # ``start`` 取**本步生效起点**(部分消费时比 item.offset 大),
            # 它由调用方通过 eff_starts 平行传入 —— **不能**回写 item.offset
            # 来达到同样效果:那个字段是载荷尺寸协议(见 _schedule_mixed)。
            #
            # **别写成 ``start + got > recv_rows`` 的等价变形去减 offset**:
            # 续传 chunk 的 offset > 0,而它前面的行早已计入 recv_rows,减掉
            # offset 就会把"前面已经收到的部分"漏算 —— 真机实录:一个
            # offset=1164/已收 2346 的续传请求被误判成越界
            # ("only 2346 have arrived" 而它算的区间 [1164, 3510) 其实合法)。
            start = int(eff_starts[k]) if eff_starts else item.offset
            if start + got > recv_rows:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid} seqno="
                    f"{notify.seqno}: native scheduled rows "
                    f"[{start}, {start + got}) but only "
                    f"{recv_rows} have arrived in total "
                    f"(window={w} chunk_rows={item.num_tokens} "
                    f"chunk_offset={item.offset} decode_exact={decode_exact})"
                )

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
    # 前缀协商:抢占重准入统一路径(设计 §5.3)与 refill 闸门                #
    # ------------------------------------------------------------------ #
    def _lwd_prep_prefix_readmission(self, request) -> bool:
        """被抢占请求的重准入预备:只读重查 H' -> 短钉残块 + 快照。

        对所有被抢占请求生效(不止命中过的请求):H' > S 是 self-hit——
        未命中过的请求被抢占后可命中自己此前算过且入缓存的块,抢占
        重算直接变便宜(原生即有此性质,host 暂存全覆盖故零额外处理)。

        命中上限 = num_prompt_tokens - 1(尾部至少保留 1 个 token 参与
        计算;host 暂存只覆盖 prompt 段,命中不进 decode 区);另做到达
        钳制 H''=min(H', recv_rows 下对齐调度块粒度)——H' 不得超过
        本请求已收行数(stager ``consumed <= recv_rows`` 不变量),且
        H'' 恒为各组块整数倍(与命中边界同源对齐)。

        Returns:
            True = H'' >= 暂存地板 S,快照就绪,本步可开窗;
            False = H'' < S(命中块已被逐出),已发 LwdRefillNotify 并挂
            refill 闸门,本步不可见,等补发完成(lwd_refill_done)摘闸。

        Raises:
            RuntimeError: refill 通道未装配(引擎未注入 lwd_refill_sender,
                属接线错误,fail-fast)。
        """
        reservations = self.lwd_prefix_reservations
        floor = getattr(request, "lwd_resume_cap_tokens", 0)
        if request.skip_reading_prefix_cache:
            groups, h_prime = (), 0
        else:
            groups, h_prime = (
                self.kv_cache_manager.coordinator.find_longest_cache_hit(
                    request.block_hashes,
                    max_cache_hit_length=max(0, request.num_prompt_tokens - 1),
                )
            )
        # 到达钳制:H' 可能越过本请求的已收高水位(共享前缀的其他请求
        # 把缓存链补得比本请求已收到的行更长)。stager 不变量
        # ``0 <= consumed <= recv_rows`` 要求准入边界不超过已收行数,
        # 否则 runner 的读指针前跳会越过未写入行。向下对齐到**调度块
        # 粒度**(scheduler_block_size = 各组块大小 LCM,与命中边界同源
        # ——混合 KV 组下 hash 块是各组块大小的 GCD,钳到 hash 粒度会
        # 破坏组块对齐,truncate_groups 的整除假设随之失效);H''=0
        # 退化为无命中(截空不钉块,走原生全量重算暂存行)。只截钉块/
        # 快照,不回写台账。
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        recv_rows = int(ledger.get(request.request_id, 0))
        align = self.kv_cache_manager.coordinator.scheduler_block_size
        h_clamped = min(h_prime, recv_rows // align * align)
        if h_clamped < h_prime:
            logger.info(
                "[Lwd][prefix] clamp readmission hit req=%s: H'=%d -> %d "
                "(recv_rows=%d align=%d)",
                request.request_id, h_prime, h_clamped,
                recv_rows, align,
            )
            groups = reservations.truncate_groups(groups, h_clamped)
            h_prime = h_clamped
        # 钉住残块 + fail-closed 快照:防准入/refill 期间残块再被逐出
        # (补发赶不上逐出的追赶抖动);预留短窗口,首次成功分配后释放
        reservations.reserve_hit(request, groups, h_prime)
        if h_prime >= floor:
            return True
        sender = getattr(self, "lwd_refill_sender", None)
        if sender is None:
            raise RuntimeError(
                "[Lwd][prefix] refill channel not wired: "
                "scheduler.lwd_refill_sender missing (engine init order?)"
            )
        self._lwd_refill_gate.add(request.request_id)
        self._lwd_refill_pending[request.request_id] = h_prime
        logger.info(
            "[Lwd][prefix] refill required req=%s: H'=%d < floor S=%d, "
            "requesting edge resend of [%d, %d)",
            request.request_id, h_prime, floor, h_prime, floor,
        )
        sender(request.request_id, h_prime, floor)
        return False

    def lwd_refill_done(self, request_id: str) -> None:
        """refill 补发完成(引擎 done MQ 驱动):摘闸门 + 推进暂存地板。

        补发行填齐 [H', S) 后,stager 覆盖区间扩到 [H', prompt):暂存
        地板推进到 H'(下一次抢占以新地板判补发);请求按钉住的 H'
        快照恢复可见。幂等:迟到/重复上报(请求已终结、闸门已摘)
        告警后忽略(§6 竞态:refill 在途 abort / 迟到 refill_done)。"""
        h_prime = self._lwd_refill_pending.pop(request_id, None)
        if h_prime is None:
            logger.warning(
                "[Lwd][prefix] stale refill_done req=%s (not gated), ignored",
                request_id,
            )
            return
        self._lwd_refill_gate.discard(request_id)
        request = self.requests.get(request_id)
        if request is None:
            # 补发期间请求已终结:预留/闸门由 _free_request 兜底清理
            return
        request.lwd_resume_cap_tokens = h_prime
        logger.info(
            "[Lwd][prefix] refill done req=%s: gate lifted, floor -> %d",
            request_id, h_prime,
        )

    # ------------------------------------------------------------------ #
    # 调度主体(arrival 驱动,design v8)                                    #
    # ------------------------------------------------------------------ #
    def _schedule_impl(self) -> SchedulerOutput:
        # (2) 收集 decode 集(判据:prompt 已算完 = 可采样)。
        decode_ids = self._lwd_collect_decode_requests()
        # decode 的 token 账一次算清(decode 优先);KV 完全交给原生
        # (waiting 分配失败即 break,天然 defer)。
        decode_exact, _ = self._lwd_decode_tokens(decode_ids)
        room = self.max_num_scheduled_tokens - decode_exact
        # ---- arrival 驱动(design v8)----
        # 不再有 notify 队列/chunk 边界:任何"已收未算完"的 prefill 请求
        # 本步都可下发,窗口 = min(已收 − 已算, room)。已收行数
        # ``recv_rows`` 由 worker 在分段 D2H 完成后随水位捎带上报(prompt
        # 坐标累计高水位);已算行数取原生水位 num_computed_tokens。
        # 可见集谓词(recv_rows > computed)同时是安全闸:零值占位绝不进
        # batch(读了不报错,静默错算)。chunk 化从此只是边侧的流量整形。
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        reservations = getattr(self, "lwd_prefix_reservations", None)
        windowed: list[tuple] = []  # (request, n_rows)
        if room > 0:
            for request in itertools.chain(
                self.running, self.waiting, self.skipped_waiting
            ):
                rid = request.request_id
                if rid in self._lwd_dropped:
                    continue  # abort 冻结:剩余行永不下发
                computed = request.num_computed_tokens
                if computed >= request.num_prompt_tokens:
                    continue  # decode 态:由 decode_ids 覆盖
                start = computed
                if self._lwd_prefix_enabled:
                    if rid in self._lwd_refill_gate:
                        continue  # refill 补发未到齐:洞未填齐整体不可见
                    if computed == 0:
                        if (
                            request.num_preemptions > 0
                            and reservations is not None
                            and not reservations.has(rid)
                            and not self._lwd_prep_prefix_readmission(request)
                        ):
                            continue  # H'<S:refill 在途,挂闸门等补发
                        # 命中请求的起点是预留命中 H(被命中的 [0,H) 零值行
                        # 永不进窗;recv_rows 亦自 H 起涨),不是裸水位 0
                        start = prefix_start(request, reservations)
                arrived = int(ledger.get(rid, 0)) - start
                if arrived <= 0:
                    continue  # 数据还没到
                n = min(arrived, room)
                if n < 1:
                    break  # 预算尽
                # 攒满批:数据未收完且不够一个满窗(数据限速的残余窗)
                # 时本步不开窗,攒到与下一块合并成大窗 —— 消除"碎窗独占
                # 一步"的固定开销尾巴(一块被预算截断、下一块还在线上时,
                # 残余几行会被 async 调度单独排成一个 batch)。到达限速下
                # 这并不损失吞吐(闲置是数据依赖的必然),延迟上限为一个
                # chunk 到达间隔。末段(已收 == prompt 全长)不受此限,
                # 保证请求必然算完;抢占重算后 recv_rows 满值,天然不命中。
                # start + arrived == recv_rows(起点偏移恒等式),末段判据
                # 对命中请求同样成立。
                if n < room and start + arrived < request.num_prompt_tokens:
                    continue
                # start 必须此刻捕获:原生 schedule() 会经
                # ``_update_after_schedule`` **乐观推进** num_computed,
                # 下发后再读就是窗口末尾而不是起点。
                windowed.append((request, n, start))
                room -= n
        if windowed:
            out = self._lwd_dispatch_windows(windowed, decode_ids,
                                             decode_exact)
        else:
            out = self._lwd_schedule_for_visible_reqs(decode_ids)
        self._lwd_last_windowed = len(windowed)
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

        返回 ``(decode_exact, {req_id: d_i})``。d_i 现在只喂 token 账
        (KV 块账已交还原生);per_req 保留给调用方做诊断。
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

    def _lwd_dispatch_windows(
        self,
        windowed: list[tuple],
        decode_ids: list[str],
        decode_exact: int,
    ) -> SchedulerOutput:
        """混排步:decode 集 + 所有"已收未算完"请求的窗口同批下发。

        arrival 驱动(design v8):窗口 = min(已收 − 已算, room),无 notify
        队列、无 chunk 边界、无 seqno 对账 —— 注入落点恒为
        ``num_computed_tokens``,runner 按请求从自己的 host buffer 顺序
        取数。

        **按请求下发**:窗口经 ``lwd_window_cap`` 逐请求挂给原生(唯一
        上界),预算压成 ``decode_exact + Σ窗口`` 作总量上界;步后逐请求
        W2 核验(``start + got ≤ recv_rows`` 且 ``got ≤ 窗口``,恒真,纯
        看门狗);原生少排(块对齐截尾/KV 分配失败)合法,下一步按
        num_computed 重算。
        """
        logger.info(
            "[Lwd][cloud-sched] mixed step reqs=%s windows=%s",
            [r.request_id for r, _n, _s in windowed],
            [n for _r, n, _s in windowed],
        )
        total_windows = sum(n for _r, n, _s in windowed)
        req_ids = decode_ids + [r.request_id for r, _n, _s in windowed]
        saved_budget = self.max_num_scheduled_tokens
        # 逐请求上界:把每个 windowed 请求本步可排的行数挂到请求对象上,
        # 原生 running/waiting 循环算 num_new_tokens 时对它有 min —— 不能
        # 只靠"压缩总预算 + FIFO 前缀序":那个组合不等于逐请求取小,池化
        # 预算会被排在前面的请求超领(真机实录 seqno=101:窗口 8 行,原生
        # 按整条 prompt 剩余排了 138 行,把 decode_exact 的份额吃掉)。
        # 预算压缩保留为总量上界,双保险。
        for request, n, _s in windowed:
            request.lwd_window_cap = n
        self.max_num_scheduled_tokens = decode_exact + total_windows
        try:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        finally:
            self.max_num_scheduled_tokens = saved_budget
            for request, _n, _s in windowed:
                request.lwd_window_cap = None
        # ---- W2(design v7 §7):本方案唯一"静默错算"型风险的兜底 ----
        # num_computed 在此刻仍是步初值(update_from_output 未跑)。
        ledger = getattr(self, "lwd_recv_rows", None) or {}
        for request, w, start in windowed:
            rid = request.request_id
            got = int(out.num_scheduled_tokens.get(rid, 0))
            if got == 0:
                continue
            recv_rows = int(ledger.get(rid, 0))
            if recv_rows <= 0:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid}: scheduled "
                    f"{got} tokens but the data plane reported recv_rows=0 "
                    f"(该请求不该进 batch —— 可见集谓词失效)"
                )
            # start 是准入时(原生乐观推进**之前**)捕获的窗口起点。
            if got > w or start + got > recv_rows:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] W2 VIOLATION req={rid}: native "
                    f"scheduled rows [{start}, {start + got}) — window={w}, "
                    f"recv_rows={recv_rows} (lwd_window_cap 失守)"
                )
        native_prefill = sum(
            int(out.num_scheduled_tokens.get(r.request_id, 0))
            for r, _n, _s in windowed
        )
        if native_prefill < total_windows:
            # 合法但值得观测:几乎只在原生 KV 分配失败时出现 —— 这正是
            # "KV 开始不够"的信号,比盯着 free 块数可靠。
            logger.warning(
                "[Lwd][cloud-sched] native under-scheduled: "
                "native prefill=%d < Σwindows=%d (native-side truncation, "
                "typically KV pressure; the remainder is retried next step)",
                native_prefill, total_windows,
            )
        # 新准入请求的全量 mrope positions 随 SO 带给 runner(RequestNotify
        # → Request 透传而来;纯文本为空字节)。
        for d in out.scheduled_new_reqs:
            req = self.requests.get(d.req_id)
            if req is not None:
                d.lwd_mrope_positions = (
                    getattr(req, "lwd_mrope_positions", b"") or b""
                )
        return out

    def _lwd_drop_request_from_chunks(self, request_id: str) -> int:
        """把一个被 abort 的请求从逐请求台账里摘成"永不消费"。

        返回它被丢弃的剩余行数。已算完的行是历史事实(KV 已算、embeddings
        已注入),不回滚;只把剩余行冻结。arrival 驱动后不再有预告队列,
        剩余行直接按"prompt 剩余"记账。

        同时把它排进"待通知 worker"名单:worker 的暂存池是按请求管存储的,
        这些剩余行在 host 上还占着 pinned,必须让 worker 也冻结 —— abort 时
        worker 只冻结,真正的释放在请求结束那条路上做(design v7 §6.1/§6.3)。
        """
        already = request_id in self._lwd_dropped
        self._lwd_dropped.add(request_id)
        remaining = 0
        request = self.requests.get(request_id)
        if request is not None:
            remaining = max(
                request.num_prompt_tokens - request.num_computed_tokens, 0
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
        self._lwd_log_sched_batch(out, getattr(self, "_lwd_last_windowed", 0))
        logger.info(
            "[Lwd][perf] cloud-sched dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        return out

    def _lwd_log_sched_batch(self, out: SchedulerOutput,
                             n_prefill: int = 0) -> None:
        """[Lwd][sched] 每步调度批结构化日志(lwd_backlog_probe ⑤ 段锚点)。

        phase 判定:本步有 prefill 窗口(n_prefill>0,由 _schedule_impl
        传入)且批内还有 decode 请求即 MIXED;prefill 独占为 PREFILL;
        有排程 token 为 DECODE;否则 EMPTY。reqs 超 12 个截断,防大
        decode 批刷屏。"""
        if n_prefill:
            n_reqs = len(out.num_scheduled_tokens)
            phase = "MIXED" if n_reqs > n_prefill else "PREFILL"
        else:
            phase = "DECODE" if out.total_num_scheduled_tokens else "EMPTY"
        reqs = list(out.num_scheduled_tokens)
        reqs_str = ",".join(reqs[:12]) + (
            f",+{len(reqs) - 12}more" if len(reqs) > 12 else ""
        )
        logger.info(
            "[Lwd][sched] cloud step=%d phase=%s reqs=[%s] tokens=%d "
            "decode_ready=%d",
            self._lwd_sched_step, phase, reqs_str,
            out.total_num_scheduled_tokens,
            len(self._lwd_collect_decode_requests()),
        )
