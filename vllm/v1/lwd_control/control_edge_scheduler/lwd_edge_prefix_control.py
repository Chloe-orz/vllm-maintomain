"""边侧前缀协商:hash 粒度重建、READY 等待、超时与取消清理。

MVP 单相协商(设计 §3.3):add_request 分配协商号 nonce -> (必要时)
同步重建哈希链 -> 发 LwdRequestNotify(现有路径)-> 不入本地队列,
挂超时定时器;nonce 匹配的 READY 到达后经
``scheduler.lwd_edge_prefix_ready`` 以 num_computed=hit 入队。

线程归属:协商状态(_pending)由引擎主线程维护;超时定时器只向
input_queue 投递 error READY 消息,不直接改等待表。异步建链/
probe-first 为阶段 2 优化,本文件不含。
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdPrefixReadyNotify,
)

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_communication.lwd_notify import (
        LwdHelloNotify,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        LwdEdgeEngineCore,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LwdEdgePrefixControl:
    """边侧前缀协商控制器(引擎主线程独占等待表)。"""

    def __init__(self, engine: LwdEdgeEngineCore, timeout_s: float) -> None:
        self._engine = engine
        self._timeout_s = timeout_s
        self._nonce = 0
        self._hash_block_size = 0
        # request_id -> (request, 超时定时器);READY 等待中的请求已占
        # 云侧在途名额(云已建请求并钉块),字典引用共享给调度器,
        # 计入 lwd_edge_max_num_seqs_check 闸门。
        self._pending: dict[str, tuple[Request, threading.Timer]] = {}

    @property
    def pending(self) -> dict[str, tuple[Request, threading.Timer]]:
        """等待表引用(调度器在途名额闸门只读 len,不改写)。"""
        return self._pending

    @property
    def hash_block_size(self) -> int:
        """协商后的云侧哈希块粒度(0 = 尚未 configure;供调度器校验
        READY 命中的块对齐)。"""
        return self._hash_block_size

    def configure(self, hello: LwdHelloNotify) -> None:
        """按云侧通告的 hash_block_size 重建边侧 hasher(MVP 同步重建)。

        边侧 0 KV 组,本地默认块粒度不等于云侧 NPU 的 KV block size;
        哈希链粒度必须以云为准,否则云侧按自己的粒度查链恒不命中
        (或错配)。HELLO 在首批请求前完成时,后续请求首建即用新
        hasher;竞态窗口内的请求由 add_request 同步重建兜底。
        对齐参数/MTP 尾部规则以云为准,但边侧切块无需对齐(新架构
        云侧按 arrival 开窗,aligned_chunk_length 机制不复活)。"""
        if hello.hash_block_size <= 0:
            raise ValueError(
                "cloud HELLO advertised prefix_caching but no valid "
                f"hash_block_size ({hello.hash_block_size})"
            )
        engine = self._engine
        hash_fn = get_hash_fn_by_name(
            engine.vllm_config.cache_config.prefix_caching_hash_algo
        )
        if engine.request_block_hasher is None:
            init_none_hash(hash_fn)
        engine.request_block_hasher = get_request_block_hasher(
            hello.hash_block_size, hash_fn
        )
        self._hash_block_size = hello.hash_block_size
        logger.info(
            "[Lwd][prefix] edge hasher rebuilt: hash_block_size=%d "
            "alignment=%d(required=%s) use_eagle=%s",
            hello.hash_block_size, hello.prefix_alignment_tokens,
            hello.prefix_alignment_required, hello.prefix_use_eagle,
        )

    def add_request(self, request: Request) -> None:
        """分配协商号并预告云侧,随后挂 READY 等待(不入本地队列)。

        预告失败(发布重试耗尽抛错)时撤销等待登记,异常沿
        add_request 调用链回前端 error 通道——此时请求未入队、
        云侧零残留,无需补发 abort。"""
        engine = self._engine
        self._nonce += 1
        request.lwd_prefix_nonce = self._nonce
        # 哈希链同步重建(MVP):竞态窗口内请求可能按旧粒度建链;
        # 用协商后的引擎 hasher 清链重算(保留 mm 标识/位置/cache_salt,
        # 与原生 hasher 同键)。性能优化(异步建链/首链复用)属阶段 2。
        request.block_hashes = []
        request._block_hasher = engine.request_block_hasher
        request.update_block_hashes()
        expired = LwdPrefixReadyNotify(
            request.request_id, 0, request.lwd_prefix_nonce,
            "prefix reply timed out",
        )
        timer = threading.Timer(
            self._timeout_s,
            engine.input_queue.put_nowait,
            args=((EngineCoreRequestType.WAKEUP, expired),),
        )
        timer.daemon = True
        self._pending[request.request_id] = (request, timer)
        try:
            engine.scheduler.lwd_edge_add_request(request)
        except Exception:
            self._pending.pop(request.request_id, None)
            raise
        timer.start()

    def handle_client_request(self, request_type, reply) -> bool:
        """主线程入口:只接管前缀 READY(含超时定时器投递的 error READY)。

        nonce 不匹配或请求已取消 = 迟到消息,幂等吞掉(返回 True,
        不再交原生 WAKEUP 处理)。"""
        if request_type != EngineCoreRequestType.WAKEUP or not isinstance(
            reply, LwdPrefixReadyNotify
        ):
            return False
        pending = self._pending.get(reply.request_id)
        if pending is None or pending[0].lwd_prefix_nonce != reply.nonce:
            return True
        request, timer = self._pending.pop(reply.request_id)
        timer.cancel()
        engine = self._engine
        if reply.error:
            logger.error(
                "[Lwd][prefix] ready error req=%s: %s",
                reply.request_id, reply.error,
            )
            # 先零命中入队再走 abort 出口:让调度器/输出记账看到该请求,
            # 终结按普通 ABORT finish 回前端(不作 error 区分,也不作为
            # 零命中请求继续执行);AbortNotify 随 abort 发云,云侧释放
            # 已钉的预留块。
            engine.scheduler.lwd_edge_prefix_ready(request, 0)
            engine.abort_requests([reply.request_id])
        else:
            logger.info(
                "[Lwd][prefix] ready req=%s hit_tokens=%d",
                reply.request_id, reply.hit_tokens,
            )
            engine.scheduler.lwd_edge_prefix_ready(request, reply.hit_tokens)
        return True

    def prepare_abort(self, request_ids: list[str]) -> None:
        """READY 等待中的 abort:撤等待登记,并以零命中入队,让后续
        原生取消流程找得到请求(AbortNotify 仍会发云,云侧释放预留)。"""
        for request_id in request_ids:
            pending = self._pending.pop(request_id, None)
            if pending is not None:
                request, timer = pending
                timer.cancel()
                self._engine.scheduler.lwd_edge_prefix_ready(request, 0)

    def shutdown(self) -> None:
        """取消全部等待定时器;不从定时器线程操作调度状态。"""
        for _, timer in self._pending.values():
            timer.cancel()
