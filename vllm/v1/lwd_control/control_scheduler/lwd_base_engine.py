"""边/云引擎公共基类:调度器注入时序、配置挂载与收发两面生命周期,
接收线程消息入口与 has_work 状态转换日志的公共骨架。"""

from __future__ import annotations

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine.core import EngineCoreProc


class LwdBaseEngineCore(EngineCoreProc):
    """边/云 EngineCore 公共基类。

    子类经 lwd_scheduler_cls 声明调度器类;注入必须发生在 super() 之前
    (super 构建 scheduler 时一次性消费 scheduler_cls,后设无效)。
    收发两面构造前为 None,关停容忍未建。上游 EngineCore 会对无 KV
    cache 组的部署禁用 chunked prefill,而 LWD 边侧 chunk 是纯调度
    记账(执行层只 embed 切片),构造后恢复使开关仅作用于 LWD 引擎。"""

    lwd_scheduler_cls: type

    def __init__(self, *args, **kwargs) -> None:
        self._lwd_publisher = None
        self._lwd_subscriber = None
        # has_work 三态上次快照(仅状态转换时打日志;None = 尚未打点)
        self._lwd_last_has_work_state = None
        vllm_config = kwargs["vllm_config"]
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        chunked_prefill_wanted = scheduler_config.enable_chunked_prefill
        super().__init__(*args, **kwargs)
        if not scheduler_config.enable_chunked_prefill and chunked_prefill_wanted:
            scheduler_config.enable_chunked_prefill = True
        self.lwd_config = vllm_config.lwd_config

    def _lwd_shutdown_comm(self) -> None:
        """两面关停(幂等,容忍未建):subscriber 先关断输入。"""
        if self._lwd_subscriber is not None:
            self._lwd_subscriber.shutdown()
        if self._lwd_publisher is not None:
            self._lwd_publisher.shutdown()

    # ------------------------------------------------------------------ #
    # 接收线程消息公共入口                                                #
    # ------------------------------------------------------------------ #
    def _lwd_on_message(self, msg) -> None:
        """接收线程消息统一入口:子类 _lwd_dispatch_message 分派,返回
        False 的未知帧统一告警丢弃(两侧引擎不再各写一份)。"""
        if not self._lwd_dispatch_message(msg):
            lwd_logger.warning("[comm] drop unexpected frame %r", type(msg))

    def _lwd_dispatch_message(self, msg) -> bool:
        """子类分派钩子:True = 已处理;False = 未知帧(入口告警丢弃)。"""
        return False

    # ------------------------------------------------------------------ #
    # 引擎接口覆写                                                        #
    # ------------------------------------------------------------------ #
    def has_work(self) -> bool:
        """原生三源判定 + 状态转换日志(lwd_logger,VLLM_LWD_LOG_LEVEL
        独立控级,默认静默)。"""
        state = (
            self.engines_running,
            self.scheduler.has_requests(),
            bool(self.batch_queue),
        )
        if state != self._lwd_last_has_work_state:
            lwd_logger.info(
                "[has_work] engines_running=%s has_requests=%s batch_queue=%s",
                *state,
            )
            self._lwd_last_has_work_state = state
        return any(state)

    def shutdown(self) -> None:
        self._lwd_shutdown_comm()
        super().shutdown()
