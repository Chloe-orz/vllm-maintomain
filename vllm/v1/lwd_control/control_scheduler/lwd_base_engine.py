# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common base of the LWD edge and cloud engine cores: scheduler injection
order, config attachment and the lifecycle of both channels, the receiver
thread message entry and the has_work transition log."""

from __future__ import annotations

from typing import Any

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine.core import EngineCoreProc


class LwdBaseEngineCore(EngineCoreProc):
    """EngineCoreProc of an LWD edge or cloud.

    Subclasses declare `lwd_scheduler_cls`, injected before `super()`, which
    builds the scheduler from `scheduler_cls` once. Upstream disables chunked
    prefill for deployments without KV cache groups, while LWD edge chunks are
    pure scheduling bookkeeping, so it is restored for LWD engines afterwards.
    Both channels are None until built, and shutdown tolerates that.
    """

    lwd_scheduler_cls: type

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._lwd_publisher = None
        self._lwd_subscriber = None
        # has_work state at the last transition log; None = not logged yet.
        self._lwd_last_has_work_state: tuple[bool, bool, bool] | None = None
        vllm_config = kwargs["vllm_config"]
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        chunked_prefill_wanted = scheduler_config.enable_chunked_prefill
        super().__init__(*args, **kwargs)
        if not scheduler_config.enable_chunked_prefill and chunked_prefill_wanted:
            scheduler_config.enable_chunked_prefill = True
        self.lwd_config = vllm_config.lwd_config

    def _lwd_shutdown_comm(self) -> None:
        """Shut down both channels (idempotent, tolerates unbuilt ones), the
        subscriber first to cut off input."""
        if self._lwd_subscriber is not None:
            self._lwd_subscriber.shutdown()
        if self._lwd_publisher is not None:
            self._lwd_publisher.shutdown()

    # ------------------------------------------------------------------ #
    # Receiver thread message entry                                      #
    # ------------------------------------------------------------------ #
    def _lwd_on_message(self, msg) -> None:
        """Common entry of received messages: `_lwd_dispatch_message` handles
        them, unknown types are dropped here with a warning (so the two engines
        do not repeat it)."""
        if not self._lwd_dispatch_message(msg):
            lwd_logger.warning("[comm] drop unexpected frame %r", type(msg))

    def _lwd_dispatch_message(self, msg) -> bool:
        """Subclass hook. True = handled, False = unknown (dropped with a
        warning by the entry)."""
        return False

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def has_work(self) -> bool:
        """Native three-source check, logging its transitions (lwd_logger,
        level VLLM_LWD_LOG_LEVEL, silent by default)."""
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
