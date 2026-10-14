# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LwdPrefillNotifyQueue 测试:FIFO、身份消费、abort 清扫与并发烟雾。"""

from __future__ import annotations

import threading

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue import (
    LwdPrefillNotifyQueue,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdRangeItem,
    LwdRangeNotify,
)


def notify(seqno: int, items: list[LwdRangeItem] | None = None) -> LwdRangeNotify:
    return LwdRangeNotify(seqno=seqno, items=items or [])


def test_fifo_peek_pop_front():
    queue = LwdPrefillNotifyQueue()
    assert queue.peek() is None
    n0, n1 = notify(0), notify(1)
    queue.append(n0)
    queue.append(n1)
    assert len(queue) == 2
    assert queue.peek() is n0
    assert queue.pop_front(n0) is True
    assert queue.peek() is n1
    assert queue.pop_front(n1) is True
    assert len(queue) == 0


def test_pop_front_identity_mismatch():
    """身份不符(队首已被替换/摘除):返回 False,队列原样。"""
    queue = LwdPrefillNotifyQueue()
    n0, n1 = notify(0), notify(1)
    queue.append(n0)
    queue.append(n1)
    assert queue.pop_front(n1) is False
    assert len(queue) == 2
    assert queue.peek() is n0


def test_purge_mixed_items():
    """Cancel model rows without changing the tensor size or sequence."""
    queue = LwdPrefillNotifyQueue()
    mixed = notify(0, [LwdRangeItem("a", 0, 5), LwdRangeItem("b", 0, 5)])
    solo = notify(1, [LwdRangeItem("c", 0, 3)])
    queue.append(mixed)
    queue.append(solo)
    assert queue.purge("a") == 1
    assert mixed.items == [LwdRangeItem("a", 0, 5), LwdRangeItem("b", 0, 5)]
    assert mixed.cancelled_req_ids == ["a"]
    assert len(queue) == 2
    assert queue.purge("b") == 1
    assert len(queue) == 2
    assert mixed.cancelled_req_ids == ["a", "b"]
    assert queue.purge("b") == 0
    assert queue.purge("zzz") == 0


def test_cancelled_batch_remains_until_its_wire_receive_is_scheduled():
    """Even an entirely cancelled batch must be consumed once."""
    queue = LwdPrefillNotifyQueue()
    n0 = notify(0, [LwdRangeItem("a", 0, 5)])
    queue.append(n0)
    queue.purge("a")
    assert queue.peek() is n0
    assert queue.pop_front(n0) is True
    assert queue.peek() is None


def test_concurrent_append_pop_monotonic():
    """并发烟雾:IO 线程持续投递,主线程按序消费,seqno 严格递增。"""
    queue = LwdPrefillNotifyQueue()
    total = 2000
    producer_error: list[Exception] = []

    def produce():
        try:
            for i in range(total):
                queue.append(notify(i))
        except Exception as exc:  # pragma: no cover
            producer_error.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    popped: list[int] = []
    while len(popped) < total:
        head = queue.peek()
        if head is None:
            continue
        if queue.pop_front(head):
            popped.append(head.seqno)
    thread.join()
    assert not producer_error
    assert popped == sorted(popped)
    assert len(popped) == total
