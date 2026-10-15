"""控制面通信收发器:ZMQ socket 句柄,只做收发,无线程;通知协议住 lwd_notify。
send/recv 仅持有线程可调(zmq 单线程亲和);term 是跨线程打断阻塞收发的可靠手段。"""

from __future__ import annotations

import zmq
from vllm.lwd_logger import lwd_logger


class LwdControlCommunicator:
    """单向平面收发器:endpoint socket 的收发句柄,无线程。
    endpoint=None 为延迟连接态,由持有方线程经 retarget 补连(目标装配期不可知)。"""

    def __init__(self, endpoint: str | None, socket_type: int, *, bind: bool) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(socket_type)
        # 退出时不因未发完的帧挂死进程;显式 close(0) 可覆盖
        self._socket.setsockopt(zmq.LINGER, 2000)
        self._endpoint: str | None = None
        if endpoint is not None:
            if bind:
                self._socket.bind(endpoint)
            else:
                self._socket.connect(endpoint)
            self._endpoint = endpoint
        lwd_logger.info(
            "[zmq] communicator up: type=%s mode=%s endpoint=%s",
            "PUSH" if socket_type == zmq.PUSH else "PULL",
            "bind" if bind else "connect",
            endpoint,
        )

    def retarget(self, endpoint: str) -> None:
        """换连接目标(仅持有线程):先连新再断旧,换址期消息不丢;同址幂等。"""
        if endpoint == self._endpoint:
            return
        self._socket.connect(endpoint)
        if self._endpoint is not None:
            self._socket.disconnect(self._endpoint)
        lwd_logger.info(
            "[zmq] retarget: %s -> %s", self._endpoint, endpoint
        )
        self._endpoint = endpoint

    def send(self, data: bytes) -> None:
        """阻塞发送;无对端时阻塞,由调用方有界队列形成对端背压。"""
        self._socket.send(data)

    def poll(self, timeout: int) -> bool:
        """等待 socket 可读;timeout 毫秒,就绪返回 True,超时返回 False。"""
        return self._socket.poll(timeout) != 0

    def recv(self, block: bool = True) -> bytes:
        """接收;block=False 非阻塞,无消息时抛 zmq.Again。"""
        return self._socket.recv(flags=0 if block else zmq.NOBLOCK)

    def close(self) -> None:
        """close(0) 立即返回并丢弃未发帧;不保证唤醒阻塞中的 recv(平台相关)。"""
        self._socket.close(0)

    def terminate(self) -> None:
        """term context:使阻塞的收发以 ETERM 返回并阻塞至收尾,供持有方
        join 超时后兜底关停。"""
        self._context.term()
