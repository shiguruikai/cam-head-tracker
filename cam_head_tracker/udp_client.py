import logging
import socket
import struct
import threading

import numpy as np

logger = logging.getLogger(__name__)

_PACKER = struct.Struct("<dddddd")


def _create_udp_client(host: str, port: int) -> tuple[socket.socket, tuple]:
    info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)[0]
    family, socket_type, proto, _, addr = info
    sock = socket.socket(family, socket_type, proto)
    return sock, addr


class UDPClient:
    def __init__(self, host: str, port: int):
        self._sock: socket.socket | None = None
        self._sock, self._addr = _create_udp_client(host, port)
        self._lock = threading.Lock()
        self._closed = False

        logger.debug("Created UDPClient: host=%s, port=%s", self._addr[0], self._addr[1])

    def send(self, data: np.ndarray):
        if self._closed:
            return

        with self._lock:
            if self._closed:
                return

            try:
                self._sock.sendto(_PACKER.pack(*data), self._addr)
            except OSError as e:
                logger.debug("UDP send failed: %s", e)

    def close(self):
        if self._closed:
            return

        with self._lock:
            if self._closed:
                return

            try:
                self._sock.close()
            finally:
                self._closed = True
                self._sock = None
