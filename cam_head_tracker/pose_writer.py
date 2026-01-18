import logging
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from cam_head_tracker.tracker import Pose

logger = logging.getLogger(__name__)


class PoseWriter(ABC):
    def __init__(self, *, ignore_write_error: bool = False):
        self.ignore_write_error = ignore_write_error
        self._lock = threading.Lock()
        self._closed = False

    def write(self, pose: Pose) -> None:
        if self._closed:
            return

        with self._lock:
            if self._closed:
                return

            try:
                self._write(pose)
            except Exception as e:
                logger.debug("[%s] Write failed: %s", self.__class__.__name__, e)

                if not self.ignore_write_error:
                    raise

    def close(self) -> None:
        if self._closed:
            return

        with self._lock:
            if self._closed:
                return

            try:
                self._close()
            finally:
                self._closed = True

    @abstractmethod
    def _write(self, pose: Pose): ...

    @abstractmethod
    def _close(self): ...


class UDPPoseWriter(PoseWriter):
    _PACKER = struct.Struct("<dddddd")

    def __init__(self, *, host: str, port: int):
        super().__init__(ignore_write_error=True)

        try:
            self._sock, self._addr = self._create_udp_client(host, port)
        except socket.gaierror as e:
            logger.debug(
                "[%s] Failed to create UDP socket: host=%s, port=%s, error=%s", self.__class__.__name__, host, port, e
            )
            raise

        logger.debug("[%s] Created UDP socket: host=%s, port=%s", self.__class__.__name__, self._addr[0], self._addr[1])

    def _write(self, pose: Pose):
        self._sock.sendto(self._PACKER.pack(*pose), self._addr)

    def _close(self):
        self._sock.close()

    @staticmethod
    def _create_udp_client(host: str, port: int) -> tuple[socket.socket, tuple]:
        info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)[0]
        family, socket_type, proto, _, addr = info
        sock = socket.socket(family, socket_type, proto)
        return sock, addr


class CsvPoseWriter(PoseWriter):
    def __init__(self, *, path: str | Path, append: bool = False):
        super().__init__()

        self._path = Path(path).absolute()
        self._append = append
        self._file: IO | None = None
        self._initialized = False

    def _write(self, pose: Pose):
        if not self._initialized:
            self._initialize()

        epoch_ms = int(time.time() * 1000)

        # 小数点以下6桁で書き込む
        self._file.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(epoch_ms, *pose))

    def _close(self):
        if self._file:
            self._file.close()

    def _initialize(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self._append else "w"

        exists_header = mode == "a" and self._path.stat().st_size > 0

        self._file = open(self._path, mode, encoding="utf-8")

        logger.debug("[%s] Opened csv file: %s", self.__class__.__name__, self._path)

        if not exists_header:
            self._file.write("epoch_ms,x,y,z,yaw,pitch,roll\n")

        self._initialized = True
