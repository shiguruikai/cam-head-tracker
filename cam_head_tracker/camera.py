import logging
import queue
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FFMPEG_PATH = Path(__file__).parent / "assets/ffmpeg.exe"

_startupinfo = subprocess.STARTUPINFO()

# Pyinstallerで作成したexeから実行した場合にコンソールを表示させないための設定
_startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
_startupinfo.wShowWindow = subprocess.SW_HIDE


def get_camera_device_names() -> list[str]:
    name_regex = re.compile(r'"(.+?)"\s+\(video\)')
    cmd = [str(FFMPEG_PATH), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        startupinfo=_startupinfo,
        encoding="utf-8",
    )
    names: list[str] = []
    for line in proc.stdout.splitlines():
        m = name_regex.search(line)
        if not m:
            continue
        names.append(m.group(1))
    return names


@dataclass(frozen=True)
class CameraDeviceOption:
    name: str
    format: str
    format_type: str
    width: int
    height: int
    fps: float


def get_camera_device_options(device_name: str) -> list[CameraDeviceOption]:
    options_set: set[CameraDeviceOption] = set()

    options_regex = re.compile(
        r"(?:vcodec=(?P<vcodec>\S+)|pixel_format=(?P<pixel_format>\S+))\s+"
        r"min s=(?P<min_w>\d+)x(?P<min_h>\d+)\s+fps=(?P<min_fps>\d+(?:\.\d+)?)\s+"
        r"max s=(?P<max_w>\d+)x(?P<max_h>\d+)\s+fps=(?P<max_fps>\d+(?:\.\d+)?)"
    )

    cmd = [str(FFMPEG_PATH), "-hide_banner", "-list_options", "true", "-f", "dshow", "-i", f"video={device_name}"]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        startupinfo=_startupinfo,
        encoding="utf-8",
    )

    for line in proc.stdout.splitlines():
        m = options_regex.search(line)

        if not m:
            continue

        d = m.groupdict()

        f_type = "vcodec" if d["vcodec"] else "pixel_format"
        f_val = d[f_type]
        min_w, max_w = int(d["min_w"]), int(d["max_w"])
        min_h, max_h = int(d["min_h"]), int(d["max_h"])
        min_fps, max_fps = float(d["min_fps"]), float(d["max_fps"])

        for w in {min_w, max_w}:
            for h in {min_h, max_h}:
                for fps in {min_fps, max_fps}:
                    if fps < 1:
                        continue
                    options_set.add(
                        CameraDeviceOption(
                            name=device_name, format_type=f_type, format=f_val, width=w, height=h, fps=fps
                        )
                    )

    return sorted(options_set, key=lambda x: (x.format_type, x.format, x.width, x.height, x.fps))


class VideoCapture:
    _TIMEOUT_SECONDS = 5

    def __init__(
        self,
        name: str,
        input_format_type: str,
        input_format: str,
        width: int,
        height: int,
        fps: float,
    ):
        self._name = name
        self._input_format_type = input_format_type
        self._input_format = input_format
        self._width = width
        self._height = height
        self._fps = fps
        self._channels = 3
        self._frame_size = width * height * self._channels
        self._frame_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        cmd = [
            str(FFMPEG_PATH),
            "-hide_banner",
            "-fflags", "nobuffer", # 入力ストリームの遅延を減らす。
            "-fflags", "discardcorrupt", # 破損パケットをデコーダーに渡さず破棄する。
            "-flags", "low_delay", # エンコード／デコードの遅延を減らす。※H.264入力の場合以外では効果はなさそう。
            "-err_detect", "explode", # デコーダーでデータの破損を検知した際、デコードせず破棄する。
            "-f", "dshow",
            f"-{self._input_format_type}", f"{self._input_format}",
            "-video_size", f"{self._width}x{self._height}",
            "-framerate", f"{self._fps}",
            "-i", f"video={self._name}",
            "-c:v", "rawvideo",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ]  # fmt: skip
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if logger.isEnabledFor(logging.DEBUG) else subprocess.DEVNULL,
            startupinfo=_startupinfo,
            bufsize=self._frame_size,
        )
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._stdout_task = self._executor.submit(self._read_stdout)
        self._stderr_task = self._executor.submit(self._read_stderr)
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def is_opened(self) -> bool:
        return self._proc.poll() is None

    def _read_stdout(self):
        poll = self._proc.poll
        read = self._proc.stdout.read
        q = self._frame_queue

        try:
            while poll() is None:
                frame = read(self._frame_size)

                if len(frame) != self._frame_size:
                    break

                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass

                q.put(frame)
        finally:
            q.shutdown(True)

    def _read_stderr(self):
        if self._proc.stderr is None:
            return

        poll = self._proc.poll
        readline = self._proc.stderr.readline

        while poll() is None:
            line = readline()

            if not line:
                break

            logger.debug(line.decode("utf-8", errors="backslashreplace").strip())

    def _send_q(
        self,
    ):
        self._proc.stdin.write(b"q")
        self._proc.stdin.close()

    def close(self):
        if self._proc is None:
            return

        with self._lock:
            if self._proc is None:
                return

            try:
                # 1. qキーの入力による終了を試行
                if self._proc.poll() is None:
                    send_q_task = self._executor.submit(self._send_q)
                    try:
                        send_q_task.result(self._TIMEOUT_SECONDS)
                    except (ValueError, OSError, TimeoutError) as e:
                        logger.warning("failed to send q key to ffmpeg process", exc_info=e)
                else:
                    logger.warning("ffmpeg process terminated before close()")

                # 2. 読み取りタスクの終了待機
                stdout_exc = self._stdout_task.exception(self._TIMEOUT_SECONDS)
                stderr_exc = self._stderr_task.exception(self._TIMEOUT_SECONDS)
                if stdout_exc is not None:
                    logger.error("stdout_exc:", exc_info=stdout_exc)
                if stderr_exc is not None:
                    logger.error("stderr_exc:", exc_info=stderr_exc)

                # 3. プロセスの終了待機
                self._proc.wait(self._TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, TimeoutError):
                # タイムアウトした場合、finallyで強制終了
                pass
            finally:
                # 4. 強制終了
                if self._proc.poll() is None:
                    self._proc.kill()
                    self._proc.wait()

                # 5. リソース解放
                for r in [self._proc.stdin, self._proc.stdout, self._proc.stderr]:
                    if r:
                        try:
                            r.close()
                        except OSError:
                            pass

                self._executor.shutdown()
                self._proc = None

    def read(
        self,
        wait: bool = False,
    ) -> npt.NDArray[np.uint8] | None:
        try:
            frame = self._frame_queue.get(block=wait)
        except (queue.Empty, queue.ShutDown):
            return None

        return np.frombuffer(frame, np.uint8).reshape((self._height, self._width, self._channels))

    def frames(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        while True:
            frame = self.read(wait=True)
            if frame is None:
                break
            yield frame
