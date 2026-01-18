import argparse
import ctypes
import logging
import os
import platform
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox

from cam_head_tracker.gui import CamHeadTrackerApp
from cam_head_tracker.tracker import MediapipeTracker

logger = logging.getLogger(__name__)


def get_config_paths() -> list[Path]:
    """設定ファイルが存在する可能性のあるパスのリストを読み込む優先順で返す。"""
    paths = []

    # 1. C:\Users\ユーザー名\AppData\Roaming\CamHeadTracker\config.ini
    appdata = os.getenv("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "CamHeadTracker" / "config.ini")

    # 2. ~/.config/CamHeadTracker/config.ini
    paths.append(Path.home() / ".config" / "CamHeadTracker" / "config.ini")

    # 3. 実行ファイルと同じディレクトリ（ポータブル）
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).parent
    else:
        exe_path = Path(__file__).parent.parent
    paths.append(exe_path / "config.ini")

    return paths


def handle_exception(exc, val, tb):
    logging.error("handle_exception:", exc_info=(exc, val, tb))
    err_msg = "".join(traceback.format_exception(exc, val, tb))
    messagebox.showerror("Error", err_msg)
    sys.exit(1)


def enable_dpi_awareness():
    """モニターのDPIに応じてGUIが自動的にスケーリングされるようにする。"""
    if platform.system() == "Windows":
        try:
            # Windows 8.1以降
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            try:
                # Windows Vista以降
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


def main():
    parser = argparse.ArgumentParser(prog=Path(__file__).name, description="CamHeadTracker")
    parser.add_argument("--verbose", help="enable verbose output", action="store_true")
    parser.add_argument("--csv-output", help="path to csv output file", type=Path)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)7s %(name)-24s : %(message)s",
        datefmt="[%H:%M:%S]",
    )

    logger.debug("Parsed args: %s", args)

    enable_dpi_awareness()

    root = tk.Tk()
    root.report_callback_exception = handle_exception

    with CamHeadTrackerApp(
        root, config_paths=get_config_paths(), csv_output_path=args.csv_output
    ):
        root.mainloop()


if __name__ == "__main__":
    main()
