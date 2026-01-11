import argparse
import logging
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox

from cam_head_tracker.gui import CamHeadTrackerApp

logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.ini"


def handle_exception(exc, val, tb):
    logging.error("handle_exception:", exc_info=(exc, val, tb))
    err_msg = "".join(traceback.format_exception(exc, val, tb))
    messagebox.showerror("Error", err_msg)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog=Path(__file__).name, description="CamHeadTracker")
    parser.add_argument("--verbose", help="enable verbose output", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)7s %(name)-24s : %(message)s",
        datefmt="[%H:%M:%S]",
    )

    logger.debug("Parsed args: %s", args)

    root = tk.Tk()
    root.report_callback_exception = handle_exception

    with CamHeadTrackerApp(root, config_path=CONFIG_FILE_PATH):
        root.mainloop()


if __name__ == "__main__":
    main()
