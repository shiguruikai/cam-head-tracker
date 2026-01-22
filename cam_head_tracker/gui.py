import configparser
import logging
import queue
import socket
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Hashable, ParamSpec

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from cam_head_tracker.camera import CameraDeviceOption, VideoCapture, get_camera_device_names, get_camera_device_options
from cam_head_tracker.frame_rate_counter import FrameRateCounter
from cam_head_tracker.pose_writer import CsvPoseWriter, UDPPoseWriter
from cam_head_tracker.tracker import MediapipeTracker, Pose, PoseCorrector

logger = logging.getLogger(__name__)


APP_NAME = "Head Tracker"
ICON_FILE_PATH = Path(__file__).parent / "assets/icon.png"

DEFAULT_DISTANCE_SCALE = 1.0
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 4242
DEFAULT_PREVIEW_TEXT_COLOR = "#00FF00"

# キャリブレーションに必要なサンプルデータ数
N_CALIBRATION_SAMPLES = 30

X, Y, Z, YAW, PITCH, ROLL = 0, 1, 2, 3, 4, 5

P = ParamSpec("P")


class TkUIExecutor:
    def __init__(self, master: tk.Misc, interval_ms: int = 15):
        self._master = master
        self._interval_ms = interval_ms
        self._queue: queue.Queue[tuple[Callable, Any, Any]] = queue.Queue()
        self._schedules: dict[Hashable, tuple[Callable, Any, Any]] = {}
        self._lock = threading.Lock()
        self._after_id = self._master.after(self._interval_ms, self._ui_loop)
        self._is_running = True

    def set_interval_ms(self, interval_ms: int):
        self._interval_ms = interval_ms

    def submit(self, func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):
        if not self._is_running:
            return

        self._queue.put((func, args, kwargs))

    def schedule(self, key: Hashable, func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):
        if not self._is_running:
            return

        with self._lock:
            # 既存のキーを削除して末尾に追加
            if key in self._schedules:
                del self._schedules[key]
            self._schedules[key] = func, args, kwargs

    def clear(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            self._schedules.clear()

    def shutdown(self):
        if not self._is_running:
            return
        self._is_running = False
        self._master.after_cancel(self._after_id)
        self.clear()

    def _ui_loop(self):
        if not self._is_running:
            return

        tasks: list[tuple[Callable, Any, Any]] = []

        try:
            while True:
                tasks.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        with self._lock:
            if self._schedules:
                tasks.extend(self._schedules.values())
                self._schedules.clear()

        for func, args, kwargs in tasks:
            func(*args, **kwargs)

        if self._is_running:
            self._after_id = self._master.after(self._interval_ms, self._ui_loop)


class CamHeadTrackerApp(tk.Frame):
    def __init__(self, root: tk.Tk, *, config_paths: list[Path], csv_output_path: Path | None = None, **kwargs):
        super().__init__(root, **kwargs)

        self.config_paths = config_paths
        self.config_path = config_paths[-1]

        self.is_calibrating = False
        self.should_preview = True
        self.should_resize_preview = True
        self.preview_canvas_height = 0
        self.preview_text_color = DEFAULT_PREVIEW_TEXT_COLOR
        self.cam_options: list[CameraDeviceOption] = []
        self.cap: VideoCapture | None = None
        self.track_thread: threading.Thread | None = None

        self.udp_pose_writer: UDPPoseWriter | None = None
        self.csv_pose_writer: CsvPoseWriter | None = CsvPoseWriter(path=csv_output_path) if csv_output_path else None

        self.tracker = MediapipeTracker()
        self.corrector = PoseCorrector()
        self.fps_counter = FrameRateCounter()
        self.ui_executor = TkUIExecutor(root)

        self.root = root
        self.root.title(APP_NAME)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)
        self.root.iconphoto(True, tk.PhotoImage(file=ICON_FILE_PATH))

        style = ttk.Style(self.root)
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TLabelframe", padding=(10, 5, 10, 10))
        style.configure("TEntry", padding=(5, 2, 5, 2))
        style.configure("Calibration.TLabel", font=("Courier", 10))

        self.main_frame = ttk.Frame(self.root, padding=(20, 10, 20, 16))
        self.main_frame.pack(fill="both", expand=True)

        side_panel = ttk.Frame(self.main_frame)
        side_panel.pack(side="left", anchor="n")

        # キャプチャ画像表示用のキャンバス
        self.preview_canvas = tk.Canvas(self.main_frame, width=0, highlightthickness=0)
        self.preview_canvas.bind("<Configure>", self.onresize_preview_canvas)
        self.preview_canvas_tk_image = None  # 参照保持用
        # 画像
        self.preview_canvas.create_image(0, 0, tags="image", anchor="nw", image=self.preview_canvas_tk_image)
        # 中心線（縦横）
        self.preview_canvas.create_line(0, 0, 0, 0, tags="v_line", fill="white")
        self.preview_canvas.create_line(0, 0, 0, 0, tags="h_line", fill="white")
        # FPS表示用テキスト
        self.preview_canvas.create_text(
            10, 10, tags="fps_text", anchor="nw", font=("Courier", 20, "bold"), fill=DEFAULT_PREVIEW_TEXT_COLOR
        )
        # トラッカーデータ表示用テキスト
        self.preview_canvas.create_text(
            10, 40, tags="pose_text", anchor="nw", font=("Courier", 20, "bold"), fill=DEFAULT_PREVIEW_TEXT_COLOR
        )

        # --- Camera Device ---
        cam_labelframe = ttk.Labelframe(side_panel, text="Camera Device")
        cam_labelframe.pack(fill="x")

        # カメラ名コンボボックス
        cam_names = get_camera_device_names()
        self.cam_cbx_var = tk.StringVar(value="")
        self.cam_cbx_var.trace("w", self.onchange_cam_cbx)
        self.cam_cbx = ttk.Combobox(cam_labelframe, values=cam_names, textvariable=self.cam_cbx_var, state="readonly")
        self.cam_cbx.pack(fill="x")

        # カメラオプションコンボボックス
        self.cam_option_var = tk.StringVar(value="")
        self.cam_option_var.trace("w", self.onchange_cam_option_cbx)
        self.cam_option_cbx = ttk.Combobox(
            cam_labelframe, values=[], textvariable=self.cam_option_var, state="readonly"
        )
        self.cam_option_cbx.pack(fill="x", pady=(5, 0))

        # プレビュー切り替えボタン
        self.preview_ckb_var = tk.BooleanVar(value=True)
        self.preview_ckb_var.trace("w", self.onchange_preview_ckb)
        self.preview_ckb = ttk.Checkbutton(cam_labelframe, text="Enable Preview", variable=self.preview_ckb_var)
        self.preview_ckb.pack(anchor="w", pady=(5, 0))

        # --- Calibration ---
        cal_labelframe = ttk.LabelFrame(side_panel, text="Calibration")
        cal_labelframe.pack(fill="x", pady=(10, 0))

        # 距離スケールラベル
        self.scale_lbl_var = tk.StringVar()
        self.scale_lbl = ttk.Label(cal_labelframe, textvariable=self.scale_lbl_var)
        self.scale_lbl.pack()

        # 距離スケールスライダー
        self.distance_scale_var = tk.DoubleVar()
        self.distance_scale_var.trace("w", self.onchange_distance_scale)
        self.distance_scale_var.set(DEFAULT_DISTANCE_SCALE)
        self.distance_scale = ttk.Scale(
            cal_labelframe, from_=0.5, to=3.0, orient="horizontal", variable=self.distance_scale_var
        )
        self.distance_scale.pack(fill="x")

        # キャリブレーション開始ボタン
        self.cal_btn = ttk.Button(cal_labelframe, text="Start Calibration", command=self.start_calibration)
        self.cal_btn.pack(fill="x", pady=(20, 0))

        # キャリブレーションプログレスバー
        self.cal_pbar_var = tk.IntVar(value=0)
        self.cal_pbar = ttk.Progressbar(
            cal_labelframe, mode="determinate", maximum=N_CALIBRATION_SAMPLES, variable=self.cal_pbar_var
        )
        self.cal_pbar.pack(fill="x", pady=(5, 0))

        # キャリブレーション結果ラベル
        self.cal_result_lbl_var = tk.StringVar()
        self.update_calibration_ui()
        self.cal_result_lbl = ttk.Label(
            cal_labelframe, textvariable=self.cal_result_lbl_var, style="Calibration.TLabel"
        )
        self.cal_result_lbl.pack(pady=(5, 0))

        # --- UDP Client ---
        self.udp_lf = ttk.LabelFrame(side_panel, text="UDP Client")
        self.udp_lf.pack(fill="x", pady=(10, 0))

        self.udp_frame = ttk.Frame(self.udp_lf)
        self.udp_frame.grid()

        # Host / IPアドレス入力ボックス
        ttk.Label(self.udp_frame, text="Host / IP").grid(row=0, column=0, padx=(0, 10), sticky="e")
        self.udp_host_var = tk.StringVar(value=DEFAULT_UDP_HOST)
        self.udp_host_input = ttk.Entry(self.udp_frame, textvariable=self.udp_host_var, justify="left")
        self.udp_host_input.grid(row=0, column=1, pady=(2, 0))

        # Port入力ボックス
        ttk.Label(self.udp_frame, text="Port").grid(row=1, column=0, padx=(0, 10), pady=(5, 0), sticky="e")
        self.udp_port_var = tk.IntVar(value=DEFAULT_UDP_PORT)
        self.udp_port_input = ttk.Entry(self.udp_frame, textvariable=self.udp_port_var, justify="right")
        self.udp_port_input.grid(row=1, column=1, pady=(5, 0))

        # Applyボタン
        self.udp_apply_btn = ttk.Button(self.udp_frame, text="Apply", command=self.setup_udp_client)
        self.udp_apply_btn.grid(row=2, columnspan=2, sticky="e", pady=(10, 0))

        self.load_config()

        self.after(100, self.setup_udp_client)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.cap:
            self.cap.close()
        if self.track_thread:
            self.track_thread.join()
        if self.udp_pose_writer:
            self.udp_pose_writer.close()
        if self.csv_pose_writer:
            self.csv_pose_writer.close()
        self.tracker.close()
        self.ui_executor.shutdown()

    def on_close(self):
        self.save_config()
        self.close()
        self.root.destroy()

    def save_config(self):
        try:
            udp_port = self.udp_port_var.get()
        except tk.TclError:
            udp_port = ""

        config = configparser.ConfigParser()

        config["Camera"] = {
            "name": self.cam_cbx_var.get(),
            "option": self.cam_option_cbx.get(),
            "enable_preview": self.preview_ckb_var.get(),
        }

        cam_pose = self.corrector.get_cam_pose()
        offset_yaw, offset_pitch, offset_roll = self.corrector.get_offset_angles()
        config["Calibration"] = {
            "is_calibrated": str(self.corrector.is_calibrated()),
            "distance_scale": f"{self.corrector.get_distance_scale():.6f}",
            "cam_x": f"{cam_pose[X]:.6f}",
            "cam_y": f"{cam_pose[Y]:.6f}",
            "cam_z": f"{cam_pose[Z]:.6f}",
            "cam_yaw": f"{cam_pose[YAW]:.6f}",
            "cam_pitch": f"{cam_pose[PITCH]:.6f}",
            "cam_roll": f"{cam_pose[ROLL]:.6f}",
            "offset_yaw": f"{offset_yaw:.6f}",
            "offset_pitch": f"{offset_pitch:.6f}",
            "offset_roll": f"{offset_roll:.6f}",
        }

        config["UDPClient"] = {
            "host": self.udp_host_var.get().strip(),
            "port": udp_port,
        }

        config["UI"] = {
            "preview_text_color": self.preview_text_color,
        }

        def save_config_file(path: Path) -> bool:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    config.write(f)
            except Exception as e:
                logger.debug("Failed to save config to %s", path, exc_info=e)
                return False
            else:
                logger.debug("Saved config to %s", path)
                return True

        # 読み込んだ設定ファイルに保存
        if save_config_file(self.config_path):
            return

        # 保存できなかった場合、別のパスに保存してよいか1度だけ尋ねて保存
        other_path = next((p for p in self.config_paths if p != self.config_path))
        if messagebox.askyesno(
            "Save config file",
            f"Unable to write config.ini to folder where exe is located.\nSave to {other_path.parent} instead?",
        ):
            if save_config_file(other_path):
                self.config_path = other_path
            else:
                messagebox.showerror("Error", "Failed to save config.ini")

    def load_config(self):
        try:
            config = configparser.ConfigParser()

            for path in self.config_paths:
                if not path.exists():
                    continue

                config.read(path, encoding="utf-8")
                logger.debug("Loaded config from %s", path)
                self.config_path = path
                break
            else:
                logger.debug("config.ini not found, using default settings.")
                self.config_path = self.config_paths[-1]
                return

            if "Camera" in config:
                if "enable_preview" in config["Camera"]:
                    self.preview_ckb_var.set(config["Camera"].getboolean("enable_preview"))

                cam_name = config["Camera"]["name"]
                cam_option = config["Camera"]["option"]

                if cam_name and cam_name in self.cam_cbx["values"]:
                    self.cam_cbx_var.set(cam_name)

                    if cam_option and cam_option in self.cam_option_cbx["values"]:
                        self.cam_option_cbx.set(cam_option)

            if "Calibration" in config:
                if "distance_scale" in config["Calibration"]:
                    self.corrector.set_distance_scale(config["Calibration"].getfloat("distance_scale"))

                is_calibrated = config["Calibration"].getboolean("is_calibrated", False)
                if is_calibrated:
                    self.corrector.set_calibrated_data(
                        cam_pose=(
                            config["Calibration"].getfloat("cam_x", 0.0),
                            config["Calibration"].getfloat("cam_y", 0.0),
                            config["Calibration"].getfloat("cam_z", 0.0),
                            config["Calibration"].getfloat("cam_yaw", 0.0),
                            config["Calibration"].getfloat("cam_pitch", 0.0),
                            config["Calibration"].getfloat("cam_roll", 0.0),
                        ),
                        offset_angles=(
                            config["Calibration"].getfloat("offset_yaw", 0.0),
                            config["Calibration"].getfloat("offset_pitch", 0.0),
                            config["Calibration"].getfloat("offset_roll", 0.0),
                        ),
                    )

                self.update_calibration_ui()

            if "UDPClient" in config:
                if "host" in config["UDPClient"]:
                    self.udp_host_var.set(config["UDPClient"]["host"].strip())

                if "port" in config["UDPClient"]:
                    port = config["UDPClient"]["port"].strip()
                    if port:
                        self.udp_port_var.set(int(port))

            if "UI" in config:
                if "preview_text_color" in config["UI"]:
                    self.preview_text_color = config["UI"]["preview_text_color"]
                    self.preview_canvas.itemconfigure("fps_text", fill=self.preview_text_color)
                    self.preview_canvas.itemconfigure("pose_text", fill=self.preview_text_color)
        except Exception:
            error_msg = f"Failed to load config from {self.config_path}"
            logging.exception(error_msg)
            messagebox.showerror("Error", error_msg)

    def track_loop(self):
        try:
            for frame in self.cap.frames():
                self.track_loop_inner(frame, int(time.perf_counter() * 1000))
        except Exception as e:
            logger.exception("track loop unexpected error")

            cap = self.cap
            if cap:
                threading.Thread(target=cap.close, daemon=True).start()

            err_msg = "".join(traceback.format_exception(e))
            self.ui_executor.submit(messagebox.showerror, "Error", err_msg)
            self.ui_executor.schedule("hide_preview_canvas", self.hide_preview_canvas)
            self.ui_executor.schedule("cam_option_var.set", self.cam_option_var.set, "")

    def track_loop_inner(self, frame: np.ndarray, timestamp_ms: int):
        self.fps_counter.update()

        # 顔ランドマーク検出
        tracker_result = self.tracker.estimate(frame, timestamp_ms)
        pose: Pose | None = None

        # 顔ランドマークが検出できた場合の処理
        if tracker_result:
            # キャリブレーション中の場合
            if self.is_calibrating:
                if self.corrector.add_calibration_sample(tracker_result.matrix):
                    sample_len = self.corrector.get_calibration_sample_len()

                    # プログレスバーを更新
                    self.ui_executor.schedule("cal_pbar_var.set", self.cal_pbar_var.set, sample_len)

                    # サンプルデータ数が規定数を超えたらキャリブレーションを実行
                    if sample_len >= N_CALIBRATION_SAMPLES:
                        self.is_calibrating = False
                        self.corrector.calibrate()
                        self.ui_executor.schedule("update_calibration_ui", self.update_calibration_ui)

            # データを補正
            pose = self.corrector.correct(tracker_result.matrix)

            # データをUDP送信
            if self.udp_pose_writer:
                self.udp_pose_writer.write(pose)

            # データをCSV出力
            if self.csv_pose_writer:
                self.csv_pose_writer.write(pose)

        # プレビューの更新処理
        if self.should_preview and self.preview_canvas_height > 1:
            preview_pil_image = Image.fromarray(frame, mode="RGB")

            # キャンバスの高さに合わせてリサイズ
            preview_scale = self.preview_canvas_height / frame.shape[0]
            preview_pil_image = preview_pil_image.resize(
                (int(preview_pil_image.width * preview_scale), self.preview_canvas_height), Image.Resampling.NEAREST
            )
            preview_width, preview_height = preview_pil_image.size

            # 顔ランドマークが検出できた場合、検出した顔の輪郭を描画する。
            if tracker_result and tracker_result.landmarks:
                points = [(round(x * preview_width), round(y * preview_height)) for x, y in tracker_result.landmarks]
                image_draw = ImageDraw.Draw(preview_pil_image)
                image_draw.line(points, fill=(0, 255, 0), width=2)

            # プレビューの更新
            self.ui_executor.schedule("update_preview_canvas", self.update_preview_canvas, preview_pil_image, pose)

    def update_preview_canvas(self, pil_image: Image.Image, pose: np.ndarray | None):
        tk_image = ImageTk.PhotoImage(pil_image)

        if self.should_resize_preview:
            self.should_resize_preview = False

            # 画像のサイズに合わせてキャンバスのサイズと中心線の位置を変更
            w, h = tk_image.width(), tk_image.height()
            cx, cy = w // 2, h // 2
            self.preview_canvas.configure(width=w, height=h)
            self.preview_canvas.coords("v_line", cx, 0, cx, h)
            self.preview_canvas.coords("h_line", 0, cy, w, cy)

        self.preview_canvas_tk_image = tk_image
        self.preview_canvas.itemconfigure("image", image=tk_image)

        fps = f"{self.fps_counter.get_fps():.0f} FPS"
        self.preview_canvas.itemconfigure("fps_text", text=fps)

        if pose is not None:
            self.ui_executor.schedule(
                "update_pose_text",
                self.preview_canvas.itemconfigure,
                "pose_text",
                {
                    "text": (
                        f"X     {pose[X]:>5.1f} cm\n"
                        f"Y     {pose[Y]:>5.1f} cm\n"
                        f"Z     {pose[Z]:>5.1f} cm\n"
                        f"Yaw   {pose[YAW]:>5.1f} °\n"
                        f"Pitch {pose[PITCH]:>5.1f} °\n"
                        f"Roll  {pose[ROLL]:>5.1f} °"
                    )
                },
            )

    def update_calibration_ui(self):
        if self.is_calibrating:
            self.cal_btn.config(text="Stop Calibration")
        else:
            self.cal_btn.config(text="Start Calibration")

        self.distance_scale_var.set(self.corrector.get_distance_scale())

        if self.corrector.is_calibrated():
            self.cal_pbar_var.set(self.cal_pbar["maximum"])

            cam_pose = self.corrector.get_cam_pose()
            offset_yaw, offset_pitch, offset_roll = self.corrector.get_offset_angles()

            self.cal_result_lbl_var.set(
                f"""
Camera X      {cam_pose[X]:>5.1f} cm
Camera Y      {cam_pose[Y]:>5.1f} cm
Camera Z      {cam_pose[Z]:>5.1f} cm
Camera Yaw    {cam_pose[YAW]:>5.1f} °
Camera Pitch  {cam_pose[PITCH]:>5.1f} °
Camera Roll   {cam_pose[ROLL]:>5.1f} °
Offset Yaw    {offset_yaw:>5.1f} °
Offset Pitch  {offset_pitch:>5.1f} °
Offset Roll   {offset_roll:>5.1f} °
""".strip()
            )
        else:
            self.cal_pbar_var.set(0)

            self.cal_result_lbl_var.set(
                """
Camera X       --.- cm
Camera Y       --.- cm
Camera Z       --.- cm
Camera Yaw     --.- °
Camera Pitch   --.- °
Camera Roll    --.- °
Offset Yaw     --.- °
Offset Pitch   --.- °
Offset Roll    --.- °
""".strip()
            )

    def show_preview_canvas(self):
        self.preview_canvas.pack(fill="both", expand=True, padx=(10, 0))

        self.ui_executor.set_interval_ms(10)

    def hide_preview_canvas(self):
        self.preview_canvas.pack_forget()
        self.preview_canvas_tk_image = None

        self.ui_executor.set_interval_ms(100)

    def start_calibration(self):
        self.corrector.reset_calibration()
        self.corrector.set_distance_scale(self.distance_scale_var.get())
        self.cal_pbar_var.set(0)
        self.is_calibrating = not self.is_calibrating
        self.update_calibration_ui()

    def setup_udp_client(self):
        if self.udp_pose_writer:
            self.udp_pose_writer.close()

        host = self.udp_host_var.get().strip()

        if not host:
            return

        try:
            port = self.udp_port_var.get()
        except tk.TclError as e:
            logger.debug("Invalid UDP Port: %s", e)
            messagebox.showerror("Error", "Invalid Port")
            return

        try:
            self.udp_pose_writer = UDPPoseWriter(host=host, port=port)
        except socket.gaierror:
            messagebox.showerror("Error", "Invalid Host / IP")

    def onresize_preview_canvas(self, _):
        height = self.preview_canvas.winfo_height()
        if self.preview_canvas_height != height:
            self.preview_canvas_height = height
            self.should_resize_preview = True

    def onchange_cam_cbx(self, *_):
        selected_cam_name = self.cam_cbx_var.get()
        self.cam_options = get_camera_device_options(selected_cam_name)
        cam_options_cbx_list = [
            f"{o.format}, {o.width}x{o.height}, {o.fps:.1f}fps" for i, o in enumerate(self.cam_options)
        ]
        self.cam_option_var.set("")
        self.cam_option_cbx.configure(values=cam_options_cbx_list)

    def onchange_cam_option_cbx(self, *_):
        # 1. プレビューを非表示
        self.hide_preview_canvas()

        # 2. キャプチャを閉じる
        if self.cap:
            self.cap.close()

        # 3. トラッキングの終了を待機
        if self.track_thread:
            self.track_thread.join()

        # 4. 未選択状態の場合、終了
        selected_index = self.cam_option_cbx.current()
        if selected_index == -1:
            return

        cam_option = self.cam_options[selected_index]

        # 5. 新しいキャプチャを開始
        self.cap = VideoCapture(
            name=cam_option.name,
            input_format_type=cam_option.format_type,
            input_format=cam_option.format,
            width=cam_option.width,
            height=cam_option.height,
            fps=cam_option.fps,
        )

        # 6. 初期化
        self.should_resize_preview = True
        self.ui_executor.clear()
        self.fps_counter.reset()
        self.preview_canvas.itemconfig(
            "pose_text",
            text="""
X      --.- cm
Y      --.- cm
Z      --.- cm
Yaw    --.- °
Pitch  --.- °
Roll   --.- °
""".strip(),
        )

        # 7. トラッキングを開始
        self.track_thread = threading.Thread(target=self.track_loop, daemon=True)
        self.track_thread.start()

        # 8. プレビューを表示
        if self.should_preview:
            self.show_preview_canvas()

    def onchange_preview_ckb(self, *_):
        self.should_preview = self.preview_ckb_var.get()
        if self.should_preview:
            self.show_preview_canvas()
        else:
            self.hide_preview_canvas()

    def onchange_distance_scale(self, *_):
        distance_scale = self.distance_scale_var.get()
        self.scale_lbl_var.set(f"Distance Scale: {distance_scale:.2f}")
        self.corrector.set_distance_scale(distance_scale)
