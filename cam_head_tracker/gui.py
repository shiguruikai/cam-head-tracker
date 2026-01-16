import configparser
import itertools
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

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from cam_head_tracker.camera import CameraDeviceOption, VideoCapture, get_camera_device_names, get_camera_device_options
from cam_head_tracker.frame_rate_counter import FrameRateCounter
from cam_head_tracker.tracker import PoseCorrector, create_face_landmarker, mp_matrix_to_pose
from cam_head_tracker.udp_client import UDPClient

logger = logging.getLogger(__name__)

APP_NAME = "Head Tracker"
ICON_FILE_PATH = Path(__file__).parent / "assets/icon.png"

DEFAULT_DISTANCE_SCALE = 1.3
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 4242
DEFAULT_PREVIEW_TEXT_COLOR = "#00FF00"

# キャリブレーションに必要なサンプルデータ数
N_CALIBRATION_SAMPLES = 30

# プレビューで描画する顔面の線
PREVIEW_FACE_LANDMARKS_CONNECTIONS = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL + [
    mp.tasks.vision.FaceLandmarksConnections.Connection(a, b)
    for a, b in itertools.pairwise(
        [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]
    )
]

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
    def __init__(self, root: tk.Tk, config_path: Path, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        self.config_path = config_path

        self.is_calibrating = False
        self.should_preview = True
        self.should_resize_preview = True
        self.preview_canvas_height = 0
        self.preview_text_color = DEFAULT_PREVIEW_TEXT_COLOR
        self.cam_options: list[CameraDeviceOption] = []
        self.cap: VideoCapture | None = None
        self.track_thread: threading.Thread | None = None
        self.udp_client: UDPClient | None = None

        self.face_landmarker = create_face_landmarker()
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
        if self.udp_client:
            self.udp_client.close()
        self.face_landmarker.close()
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
        config["Calibration"] = {
            "is_calibrated": str(self.corrector.is_calibrated()),
            "distance_scale": self.corrector.get_distance_scale(),
            "cam_angle": self.corrector.get_cam_angle(),
            "cam_height": self.corrector.get_cam_height(),
            "offset_pitch": self.corrector.get_offset_pitch(),
        }
        config["UDPClient"] = {
            "host": self.udp_host_var.get().strip(),
            "port": udp_port,
        }
        config["UI"] = {
            "preview_text_color": self.preview_text_color,
        }

        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                config.write(f)
        except Exception:
            logger.exception("Failed to save config.ini")
            messagebox.showerror("Error", "Failed to save config.ini")

    def load_config(self):
        try:
            config = configparser.ConfigParser()
            config.read(self.config_path, encoding="utf-8")

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
                if "is_calibrated" in config["Calibration"]:
                    self.corrector.set_calibrated(config["Calibration"].getboolean("is_calibrated"))

                if self.corrector.is_calibrated():
                    if "distance_scale" in config["Calibration"]:
                        self.corrector.set_distance_scale(config["Calibration"].getfloat("distance_scale"))
                    if "cam_angle" in config["Calibration"]:
                        self.corrector.set_cam_angle(config["Calibration"].getfloat("cam_angle"))
                    if "cam_height" in config["Calibration"]:
                        self.corrector.set_cam_height(config["Calibration"].getfloat("cam_height"))
                    if "offset_pitch" in config["Calibration"]:
                        self.corrector.set_offset_pitch(config["Calibration"].getfloat("offset_pitch"))

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
            logging.exception("Failed to load config.ini")
            messagebox.showerror("Error", "Failed to load config.ini")

    def track_loop(self):
        try:
            for frame in self.cap.frames():
                self.track_loop_inner(frame)
        except Exception as e:
            logger.exception("track loop unexpected error")

            cap = self.cap
            if cap:
                threading.Thread(target=cap.close, daemon=True).start()

            err_msg = "".join(traceback.format_exception(e))
            self.ui_executor.submit(messagebox.showerror, "Error", err_msg)
            self.ui_executor.schedule("hide_preview_canvas", self.hide_preview_canvas)
            self.ui_executor.schedule("cam_option_var.set", self.cam_option_var.set, "")

    def track_loop_inner(self, frame):
        self.fps_counter.update()

        # 顔ランドマーク検出
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.perf_counter() * 1000)
        landmarker_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose: np.ndarray | None = None

        # 顔ランドマークが検出できた場合の処理
        if landmarker_result.facial_transformation_matrixes:
            raw_pose = mp_matrix_to_pose(landmarker_result.facial_transformation_matrixes[0])

            # キャリブレーション中の場合
            if self.is_calibrating:
                if self.corrector.add_calibration_sample(raw_pose):
                    sample_len = self.corrector.get_calibration_sample_len()

                    # プログレスバーを更新
                    self.ui_executor.schedule("cal_pbar_var.set", self.cal_pbar_var.set, sample_len)

                    # サンプルデータ数が規定数を超えたらキャリブレーションを実行
                    if sample_len >= N_CALIBRATION_SAMPLES:
                        self.is_calibrating = False
                        self.corrector.calibrate()
                        self.ui_executor.schedule("update_calibration_ui", self.update_calibration_ui)

            # データを補正
            pose = self.corrector.correct(raw_pose)

            # データをUDP送信
            udp_client = self.udp_client
            if udp_client:
                udp_client.send(pose)

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
            if landmarker_result.face_landmarks:
                landmarks = landmarker_result.face_landmarks[0]
                draw = ImageDraw.Draw(preview_pil_image)

                points = []
                for conn in PREVIEW_FACE_LANDMARKS_CONNECTIONS:
                    for i in (conn.start, conn.end):
                        landmark = landmarks[i]
                        x = round(landmark.x * preview_width)
                        y = round(landmark.y * preview_height)
                        points.append((x, y))
                draw.line(points, fill=(0, 255, 0), width=2)

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
        self.distance_scale_var.set(self.corrector.get_distance_scale())

        if self.corrector.is_calibrated():
            self.cal_pbar_var.set(self.cal_pbar["maximum"])

            self.cal_result_lbl_var.set(
                f"""
Camera Angle  {self.corrector.get_cam_angle():>5.1f} °
Camera Height {self.corrector.get_cam_height():>5.1f} cm
Offset Pitch  {self.corrector.get_offset_pitch():>5.1f} °
""".strip()
            )
        else:
            self.cal_pbar_var.set(0)
            self.cal_result_lbl_var.set(
                """
Camera Angle   --.-°
Camera Height  --.- cm
Offset Pitch   --.- °
""".strip()
            )

        if self.is_calibrating:
            self.cal_btn.config(text="Stop Calibration")
        else:
            self.cal_btn.config(text="Start Calibration")

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
        if self.udp_client:
            self.udp_client.close()
            self.udp_client = None

        host = self.udp_host_var.get().strip()

        if not host:
            return

        try:
            port = self.udp_port_var.get()
        except tk.TclError as e:
            logger.debug("Invalid UDP Port: %s", e)
            messagebox.showerror("Error", "Invalid UDP Client Port")
            return

        try:
            self.udp_client = UDPClient(host, port)
        except socket.gaierror as e:
            logger.debug("Failed to create UDPClient: host=%s, port=%s, error=%s", host, port, e)
            messagebox.showerror("Error", "Invalid UDP Client Host / IP")

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
