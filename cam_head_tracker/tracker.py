import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarksConnections

FACE_LANDMARKER_MODEL_PATH = Path(__file__).parent / "assets/face_landmarker.task"

X, Y, Z, YAW, PITCH, ROLL = 0, 1, 2, 3, 4, 5

Pose = tuple[float, float, float, float, float, float]
Angles = tuple[float, float, float]
Landmark = tuple[float, float]


def rotation_matrix_to_euler_angles(matrix: np.ndarray) -> Angles:
    """
    3x3 回転行列をオイラー角（Yaw, Pitch, Roll）に変換します。単位は度 (degree) です。
    ZYX順序に基づいて分解します。
    """
    pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
    yaw = math.degrees(math.atan2(-matrix[2, 0], math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)))
    roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return yaw, pitch, roll


def euler_angles_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    オイラー角（Yaw, Pitch, Roll）を 3x3 回転行列に変換します。単位は度 (degree) です。
    ZYX順序で回転行列を合成します。
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(p), -math.sin(p)],
            [0, math.sin(p), math.cos(p)],
        ]
    )
    r_y = np.array(
        [
            [math.cos(y), 0, math.sin(y)],
            [0, 1, 0],
            [-math.sin(y), 0, math.cos(y)],
        ]
    )
    r_z = np.array(
        [
            [math.cos(r), -math.sin(r), 0],
            [math.sin(r), math.cos(r), 0],
            [0, 0, 1],
        ]
    )
    # 適用順序: Rx -> Ry -> Rz
    return r_z @ (r_y @ r_x)


@dataclass(slots=True)
class TrackerResult:
    """トラッキング結果を保持するデータクラス"""

    matrix: np.ndarray  # 4x4 同次変換行列
    landmarks: list[Landmark]  # 正規化されたランドマーク座標


class Tracker(Protocol):
    """トラッカーのインターフェース定義"""

    def estimate(self, frame: np.ndarray, timestamp_ms: int) -> TrackerResult | None: ...
    def close(self) -> None: ...


class MediapipeTracker(Tracker):
    """MediaPipe Face Landmarker を使用したトラッカーの実装"""

    # プレビュー表示用に抽出するランドマークの接続定義
    PREVIEW_FACE_LANDMARKS_CONNECTIONS = FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL + [
        FaceLandmarksConnections.Connection(a, b)
        for a, b in itertools.pairwise(
            [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]
        )
    ]
    # 上記接続で使用されるランドマークのインデックス一覧
    LANDMARK_INDEXES = [i for conn in PREVIEW_FACE_LANDMARKS_CONNECTIONS for i in (conn.start, conn.end)]

    def __init__(self):
        self._landmarker = self._create_face_landmarker()

    def estimate(self, frame: np.ndarray, timestamp_ms: int) -> TrackerResult | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.facial_transformation_matrixes or not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        # MediaPipeが出力するのは 4x4 同次変換行列
        return TrackerResult(
            matrix=result.facial_transformation_matrixes[0],
            landmarks=[(landmarks[i].x, landmarks[i].y) for i in self.LANDMARK_INDEXES],
        )

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def _create_face_landmarker() -> FaceLandmarker:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
        )
        return FaceLandmarker.create_from_options(options)


class PoseCorrector:
    """
    カメラの設置位置や角度をキャリブレーションし、
    トラッキングデータを「モニター正面基準のワールド座標系」に補正するクラス。

    キャリブレーション手順:
        1. モニターに正対し、頭を垂直（ロール角ゼロ）に保つ。
        2. モニターの正面方向（Z軸）に沿って、真っ直ぐ前後に平行移動する。

    補足:
        - Mediapipeの出力に合わせて内部計算には dtype=np.float32 を使用する。
    """

    def __init__(self):
        self._is_calibrated = False
        self._calibration_samples: list[np.ndarray] = []
        self._distance_scale = 1.0

        # カメラ座標系のベクトルをワールド座標系（モニター基準）へ投影するための基底変換行列
        self._camera_to_world_rot = np.eye(3, dtype=np.float32)

        # ワールド座標系における原点オフセット（位置補正用ベクトル）
        self._origin_offset = np.zeros(3, dtype=np.float32)

        # 初期姿勢（正対状態）を角度ゼロとするための回転オフセット行列
        self._zero_rot_offset = np.eye(3, dtype=np.float32)

        # 事前計算用の回転オフセット行列
        self._combined_rot_offset = self._zero_rot_offset @ self._camera_to_world_rot

    def reset_calibration(self):
        """キャリブレーション状態を初期化します。"""
        self._is_calibrated = False
        self._calibration_samples = []
        self._camera_to_world_rot = np.eye(3, dtype=np.float32)
        self._origin_offset = np.zeros(3, dtype=np.float32)
        self._zero_rot_offset = np.eye(3, dtype=np.float32)
        self._combined_rot_offset = self._zero_rot_offset @ self._camera_to_world_rot

    def is_calibrated(self) -> bool:
        """キャリブレーション済みかどうかを返します。"""
        return self._is_calibrated

    def get_distance_scale(self) -> float:
        return self._distance_scale

    def set_distance_scale(self, scale: float):
        """移動距離の倍率を設定します。"""
        self._distance_scale = scale

    def get_calibration_sample_len(self) -> int:
        """現在蓄積されているサンプルデータ数を返します。"""
        return len(self._calibration_samples)

    def get_cam_pose(self) -> Pose:
        """推定されたカメラの設置姿勢（位置とオイラー角）を返します。"""
        # カメラの回転は、ワールド座標系への基底変換行列から算出できる。
        yaw, pitch, roll = rotation_matrix_to_euler_angles(self._camera_to_world_rot)
        return self._origin_offset[X], self._origin_offset[Y], self._origin_offset[Z], yaw, pitch, roll

    def get_offset_angles(self) -> Angles:
        yaw, pitch, roll = rotation_matrix_to_euler_angles(self._zero_rot_offset)
        return yaw, pitch, roll

    def set_calibrated_data(self, cam_pose: Pose, offset_angles: Angles):
        """外部保存されたキャリブレーションデータを復元します。"""
        self._is_calibrated = True

        # 位置の復元
        self._origin_offset[X] = cam_pose[X]
        self._origin_offset[Y] = cam_pose[Y]
        self._origin_offset[Z] = cam_pose[Z]

        # 回転行列の再構築
        self._camera_to_world_rot = euler_angles_to_rotation_matrix(
            cam_pose[YAW], cam_pose[PITCH], cam_pose[ROLL]
        ).astype(np.float32)
        self._zero_rot_offset = euler_angles_to_rotation_matrix(
            offset_angles[0], offset_angles[1], offset_angles[2]
        ).astype(np.float32)

        # 事前計算用の回転オフセット行列の更新
        self._combined_rot_offset = self._zero_rot_offset @ self._camera_to_world_rot

    def add_calibration_sample(self, matrix: np.ndarray) -> bool:
        """
        キャリブレーション用サンプルを追加します。
        前回データからZ軸方向（奥行き）に一定以上動いた場合のみ採用します。

        :param matrix: MediaPipeが出力した 4x4 同次変換行列
        :return: サンプルが追加された場合はTrue
        """
        mat = matrix.copy()

        # X, Y, Z にスケール適用
        mat[:3, 3] *= self._distance_scale

        if self._calibration_samples:
            z = mat[2, 3]
            last_z = self._calibration_samples[-1][2, 3]

            # Z軸の変化が小さい場合は無視（ノイズ対策およびデータ偏重の防止）
            if abs(last_z - z) < 0.5:  # 0.5cm
                return False

        self._calibration_samples.append(mat)
        return True

    def calibrate(self):
        """
        蓄積されたサンプルデータに基づき、カメラの設置角度と位置を推定（キャリブレーション）します。

        アルゴリズム概要:
        1. 【回転】データの平均から「頭の上方向」と「正面方向」を特定。
        2. 【移動】データの主成分分析(PCA)で「前後移動の軸（Z軸）」を特定。
        3. 【基底】上記から直交するX, Y, Z軸（ワールド座標系）を構築。
        4. 【オフセット】平均位置・姿勢が「原点・ゼロ」になるよう補正値を計算。
        """
        if not self._calibration_samples:
            return

        matrices = np.array(self._calibration_samples)
        positions = matrices[:, :3, 3]  # 位置ベクトル群 (N, 3)
        rotations = matrices[:, :3, :3]  # 回転行列群 (N, 3, 3)

        # --- Step 1: カメラ座標系における平均的な頭の向きを算出 ---

        # 単純平均ではなくSVDを使って回転行列の平均を求めます。
        u, _, vh = np.linalg.svd(np.sum(rotations, axis=0))
        avg_head_rot = u @ vh  # 平均回転行列

        # MediaPipeのローカル軸に基づく方向ベクトルを抽出
        head_up_vec = avg_head_rot[:, 1]  # 頭の上方向
        head_forward_vec = avg_head_rot[:, 2]  # 頭の正面方向

        # --- Step 2: 移動データからワールド座標系のZ軸を決定 ---

        # 平均位置ベクトル
        mean_pos = np.mean(positions, axis=0)

        # 中心化した位置ベクトル群から、SVDで分散が最大の方向（第1主成分）を抽出
        centered_positions = positions - mean_pos
        _, _, vh = np.linalg.svd(centered_positions)
        moving_axis = vh[0]  # 最大分散方向（移動軸）の単位ベクトル
        moving_axis /= np.linalg.norm(moving_axis)

        # 移動軸の向きを頭の正面方向に合わせる。
        if moving_axis @ head_forward_vec < 0:
            moving_axis *= -1  # 内積が負なら逆向きなので反転

        world_z_axis = moving_axis  # ワールド座標系のZ軸(正面)

        # --- Step 3: 正規直交基底の構築 ---

        # X軸(左) = Y軸(頭上) * Z軸(正面)
        world_x_axis = np.cross(head_up_vec, world_z_axis)
        world_x_axis /= np.linalg.norm(world_x_axis)

        # Y軸(上) = Z軸(正面) * X軸(左) （直交性を保証するために再計算）
        world_y_axis = np.cross(world_z_axis, world_x_axis)
        world_y_axis /= np.linalg.norm(world_y_axis)

        # カメラ座標系のベクトルをワールド座標系（モニター基準）へ投影するための基底変換行列
        # 各行にワールド系の単位基底ベクトルを配置することで、行列積による座標変換を可能にする。
        self._camera_to_world_rot = np.array([world_x_axis, world_y_axis, world_z_axis], dtype=np.float32)

        # --- Step 4: 原点（位置）と初期姿勢（回転）のオフセットを算出 ---

        # 1. 位置オフセット
        # カメラ座標系の平均位置をワールド座標系へ投影し、そのXY成分を打ち消すベクトルをオフセットとする。
        mean_pos_world = self._camera_to_world_rot @ mean_pos
        self._origin_offset = -mean_pos_world
        self._origin_offset[Z] = 0  # Z（モニターとの距離）は維持

        # 2. 回転オフセット
        # ワールド座標系における頭の平均的な傾きを算出し、それを打ち消す逆回転（転置行列）をオフセットとする。
        avg_rot_world = self._camera_to_world_rot @ avg_head_rot
        self._zero_rot_offset = avg_rot_world.T

        # correct()の計算負荷軽減のため、基底変換と回転オフセットを合成した行列を事前に計算する。
        self._combined_rot_offset = self._zero_rot_offset @ self._camera_to_world_rot

        self._is_calibrated = True
        self._calibration_samples = []

    def correct(self, matrix: np.ndarray) -> Pose:
        """
        MediaPipeからの同次変換行列を、キャリブレーション結果を用いて補正します。

        :param matrix: MediaPipeから得られた生の 4x4 同次変換行列
        :return: 補正済みの姿勢データ (x, y, z, yaw, pitch, roll)
        """
        # カメラ座標系における位置ベクトル
        raw_pos = matrix[:3, 3] * self._distance_scale
        # カメラ座標系における回転行列
        raw_rot = matrix[:3, :3]

        # カメラ座標系からワールド座標系へ位置ベクトルを変換（剛体変換）
        # カメラの設置角度に基づく回転 + キャリブレーション位置を原点とする並進
        world_pos = self._camera_to_world_rot @ raw_pos + self._origin_offset

        # 回転行列をワールド基準に変換し、同時に初期姿勢を角度ゼロへと正規化
        # ワールド座標系回転行列 = (回転オフセット行列 @ 基底変換行列) @ カメラ座標系回転行列
        world_rot = self._combined_rot_offset @ raw_rot

        # オイラー角へ変換
        yaw, pitch, roll = rotation_matrix_to_euler_angles(world_rot)

        # opentrackの入力仕様に合わせ、符号を調整
        # 出力座標系: X(左), Y(上), Z(手前), Yaw(右), Pitch(上), Roll(反時計)
        return float(world_pos[X]), float(world_pos[Y]), -float(world_pos[Z]), -yaw, -pitch, -roll
