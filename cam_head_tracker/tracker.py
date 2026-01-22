# tracker.py

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
    ZYX順序 (R = Rz @ Ry @ Rx) に基づいて分解します。
    """
    pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
    yaw = math.degrees(math.atan2(-matrix[2, 0], math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)))
    roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return yaw, pitch, roll


def euler_angles_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    オイラー角（Yaw, Pitch, Roll）を 3x3 回転行列に変換します。単位は度 (degree) です。
    ZYX順序 (R = Rz @ Ry @ Rx) で回転行列を合成します。
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
    カメラの設置位置や角度をキャリブレーションによって推定し、
    トラッキングデータをモニター正面基準のワールド座標系に補正するクラス。

    キャリブレーション中の前提条件:
        1. モニターに向かって正対し、頭を垂直（ロール角ゼロ）に保つこと。
        2. モニターの正面方向（Z軸）に沿って真っ直ぐ前または後に平行移動すること。

    補足:
        - Mediapipeの出力に合わせて、dtype=np.float32 を使用する。
    """

    def __init__(self):
        self._is_calibrated = False
        self._calibration_samples: list[np.ndarray] = []  # 4x4 同次変換行列のリスト
        self._distance_scale = 1.0

        # カメラ座標系からワールド座標系への基底変換を行う 3x3 回転行列
        self._camera_to_world_rot = np.eye(3, dtype=np.float32)
        # ワールド座標系におけるカメラ位置ベクトル（平行移動ベクトル）
        self._cam_pos = np.zeros(3, dtype=np.float32)
        # 正面を向くための 3x3 回転行列（初期姿勢の逆行列）
        self._head_zero_rot = np.eye(3, dtype=np.float32)

        # 回転補正の事前計算用行列
        self._combined_rot_offset = self._head_zero_rot @ self._camera_to_world_rot

    def reset_calibration(self):
        """キャリブレーション状態をリセットします。"""
        self._is_calibrated = False
        self._calibration_samples = []
        self._camera_to_world_rot = np.eye(3, dtype=np.float32)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._head_zero_rot = np.eye(3, dtype=np.float32)
        self._combined_rot_offset = self._head_zero_rot @ self._camera_to_world_rot

    def is_calibrated(self) -> bool:
        """キャリブレーション済みかどうかを返します。"""
        return self._is_calibrated

    def get_distance_scale(self) -> float:
        return self._distance_scale

    def set_distance_scale(self, v: float):
        """移動距離の倍率を設定します。"""
        self._distance_scale = v

    def get_calibration_sample_len(self) -> int:
        """現在収集済みのキャリブレーションサンプル数を返します。"""
        return len(self._calibration_samples)

    def get_cam_pose(self) -> Pose:
        """推定されたカメラの設置姿勢（位置ベクトルとオイラー角）を返します。"""
        # カメラの回転は、ワールド座標系への基底変換行列の逆変換（転置行列）に相当
        yaw, pitch, roll = rotation_matrix_to_euler_angles(self._camera_to_world_rot)
        return self._cam_pos[X], self._cam_pos[Y], self._cam_pos[Z], yaw, pitch, roll

    def get_offset_angles(self) -> Angles:
        yaw, pitch, roll = rotation_matrix_to_euler_angles(self._head_zero_rot)
        return yaw, pitch, roll

    def set_calibrated_data(self, cam_pose: Pose, offset_angles: Angles):
        self._is_calibrated = True

        self._cam_pos[X] = cam_pose[X]
        self._cam_pos[Y] = cam_pose[Y]
        self._cam_pos[Z] = cam_pose[Z]

        self._camera_to_world_rot = euler_angles_to_rotation_matrix(
            cam_pose[YAW], cam_pose[PITCH], cam_pose[ROLL]
        ).astype(np.float32)

        self._head_zero_rot = euler_angles_to_rotation_matrix(
            offset_angles[0], offset_angles[1], offset_angles[2]
        ).astype(np.float32)

        # 事前計算用行列の更新
        self._combined_rot_offset = self._head_zero_rot @ self._camera_to_world_rot

    def add_calibration_sample(self, matrix: np.ndarray) -> bool:
        """
        キャリブレーション用のサンプルデータを追加します。
        データの偏りを防ぐため、前回追加したサンプルから一定の距離以上動いた場合にのみ追加します。

        :param matrix: MediaPipeが出力した 4x4 同次変換行列
        :return: サンプルが追加された場合はTrue
        """
        mat = matrix.copy()

        # 平行移動成分（位置ベクトル）にスケールを適用
        mat[:3, 3] *= self._distance_scale

        if self._calibration_samples:
            z = mat[2, 3]
            last_z = self._calibration_samples[-1][2, 3]

            # Z軸（奥行き）の変化が小さい場合、サンプルとして採用しない
            if abs(last_z - z) < 0.5:  # 0.5cm
                return False

        self._calibration_samples.append(mat)
        return True

    def calibrate(self):
        """収集したサンプルデータに基づき、カメラの設置角度と位置を推定します。"""
        if not self._calibration_samples:
            return

        matrices = np.array(self._calibration_samples)
        positions = matrices[:, :3, 3]  # 位置ベクトル群 (N, 3)
        rotations = matrices[:, :3, :3]  # 回転行列群 (N, 3, 3)

        # ----------------------------------------------------------------------
        # Step 1: 基準となる頭の方向ベクトルを算出
        # ----------------------------------------------------------------------
        # キャリブレーション中は「正対して頭を垂直にしている」と仮定し、
        # 平均的な頭の上方向ベクトルを「ワールド座標系の上方向」の基準として利用します。
        rot_sum = np.sum(rotations, axis=0)
        u, _, vh = np.linalg.svd(rot_sum)
        avg_head_rot = u @ vh  # 平均回転行列

        # MediaPipeの回転行列の列ベクトルは、それぞれのローカル軸の方向ベクトルを表す
        head_y_vec = avg_head_rot[:, 1]  # 頭の上方向ベクトル (Up)
        head_z_vec = avg_head_rot[:, 2]  # 頭の正面方向ベクトル (Forward/Back)

        # ----------------------------------------------------------------------
        # Step 2: 平行移動データからワールド座標系のZ軸（奥行き）を決定
        # ----------------------------------------------------------------------
        # 位置ベクトルの主成分分析（SVD）を行い、分散が最大となる軸（＝ユーザーの移動方向ベクトル）を抽出します。
        centered_pos = positions - np.mean(positions, axis=0)
        _, _, vh = np.linalg.svd(centered_pos)
        moving_axis = vh[0]  # 第1主成分ベクトル

        # 移動方向ベクトルの符号（前後）を頭の向きと合わせる。
        if moving_axis @ head_z_vec < 0:
            moving_axis *= -1

        # 正規化して、ワールド座標系のZ軸単位ベクトルを定義します。
        world_z_axis = moving_axis / np.linalg.norm(moving_axis)

        # ----------------------------------------------------------------------
        # Step 3: ワールド座標系のX軸（水平）とY軸（垂直）を決定
        # ----------------------------------------------------------------------
        # 「頭の上方向ベクトル」と「Z軸単位ベクトル」の外積から、それらに直交するX軸単位ベクトルを求めます。
        world_x_axis = np.cross(head_y_vec, world_z_axis)
        world_x_axis /= np.linalg.norm(world_x_axis)

        # 決定したZ軸とX軸から、それらに直交するY軸単位ベクトルを再計算します（正規直交基底の完成）。
        # ベクトル三重積の性質上、このY軸は必ず head_y_vec と鋭角（同じ向き）になります。
        world_y_axis = np.cross(world_z_axis, world_x_axis)

        # ----------------------------------------------------------------------
        # Step 4: カメラ座標系からワールド座標系への基底変換行列を構築
        # ----------------------------------------------------------------------
        # 算出した基底ベクトルを並べ、カメラ座標系からワールド座標系へ変換する 3x3 回転行列を作成します。
        self._camera_to_world_rot = np.array([world_x_axis, world_y_axis, world_z_axis])

        # ----------------------------------------------------------------------
        # Step 5: 位置と回転の原点（オフセット）を決定
        # ----------------------------------------------------------------------
        # 1. 位置のオフセット:
        #    補正後の位置 P_world = R @ P_raw + cam_pos において、
        #    キャリブレーション時の平均位置で XY が 0 になるように cam_pos (平行移動ベクトル) を設定します。
        rotated_positions = positions @ self._camera_to_world_rot.T
        self._cam_pos = -np.mean(rotated_positions, axis=0)
        self._cam_pos[Z] = 0  # Z軸はモニターからの距離を残すため、オフセットは0とします。

        # 2. 回転のオフセット:
        #    キャリブレーション時の平均的な頭の向きを「正面（角度ゼロ）」とするための回転行列を計算します。
        #    これは、ワールド座標系に変換された後の平均回転行列の逆行列（転置行列）となります。
        aligned_rotations = self._camera_to_world_rot @ rotations
        rot_sum = np.sum(aligned_rotations, axis=0)
        u, _, vh = np.linalg.svd(rot_sum)
        avg_aligned_rot = u @ vh
        self._head_zero_rot = avg_aligned_rot.T

        self._combined_rot_offset = self._head_zero_rot @ self._camera_to_world_rot

        self._is_calibrated = True
        self._calibration_samples = []

    def correct(self, matrix: np.ndarray) -> Pose:
        """
        MediaPipeからの同次変換行列を、キャリブレーション結果を用いて補正します。

        :param matrix: MediaPipeから得られた生の 4x4 同次変換行列
        :return: 補正済みの姿勢データ (x, y, z, yaw, pitch, roll)
        """
        raw_pos = matrix[:3, 3] * self._distance_scale  # 位置ベクトル
        raw_rot = matrix[:3, :3]  # 回転行列

        # 1. アフィン変換による位置補正:
        #    カメラ座標系の位置ベクトルを基底変換し、カメラ位置ベクトルを加算します。
        world_pos = self._camera_to_world_rot @ raw_pos + self._cam_pos

        # 2. 回転補正:
        #    カメラ座標系の回転行列を基底変換し、初期姿勢の逆行列を掛けてゼロ点合わせを行います。
        world_rot = self._combined_rot_offset @ raw_rot

        # 3. オイラー角へ変換
        yaw, pitch, roll = rotation_matrix_to_euler_angles(world_rot)

        # 4. opentrackの座標系定義に合わせて出力
        #    X: 左が正
        #    Y: 上が正
        #    Z: 後ろ（ユーザー側）が正
        #    Yaw: 右向きが正
        #    Pitch: 上向き正
        #    Roll: 反時計回りが正
        return float(world_pos[X]), float(world_pos[Y]), -float(world_pos[Z]), -yaw, -pitch, -roll
