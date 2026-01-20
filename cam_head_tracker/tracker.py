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
Landmark = tuple[float, float]


def rotation_matrix_to_euler_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """回転行列をオイラー角（Yaw, Pitch, Roll）に変換します。"""
    yaw = math.degrees(np.arctan2(matrix[2, 0], math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)))
    pitch = math.degrees(-np.arctan2(matrix[2, 1], matrix[2, 2]))
    roll = math.degrees(-np.arctan2(matrix[1, 0], matrix[0, 0]))
    return yaw, pitch, roll


def euler_angles_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """オイラー角（Yaw, Pitch, Roll）を回転行列に変換します。"""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(-p), -math.sin(-p)],
            [0, math.sin(-p), math.cos(-p)],
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
    # 適用順序: Z -> Y -> X
    return r_z @ r_y @ r_x


@dataclass(slots=True)
class TrackerResult:
    """トラッキング結果を保持するデータクラス"""

    matrix: np.ndarray
    landmarks: list[Landmark]


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
    トラッキングデータをモニター正面基準の座標系に補正するクラス。

    キャリブレーション中の前提条件:
        1. モニターに向かって正対し、頭を垂直（ロール角なし）に保つこと。
        2. モニターの正面方向（Z軸）に沿って真っ直ぐ前または後に移動すること。
    """

    def __init__(self):
        self._is_calibrated = False
        self._calibration_samples: list[np.ndarray] = []
        self._distance_scale = 1.0

        # カメラ座標系 -> ワールド座標系（モニター基準）への回転行列
        self._camera_to_world_rot = np.eye(3)
        # ワールド座標系におけるカメラ位置オフセット（補正ベクトル）
        self._cam_pos = np.zeros(3)
        # 正面を向くための回転補正（初期姿勢のキャンセル用）
        self._head_zero_rot = np.eye(3)

    def reset_calibration(self):
        """キャリブレーション状態をリセットする。"""
        self._is_calibrated = False
        self._calibration_samples = []
        self._camera_to_world_rot = np.eye(3)
        self._cam_pos = np.zeros(3)
        self._head_zero_rot = np.eye(3)

    def is_calibrated(self) -> bool:
        """キャリブレーション済みかどうかを返す。"""
        return self._is_calibrated

    def set_distance_scale(self, v: float):
        """移動距離の倍率を設定する。"""
        self._distance_scale = v

    def get_calibration_sample_len(self) -> int:
        """現在収集済みのキャリブレーションサンプル数を返す。"""
        return len(self._calibration_samples)

    def get_cam_pose(self) -> Pose:
        """推定されたカメラの設置姿勢（位置と角度）を返す。"""
        # カメラの回転は、ワールド変換行列の逆（転置）
        yaw, pitch, roll = rotation_matrix_to_euler_angles(self._camera_to_world_rot.T)
        return self._cam_pos[X], self._cam_pos[Y], self._cam_pos[Z], yaw, pitch, roll

    def add_calibration_sample(self, matrix: np.ndarray) -> bool:
        """
        キャリブレーション用のサンプルデータを追加する。
        データの偏りを防ぐため、前回追加したサンプルから一定距離以上動いた場合にのみ追加する。

        :param matrix: MediaPipeが出力した変換行列
        :return: サンプルが追加された場合はTrue
        """
        mat = matrix.copy()

        # 平行移動成分にスケールを適用
        mat[:3, 3] *= self._distance_scale

        if self._calibration_samples:
            z = mat[2, 3]
            last_z = self._calibration_samples[-1][2, 3]

            # Z軸（奥行き）の変化が小さい場合、サンプルとして採用しない（ノイズ・偏り防止）
            if abs(last_z - z) < 0.5:  # 0.5cm
                return False

        self._calibration_samples.append(mat)
        return True

    def calibrate(self):
        """収集したサンプルデータに基づき、カメラの設置角度と位置を推定します。"""
        if not self._calibration_samples:
            return

        matrices = np.array(self._calibration_samples)
        positions = matrices[:, :3, 3]  # 全サンプルの位置行列 (N, 3)
        rotations = matrices[:, :3, :3]  # 全サンプルの回転行列 (N, 3, 3)

        # ----------------------------------------------------------------------
        # Step 1: 基準となる頭の方向ベクトルを算出
        # ----------------------------------------------------------------------
        # キャリブレーション中は「正対して頭を垂直にしている」と仮定し、
        # 平均的な頭の向きを「世界の上方向」の基準として利用します。
        rot_sum = np.sum(rotations, axis=0)
        u, _, vh = np.linalg.svd(rot_sum)
        avg_head_rot = u @ vh

        # MediaPipeの回転行列の列ベクトルが、それぞれの軸方向を表す
        head_y_vec = avg_head_rot[:, 1]  # 頭の上方向 (Up)
        head_z_vec = avg_head_rot[:, 2]  # 頭の正面方向 (Forward/Back)

        # ----------------------------------------------------------------------
        # Step 2: 移動データからワールド座標系のZ軸（奥行き）を決定
        # ----------------------------------------------------------------------
        # 位置データの主成分分析（SVD）を行い、分散が最大となる軸（＝ユーザーが前後に動いた方向）を抽出します。
        centered_pos = positions - np.mean(positions, axis=0)
        _, _, vh = np.linalg.svd(centered_pos)
        moving_axis = vh[0]

        # 移動軸の符号（前後）を頭の向きと合わせる。
        # これにより、ユーザーがモニターに近づく/遠ざかる方向が正しくZ軸方向として定義されます。
        if moving_axis @ head_z_vec < 0:
            moving_axis *= -1

        # 正規化して、ワールド座標系のZ軸ベクトルを定義します。
        world_z_axis = moving_axis / np.linalg.norm(moving_axis)

        # ----------------------------------------------------------------------
        # Step 3: ワールド座標系のX軸（水平）とY軸（垂直）を決定
        # ----------------------------------------------------------------------
        # 「頭の上方向」と「Z軸（奥行き）」の外積から、それらに直交するX軸（水平）を求めます。
        # これにより、カメラが傾いていても、ユーザーの頭を基準とした水平な軸を定義できます。
        world_x_axis = np.cross(head_y_vec, world_z_axis)
        world_x_axis /= np.linalg.norm(world_x_axis)

        # 決定したZ軸とX軸から、それらに直交するY軸を再計算します。
        # これでX, Y, Zが互いに直交する正規直交基底（ワールド座標系）が完成します。
        world_y_axis = np.cross(world_z_axis, world_x_axis)

        # 安全策: 算出したY軸が頭の下方向を向いてしまった場合、反転させる
        if (world_y_axis @ head_y_vec) < 0:
            world_y_axis *= -1
            world_x_axis *= -1  # 右手系を維持するためにXも反転

        # ----------------------------------------------------------------------
        # Step 4: カメラ座標系からワールド座標系への変換行列を構築
        # ----------------------------------------------------------------------
        # カメラ座標系のベクトルをワールド座標系へ変換する回転行列
        self._camera_to_world_rot = np.array([world_x_axis, world_y_axis, world_z_axis])

        # ----------------------------------------------------------------------
        # Step 5: 位置と回転の原点（オフセット）を決定
        # ----------------------------------------------------------------------
        # 1. 位置のオフセット:
        #    補正後の座標 P_world = R @ P_raw + cam_pos において、
        #    キャリブレーション時の平均位置で XY が 0 になるように cam_pos を設定します。
        #    これにより、キャリブレーション後のユーザーの位置が (0, 0, Z) となります。
        rotated_positions = positions @ self._camera_to_world_rot.T
        self._cam_pos = -np.mean(rotated_positions, axis=0)
        self._cam_pos[Z] = 0  # Z軸はモニターからの距離を残すため、オフセットは0とします。

        # 2. 回転のオフセット:
        #    キャリブレーション時の平均的な頭の向きを「正面（角度ゼロ）」とするための行列を計算します。
        #    これは、ワールド座標系に変換された後の平均回転行列の逆行列（転置）となります。
        aligned_rotations = self._camera_to_world_rot @ rotations
        rot_sum = np.sum(aligned_rotations, axis=0)
        u, _, vh = np.linalg.svd(rot_sum)
        avg_aligned_rot = u @ vh
        self._head_zero_rot = avg_aligned_rot.T

        self._is_calibrated = True
        self._calibration_samples = []

    def correct(self, matrix: np.ndarray) -> Pose:
        """
        MediaPipeからの変換行列を、キャリブレーション結果を用いて補正します。

        :param matrix: MediaPipeから得られた生の変換行列
        :return: 補正済みの姿勢データ (x, y, z, yaw, pitch, roll)
        """
        raw_pos = matrix[:3, 3] * self._distance_scale
        raw_rot = matrix[:3, :3]

        # 1. 位置補正: カメラ座標 -> ワールド座標 + カメラ位置オフセット
        world_pos = self._camera_to_world_rot @ raw_pos + self._cam_pos

        # 2. 回転補正: カメラ傾き補正 -> ゼロ点合わせ
        world_rot = self._head_zero_rot @ (self._camera_to_world_rot @ raw_rot)

        # 3. オイラー角へ変換
        yaw, pitch, roll = rotation_matrix_to_euler_angles(world_rot)

        # 4. openTrackの座標系に合わせて出力
        #    X: 左が正、Y: 上が正、Z: 後ろ（ユーザー側）が正
        return world_pos[X], world_pos[Y], -world_pos[Z], yaw, pitch, roll
