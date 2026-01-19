import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarker

FACE_LANDMARKER_MODEL_PATH = Path(__file__).parent / "assets/face_landmarker.task"

X, Y, Z, YAW, PITCH, ROLL = 0, 1, 2, 3, 4, 5

Pose = tuple[float, float, float, float, float, float]
Landmark = tuple[float, float]


@dataclass(slots=True)
class TrackerResult:
    pose: Pose
    landmarks: list[Landmark]


class Tracker(Protocol):
    def estimate(self, frame: np.ndarray, timestamp_ms: int) -> TrackerResult | None: ...
    def close(self) -> None: ...


class MediapipeTracker(Tracker):
    PREVIEW_FACE_LANDMARKS_CONNECTIONS = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL + [
        mp.tasks.vision.FaceLandmarksConnections.Connection(a, b)
        for a, b in itertools.pairwise(
            [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]
        )
    ]

    def __init__(self):
        self._landmarker = self._create_face_landmarker()

    def estimate(self, frame: np.ndarray, timestamp_ms: int) -> TrackerResult | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.facial_transformation_matrixes:
            return None

        pose = self._mp_matrix_to_pose(result.facial_transformation_matrixes[0])
        points = []

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            for conn in self.PREVIEW_FACE_LANDMARKS_CONNECTIONS:
                for i in (conn.start, conn.end):
                    landmark = landmarks[i]
                    points.append((landmark.x, landmark.y))

        return TrackerResult(pose, points)

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

    @staticmethod
    def _mp_matrix_to_pose(matrix: np.ndarray) -> Pose:
        r = matrix[:3, :3]
        pitch = -np.degrees(np.arctan2(r[2, 1], r[2, 2]))
        yaw = -np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)))
        roll = -np.degrees(np.arctan2(r[1, 0], r[0, 0]))
        x, y, z = matrix[0, 3], matrix[1, 3], -matrix[2, 3]
        return x, y, z, yaw, pitch, roll


def _build_rotation_matrix(
    yaw_deg, pitch_deg, roll_deg
) -> tuple[np.ndarray, float, float, float, float, float, float, float, float, float]:
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    r_yaw = np.array(
        [
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)],
        ]
    )
    r_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(p), np.sin(p)],
            [0, -np.sin(p), np.cos(p)],
        ]
    )
    r_roll = np.array(
        [
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1],
        ]
    )

    # ロール → ピッチ → ヨー の順で回転
    rot_mat = r_yaw @ r_pitch @ r_roll

    return (
        rot_mat,
        rot_mat[0, 0],
        rot_mat[0, 1],
        rot_mat[0, 2],
        rot_mat[1, 0],
        rot_mat[1, 1],
        rot_mat[1, 2],
        rot_mat[2, 0],
        rot_mat[2, 1],
        rot_mat[2, 2],
    )


class PoseCorrector:
    def __init__(self):
        self._sample_data: list[np.ndarray] = []
        self._is_calibrated = False
        self._distance_scale = 1.0

        # カメラの物理的配置
        self._cam_x = 0.0
        self._cam_y = 0.0
        self._cam_z = 0.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0
        self._cam_roll = 0.0

        # 頭の向きのオフセット（正面を向いた時を0にする）
        self._offset_yaw = 0.0
        self._offset_pitch = 0.0
        self._offset_roll = 0.0

        # 補正用の3D回転行列を各要素に分解してキャッシュ（計算の高速化のため）
        (
            self._rot_mat,
            self._r00,
            self._r01,
            self._r02,
            self._r10,
            self._r11,
            self._r12,
            self._r20,
            self._r21,
            self._r22,
        ) = _build_rotation_matrix(0, 0, 0)

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def get_distance_scale(self) -> float:
        return self._distance_scale

    def set_distance_scale(self, v: float):
        self._distance_scale = v

    def get_cam_yaw(self) -> float:
        return self._cam_yaw

    def get_cam_pitch(self) -> float:
        return self._cam_pitch

    def get_cam_roll(self) -> float:
        return self._cam_roll

    def get_cam_x(self) -> float:
        return self._cam_x

    def get_cam_y(self) -> float:
        return self._cam_y

    def get_cam_z(self) -> float:
        return self._cam_z

    def get_offset_yaw(self) -> float:
        return self._offset_yaw

    def get_offset_pitch(self) -> float:
        return self._offset_pitch

    def get_offset_roll(self) -> float:
        return self._offset_roll

    def set_calibrated_data(self, *, cam_pose: Pose, offset_angle: tuple[float, float, float]):
        self._is_calibrated = True
        self._sample_data = []
        self._cam_x = cam_pose[X]
        self._cam_y = cam_pose[Y]
        self._cam_z = cam_pose[Z]
        self._cam_yaw = cam_pose[YAW]
        self._cam_pitch = cam_pose[PITCH]
        self._cam_roll = cam_pose[ROLL]
        self._offset_yaw, self._offset_pitch, self._offset_roll = offset_angle
        (
            self._rot_mat,
            self._r00,
            self._r01,
            self._r02,
            self._r10,
            self._r11,
            self._r12,
            self._r20,
            self._r21,
            self._r22,
        ) = _build_rotation_matrix(self._cam_yaw, self._cam_pitch, self._cam_roll)

    def add_calibration_sample(self, pose: Pose) -> bool:
        """
        キャリブレーション中のサンプルデータを追加します。
        ただし、直近に追加したデータのZから0.5cm以上離れていなければ追加しない。
        キャリブレーション中は、ユーザーが「モニター正面（原点付近）からZ軸方向（前後）にのみ移動する」という前提。

        :param pose: Pose
        :return: サンプルデータとして追加された場合は True
        """
        pose_arr = np.array(pose)

        # X, Y, Zの距離を補正
        pose_arr[:3] *= self._distance_scale

        # 外れ値の場合、追加しない
        if self._sample_data and abs(self._sample_data[-1][Z] - pose_arr[Z]) < 0.5:
            return False

        self._sample_data.append(pose_arr)
        return True

    def get_calibration_sample_len(self) -> int:
        return len(self._sample_data)

    def reset_calibration(self):
        self.set_calibrated_data(cam_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), offset_angle=(0.0, 0.0, 0.0))
        self._is_calibrated = False

    def calibrate(self):
        if not self._sample_data:
            return

        samples = np.array(self._sample_data)

        # 1. カメラから見て「傾いて見えている頭の角度」の平均を、そのまま「カメラのロール角度」とする。
        self._cam_roll = np.mean(samples[:, ROLL])

        # 2. ロールを打ち消す方向に回転し、直立化
        rotated_xyz = samples[:, :3] @ _build_rotation_matrix(0, 0, self._cam_roll)[0].T
        tx = rotated_xyz[:, X]
        ty = rotated_xyz[:, Y]
        tz = rotated_xyz[:, Z]  # Zは回転しないため samples[:, Z] と同じ

        # 3. 直立化したサンプルデータに対して線形回帰
        # Z（前後）を説明変数として、X（左右）と Y（上下）の傾きを求める。
        a = np.vstack([tz, np.ones(len(tz))]).T
        (slope_x, _), _, _, _ = np.linalg.lstsq(a, tx, rcond=None)
        (slope_y, _), _, _, _ = np.linalg.lstsq(a, ty, rcond=None)
        self._cam_yaw = -np.degrees(np.arctan(slope_x))
        self._cam_pitch = -np.degrees(np.arctan(slope_y))

        # 4. 補正用の3D回転行列
        (
            self._rot_mat,
            self._r00,
            self._r01,
            self._r02,
            self._r10,
            self._r11,
            self._r12,
            self._r20,
            self._r21,
            self._r22,
        ) = _build_rotation_matrix(self._cam_yaw, self._cam_pitch, self._cam_roll)

        # 5. 回転させたサンプルデータの「平均的なズレ」をカメラ位置（オフセット）とする。
        rotated_xyz = samples[:, :3] @ self._rot_mat.T
        self._cam_x = -np.mean(rotated_xyz[:, X])
        self._cam_y = -np.mean(rotated_xyz[:, Y])

        # 6. 頭の向きのオフセットを算出
        self._offset_yaw = -np.mean(samples[:, YAW])
        self._offset_pitch = -np.mean(samples[:, PITCH])
        self._offset_roll = -self._cam_roll

        self._is_calibrated = True
        self._sample_data = []

    def correct(self, pose: Pose) -> Pose:
        # 1. 距離のスケーリング
        x = pose[X] * self._distance_scale
        y = pose[Y] * self._distance_scale
        z = pose[Z] * self._distance_scale

        # 2. カメラの設置角度とカメラの設置位置（オフセット）の補正
        # numpyの「new_xyz = rot_mat @ xyz + cam_xyz」と同等
        new_x = x * self._r00 + y * self._r01 + z * self._r02 + self._cam_x
        new_y = x * self._r10 + y * self._r11 + z * self._r12 + self._cam_y
        new_z = x * self._r20 + y * self._r21 + z * self._r22 + self._cam_z

        # 3. 角度の補正
        return (
            new_x,
            new_y,
            new_z,
            pose[YAW] + self._offset_yaw,
            pose[PITCH] + self._offset_pitch,
            pose[ROLL] + self._offset_roll,
        )
