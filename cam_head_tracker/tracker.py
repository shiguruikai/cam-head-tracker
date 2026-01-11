import math
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarker

X, Y, Z, YAW, PITCH, ROLL = 0, 1, 2, 3, 4, 5

FACE_LANDMARKER_MODEL_PATH = Path(__file__).parent / "assets/face_landmarker.task"


def create_face_landmarker() -> FaceLandmarker:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def mp_matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    r = matrix[:3, :3]
    pitch = -np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = -np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)))
    roll = -np.degrees(np.arctan2(r[1, 0], r[0, 0]))

    x = matrix[0, 3]
    y = matrix[1, 3]
    z = -matrix[2, 3]

    return np.array([x, y, z, yaw, pitch, roll])


class PoseCorrector:
    def __init__(self):
        self._sample_data = []
        self._is_calibrated = False
        self._distance_scale = 1.0
        self._cam_height = 0.0
        self._cam_angle = 0.0
        self._cos_cam_angle = 1.0
        self._sin_cam_angle = 0.0
        self._offset_pitch = 0.0

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def set_calibrated(self, is_calibrated: bool):
        self._is_calibrated = is_calibrated

    def get_distance_scale(self) -> float:
        return self._distance_scale

    def set_distance_scale(self, distance_scale: float):
        self._distance_scale = distance_scale

    def get_cam_angle(self) -> float:
        return self._cam_angle

    def set_cam_angle(self, cam_angle: float):
        self._cam_angle = cam_angle
        cam_angle_rad = np.radians(-self._cam_angle)
        self._cos_cam_angle = math.cos(cam_angle_rad)
        self._sin_cam_angle = math.sin(cam_angle_rad)

    def get_cam_height(self) -> float:
        return self._cam_height

    def set_cam_height(self, cam_height: float):
        self._cam_height = cam_height

    def get_offset_pitch(self) -> float:
        return self._offset_pitch

    def set_offset_pitch(self, offset_pitch: float):
        self._offset_pitch = offset_pitch

    def add_calibration_sample(self, pose: np.ndarray) -> bool:
        """
        前後移動中の6DoFデータ（X, Y, Z, Yaw, Pitch, Roll）を蓄積する。
        """
        pose = np.copy(pose)
        pose[:3] *= self._distance_scale

        # 外れ値（左右に5cm以上またはヨーが10度以上またはロールが10度以上）の場合、追加しない。
        if abs(pose[X]) >= 5 or abs(pose[YAW]) >= 10 or abs(pose[ROLL]) >= 10:
            return False

        # 直近のサンプルデータから0.5cm以上距離が変化していない場合、追加しない。
        if self._sample_data and abs(self._sample_data[-1][Z] - pose[Z]) < 0.5:
            return False

        self._sample_data.append(pose)
        return True

    def get_calibration_sample_len(self) -> int:
        return len(self._sample_data)

    def reset_calibration(self):
        self._sample_data = []
        self._is_calibrated = False
        self._distance_scale = 1.0
        self._cam_height = 0.0
        self._cam_angle = 0.0
        self._cos_cam_angle = 1.0
        self._sin_cam_angle = 0.0
        self._offset_pitch = 0.0

    def calibrate(self):
        samples = np.array(self._sample_data)

        # カメラから見た顔の軌跡を最小二乗法で直線近似し、
        # カメラの傾き（slope）とカメラ座標系における高さ（intercept）を求める。
        tz = samples[:, Z]
        ty = samples[:, Y]
        a = np.vstack([tz, np.ones(len(tz))]).T
        (slope, intercept), _, _, _ = np.linalg.lstsq(a, ty, rcond=None)

        # カメラの高さと角度を算出
        cam_angle_rad = np.arctan(slope)
        self._cam_angle = -np.degrees(cam_angle_rad)
        self._cam_height = -intercept * math.cos(cam_angle_rad)

        self._cos_cam_angle = math.cos(cam_angle_rad)
        self._sin_cam_angle = math.sin(cam_angle_rad)

        # ピッチの平均をオフセットピッチとする。
        self._offset_pitch = -np.mean(samples[:, PITCH])

        self._is_calibrated = True
        self._sample_data = []

    def correct(self, pose: np.ndarray) -> np.ndarray:
        x = pose[X] * self._distance_scale
        y = pose[Y] * self._distance_scale
        z = pose[Z] * self._distance_scale
        yaw = pose[YAW]
        pitch = pose[PITCH]
        roll = pose[ROLL]

        # カメラの角度に合わせて回転補正
        new_y = y * self._cos_cam_angle - z * self._sin_cam_angle
        new_z = y * self._sin_cam_angle + z * self._cos_cam_angle

        # カメラの高さを補正
        new_y += self._cam_height

        # ピッチを補正
        new_pitch = pitch + self._offset_pitch

        return np.array([x, new_y, new_z, yaw, new_pitch, roll])
