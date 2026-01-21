import math

import numpy as np
import pytest

from cam_head_tracker.tracker import (
    PoseCorrector,
    euler_angles_to_rotation_matrix,
    rotation_matrix_to_euler_angles,
)

# -----------------------------------------------------------------------------
# 1. 数学関数（回転行列・オイラー角変換）のテスト
# -----------------------------------------------------------------------------

TEST_ANGLES = [
    (0.0, 0.0, 0.0),  # Identity
    (45.0, 0.0, 0.0),  # Yaw only
    (0.0, -30.0, 0.0),  # Pitch only
    (0.0, 0.0, 10.0),  # Roll only
    (10.0, 20.0, 30.0),  # Combined
    (-15.0, 45.0, -5.0),  # Mixed signs
    (120.0, -20.0, 180.0),  # Large angles (Aliasing check)
]


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_matrix_orthogonality(yaw, pitch, roll):
    """
    生成される回転行列が「直交行列」の性質を満たしているか検証する。
    """
    matrix = euler_angles_to_rotation_matrix(yaw, pitch, roll)

    # 直交性チェック: R * R_transpose == Identity Matrix
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)

    # 行列式チェック: det(R) == 1.0
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_rotation_round_trip_matrix_stability(yaw, pitch, roll):
    """
    Euler -> Rotation Matrix -> Euler -> Rotation Matrix の変換を行い、
    「回転行列」が維持されているか検証する。
    """
    # 1. Euler -> Matrix
    matrix_original = euler_angles_to_rotation_matrix(yaw, pitch, roll)

    # 2. Matrix -> Euler
    res_yaw, res_pitch, res_roll = rotation_matrix_to_euler_angles(matrix_original)

    # 3. Euler -> Matrix (復元)
    matrix_restored = euler_angles_to_rotation_matrix(res_yaw, res_pitch, res_roll)

    # 4. 行列の各要素が一致することを確認
    np.testing.assert_allclose(matrix_original, matrix_restored, atol=1e-9)


@pytest.mark.parametrize(
    "mat, angles",
    [
        # 回転なし
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0)),
        # Yaw 90度
        ([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], (90.0, 0.0, 0.0)),
        # Pitch 90度
        ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0.0, 90.0, 0.0)),
        # Roll 90度
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0.0, 0.0, 90.0)),
        # Yaw -90度
        ([[0, 0, -1], [0, 1, 0], [1, 0, 0]], (-90.0, 0.0, 0.0)),
        # Pitch 180度
        ([[1, 0, 0], [0, -1, 0], [0, 0, -1]], (0.0, 180.0, 0.0)),
        # Yaw 30度、Pitch 45度、Roll 60度
        (
            [
                [0.4330127018922193, -0.75, 0.5],
                [0.75, 0.25, -0.6123724356957945],
                [-0.5, 0.6123724356957945, 0.6123724356957945],
            ],
            (30.0, 45.0, 60.0),
        ),
        # Yaw 45度、Pitch 45度、Roll 90度
        (
            [
                [0.0, -0.8660254037844386, 0.8660254037844386],
                [0.8660254037844386, 0.5, 0.5],
                [-0.8660254037844386, 0.5, 0.5],
            ],
            (45.0, 45.0, 90.0),
        ),
    ],
)
def test_known_rotation_values(mat, angles):
    assert rotation_matrix_to_euler_angles(np.array(mat)) == pytest.approx(angles, abs=1e-9)


# -----------------------------------------------------------------------------
# 2. PoseCorrector（キャリブレーションロジック）のテスト
# -----------------------------------------------------------------------------


@pytest.fixture
def corrector():
    c = PoseCorrector()
    c.set_distance_scale(1.0)
    return c


def create_transform_matrix(x, y, z, yaw, pitch, roll):
    """テスト用の 4x4 同次変換行列を作成するヘルパー関数"""
    mat = np.eye(4)
    mat[:3, :3] = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    mat[:3, 3] = [x, y, z]
    return mat


def test_calibration_perfect_setup(corrector):
    """
    理想的な状態でのキャリブレーションテスト。
    入力値に対して出力値が正確に一致するか厳密に検証する。
    """
    corrector.reset_calibration()

    assert not corrector.is_calibrated()
    assert len(corrector._calibration_samples) == 0

    # Step 1: キャリブレーション (Z軸移動 0 -> -10 -> -20)
    samples = [
        create_transform_matrix(0, 0, 0, 0, 0, 0),
        create_transform_matrix(0, 0, -10, 0, 0, 0),
        create_transform_matrix(0, 0, -20, 0, 0, 0),
    ]

    for mat in samples:
        corrector.add_calibration_sample(mat)

    assert len(corrector._calibration_samples) == 3

    corrector.calibrate()

    assert corrector.is_calibrated()
    assert len(corrector._calibration_samples) == 0

    # Step 2: 補正の検証
    # 入力: MediaPipe座標系で X=5.0 (左), 回転なし
    input_x = 5.0
    input_mat = create_transform_matrix(input_x, 0, 0, 0, 0, 0)

    x, y, z, yaw, pitch, roll = corrector.correct(input_mat)

    # 検証:
    # 1. X軸: Opentrackの「左正」仕様により、入力と同じ 5.0 になるべき
    assert x == pytest.approx(input_x, abs=1e-9)

    # 2. 他の成分: 純粋な横移動なので、Y, Z, 回転成分は 0.0 であるべき
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_calibration_tilted_camera(corrector):
    """
    カメラが45度傾いている状態でのキャリブレーションテスト。
    斜め入力が水平に補正され、かつ値の大きさが保たれているか厳密に検証する。
    """
    corrector.reset_calibration()

    # カメラがZ軸周りに時計回りに45度傾いている設定
    tilt_angle = 45.0
    tilt_rot = euler_angles_to_rotation_matrix(0, 0, tilt_angle)

    # 1. キャリブレーション (Z軸移動)
    # 頭の向きは常に45度傾いている状態として記録される
    samples = [
        create_transform_matrix(0, 0, 0, 0, 0, 0),
        create_transform_matrix(0, 0, -10, 0, 0, 0),
        create_transform_matrix(0, 0, -20, 0, 0, 0),
    ]
    for mat in samples:
        mat[:3, :3] = tilt_rot
        corrector.add_calibration_sample(mat)

    corrector.calibrate()

    # 2. 検証: ワールド座標系で「左」に 10.0cm 移動したとする
    world_move_dist = 10.0

    # これを45度傾いたカメラ座標系に変換して入力データを作成する
    # ワールド左 [1, 0, 0] は カメラ座標系では [cos45, sin45, 0] となる
    rad = math.radians(tilt_angle)
    cam_x = world_move_dist * math.cos(rad)
    cam_y = world_move_dist * math.sin(rad)

    # 入力: 位置は斜め、回転は45度傾いたまま
    input_mat = np.eye(4)
    input_mat[:3, 3] = [cam_x, cam_y, 0]
    input_mat[:3, :3] = tilt_rot

    res_x, res_y, res_z, res_yaw, res_pitch, res_roll = corrector.correct(input_mat)

    # 期待値検証 (許容誤差 1e-5):

    # 1. 移動成分: 斜め入力が完全にX軸成分(10.0)に集約されていること
    assert res_x == pytest.approx(world_move_dist, abs=1e-5)  # X should be exactly 10.0
    assert res_y == pytest.approx(0.0, abs=1e-5)  # Y should be 0.0 (removed cross-talk)
    assert res_z == pytest.approx(0.0, abs=1e-5)  # Z should be 0.0

    # 2. 回転成分: 45度の傾きが補正され、0度になっていること
    assert res_roll == pytest.approx(0.0, abs=1e-5)
    assert res_yaw == pytest.approx(0.0, abs=1e-5)
    assert res_pitch == pytest.approx(0.0, abs=1e-5)
