import math

import numpy as np
import pytest

from cam_head_tracker.tracker import (
    PITCH,
    ROLL,
    YAW,
    PoseCorrector,
    X,
    Y,
    Z,
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
    (0.0, 89.9, 0.0),  # Near Gimbal Lock (Pitch +90)
    (0.0, -89.9, 0.0),  # Near Gimbal Lock (Pitch -90)
]


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_matrix_orthogonality(yaw, pitch, roll):
    """生成される回転行列が「直交行列」の性質を満たしているか検証する。"""
    matrix = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    # 直交性: R * R.T == I
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    # 行列式: det(R) == 1.0
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_rotation_round_trip_matrix_stability(yaw, pitch, roll):
    """Euler -> Matrix -> Euler -> Matrix の往復変換で行列が維持されるか検証する。"""
    matrix_original = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    res_yaw, res_pitch, res_roll = rotation_matrix_to_euler_angles(matrix_original)
    matrix_restored = euler_angles_to_rotation_matrix(res_yaw, res_pitch, res_roll)
    np.testing.assert_allclose(matrix_original, matrix_restored, atol=1e-9)


@pytest.mark.parametrize(
    "mat, angles",
    [
        # Identity
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0)),
        # Basic Axes (90 deg)
        ([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], (90.0, 0.0, 0.0)),  # Yaw 90
        ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0.0, 90.0, 0.0)),  # Pitch 90
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0.0, 0.0, 90.0)),  # Roll 90
        # Complex cases
        (
            [
                [0.4330127018922193, -0.75, 0.5],
                [0.75, 0.25, -0.6123724356957945],
                [-0.5, 0.6123724356957945, 0.6123724356957945],
            ],
            (30.0, 45.0, 60.0),
        ),
    ],
)
def test_known_rotation_values(mat, angles):
    """既知の回転行列とオイラー角の対応関係を検証する。"""
    assert rotation_matrix_to_euler_angles(np.array(mat)) == pytest.approx(angles, abs=1e-9)


# -----------------------------------------------------------------------------
# 2. PoseCorrector（キャリブレーションロジック）のテスト
# -----------------------------------------------------------------------------


@pytest.fixture
def corrector():
    c = PoseCorrector()
    c.set_distance_scale(1.0)
    return c


def create_transform_matrix(x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
    """テスト用の 4x4 同次変換行列を作成するヘルパー関数"""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    mat[:3, 3] = [x, y, z]
    return mat


def perform_standard_calibration(corrector):
    """標準的なキャリブレーション（正面を向いてZ軸移動）を実行するヘルパー"""
    corrector.reset_calibration()
    samples = [
        create_transform_matrix(0, 0, 0),
        create_transform_matrix(0, 0, -10),
        create_transform_matrix(0, 0, -20),
    ]
    for mat in samples:
        corrector.add_calibration_sample(mat)
    corrector.calibrate()


def test_calibration_sample_filtering(corrector):
    """
    Z軸方向の移動が小さい場合、サンプルとして追加されないことを検証する。
    """
    mat1 = create_transform_matrix(0, 0, 0)
    mat2 = create_transform_matrix(0, 0, 0.499)  # 0.499cm diff (Threshold is 0.5)
    mat3 = create_transform_matrix(0, 0, -0.499)  # 0.499cm diff (Threshold is 0.5)
    mat4 = create_transform_matrix(0, 0, -1.0)  # 1.0cm diff

    assert corrector.add_calibration_sample(mat1) is True
    assert corrector.add_calibration_sample(mat2) is False  # 追加されない
    assert corrector.add_calibration_sample(mat3) is False  # 追加されない
    assert corrector.add_calibration_sample(mat4) is True

    assert corrector.get_calibration_sample_len() == 2


def test_calibration_perfect_setup(corrector):
    """理想的な状態でのキャリブレーション検証。"""
    perform_standard_calibration(corrector)
    assert corrector.is_calibrated()

    # 入力: 左に5cm移動
    input_x = 5.0
    input_mat = create_transform_matrix(input_x, 0, 0)

    pose = corrector.correct(input_mat)

    # 検証: X軸のみ変化し、他は0
    assert pose[X] == pytest.approx(input_x, abs=1e-9)
    assert pose[Y] == pytest.approx(0.0, abs=1e-9)
    assert pose[Z] == pytest.approx(0.0, abs=1e-9)
    assert pose[YAW] == pytest.approx(0.0, abs=1e-9)
    assert pose[PITCH] == pytest.approx(0.0, abs=1e-9)
    assert pose[ROLL] == pytest.approx(0.0, abs=1e-9)


def test_calibration_tilted_camera(corrector):
    """カメラが45度傾いている状態での補正検証。"""
    corrector.reset_calibration()

    assert not corrector.is_calibrated()
    assert corrector.get_calibration_sample_len() == 0

    # カメラの傾き（Z軸周り45度）
    tilt_angle = 45.0
    tilt_rot = euler_angles_to_rotation_matrix(0, 0, tilt_angle)

    # キャリブレーション (傾いたまま前後移動)
    samples = [
        create_transform_matrix(0, 0, 0),
        create_transform_matrix(0, 0, -10),
        create_transform_matrix(0, 0, -20),
    ]
    for mat in samples:
        # 回転成分を上書き
        mat[:3, :3] = tilt_rot
        corrector.add_calibration_sample(mat)

    assert not corrector.is_calibrated()
    assert corrector.get_calibration_sample_len() == len(samples)

    corrector.calibrate()

    assert corrector.is_calibrated()
    assert corrector.get_calibration_sample_len() == 0

    # 検証: ワールド座標系で「左」に 10cm 移動
    # 入力データ作成: 左移動ベクトル [10, 0, 0] を 45度傾ける -> [7.07, 7.07, 0]
    world_move = 10.0
    rad = math.radians(tilt_angle)
    input_mat = np.eye(4)
    input_mat[:3, 3] = [world_move * math.cos(rad), world_move * math.sin(rad), 0]
    input_mat[:3, :3] = tilt_rot  # カメラの回転はそのまま

    pose = corrector.correct(input_mat)

    # 1. 斜め入力がX軸(10.0)に集約されること
    assert pose[X] == pytest.approx(world_move, abs=1e-6)
    assert pose[Y] == pytest.approx(0.0, abs=1e-6)
    assert pose[Z] == pytest.approx(0.0, abs=1e-6)

    # 2. ロール回転(45度)がキャンセルされ0になること
    assert pose[ROLL] == pytest.approx(0.0, abs=1e-6)


def test_distance_scaling(corrector):
    """距離スケール設定が出力に反映されるか検証する。"""
    perform_standard_calibration(corrector)

    # スケールを2倍に設定
    scale = 2.0
    corrector.set_distance_scale(scale)

    input_x = 5.0
    input_mat = create_transform_matrix(input_x, 0, 0)

    pose = corrector.correct(input_mat)

    # 入力5.0 * スケール2.0 = 10.0 になるはず
    assert pose[X] == pytest.approx(input_x * scale, abs=1e-9)


def test_save_load_state(corrector):
    """
    設定の保存・復元（set_calibrated_data）が正しく機能するか検証する。
    別のインスタンスに状態をコピーし、同じ入力に対して同じ出力を返すか確認する。
    """
    # 1. 通常通りキャリブレーション
    perform_standard_calibration(corrector)

    # 2. 状態を取得（Config保存を模倣）
    saved_cam_pose = corrector.get_cam_pose()
    saved_offset = corrector.get_offset_angles()

    # 3. 新しいインスタンスを作成し、状態を復元（Config読み込みを模倣）
    new_corrector = PoseCorrector()
    new_corrector.set_distance_scale(corrector.get_distance_scale())
    new_corrector.set_calibrated_data(saved_cam_pose, saved_offset)

    assert new_corrector.is_calibrated()

    # 4. 同じ入力に対する出力を比較
    input_mat = create_transform_matrix(5.0, 3.0, -2.0, 10.0, 5.0, 0.0)

    pose_orig = corrector.correct(input_mat)
    pose_new = new_corrector.correct(input_mat)

    np.testing.assert_allclose(pose_orig, pose_new, atol=1e-9)


def test_opentrack_coordinate_conversion(corrector):
    """
    Opentrack用の座標系変換（符号反転など）が正しく行われているか検証する。
    correct() は、Z, Yaw, Pitch, Roll の符号を反転して返す仕様。
    """
    perform_standard_calibration(corrector)

    # MediaPipe座標系での入力
    mp_x = 10.0
    mp_y = 20.0
    mp_z = 30.0
    mp_yaw = 5.0
    mp_pitch = -25.0
    mp_roll = 8.0

    input_mat = create_transform_matrix(mp_x, mp_y, mp_z, mp_yaw, mp_pitch, mp_roll)

    pose = corrector.correct(input_mat)

    assert pose[X] == pytest.approx(mp_x, abs=1e-6)
    assert pose[Y] == pytest.approx(mp_y, abs=1e-6)
    assert pose[Z] == pytest.approx(-mp_z, abs=1e-6)
    assert pose[YAW] == pytest.approx(-mp_yaw, abs=1e-6)
    assert pose[PITCH] == pytest.approx(-mp_pitch, abs=1e-6)
    assert pose[ROLL] == pytest.approx(-mp_roll, abs=1e-6)
