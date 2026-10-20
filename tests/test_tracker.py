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
    (0.0, 90, 0.0),  # Gimbal Lock (Pitch +90)
    (0.0, -90, 0.0),  # Gimbal Lock (Pitch -90)
]


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_matrix_orthogonality(yaw, pitch, roll):
    """生成される回転行列が「直交行列」の性質を満たしているか検証する。"""
    matrix = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    # 直交性: R * R.T == I
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    # 行列式: det(R) == 1.0
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("yaw, pitch, roll", TEST_ANGLES)
def test_rotation_round_trip_matrix_stability(yaw, pitch, roll):
    """Euler -> Matrix -> Euler -> Matrix の往復変換で行列が維持されるか検証する。"""
    matrix_original = euler_angles_to_rotation_matrix(yaw, pitch, roll)
    res_yaw, res_pitch, res_roll = rotation_matrix_to_euler_angles(matrix_original)
    matrix_restored = euler_angles_to_rotation_matrix(res_yaw, res_pitch, res_roll)
    np.testing.assert_allclose(matrix_original, matrix_restored, atol=1e-6)


@pytest.mark.parametrize(
    "mat, angles",
    [
        pytest.param([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0), id="Identity"),
        pytest.param([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], (90.0, 0.0, 0.0), id="Yaw 90 deg"),
        pytest.param([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0.0, 90.0, 0.0), id="Pitch 90 deg"),
        pytest.param([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0.0, 0.0, 90.0), id="Roll 90 deg"),
        pytest.param(
            [
                [0.7391989197401166, -0.5732233047033631, 0.35355339059327373],
                [0.6123724356957946, 0.35355339059327384, -0.7071067811865476],
                [0.28033008588991065, 0.7391989197401166, 0.6123724356957946],
            ],
            (30.0, 45.0, 60.0),
            id="Complex case (Y:30, P:45, R:60)",
        ),
    ],
)
def test_known_rotation_values(mat, angles):
    """既知の回転行列とオイラー角の対応関係を検証する。"""
    assert rotation_matrix_to_euler_angles(np.array(mat)) == pytest.approx(angles, abs=1e-6)


def rot_x(deg):
    """X軸（Pitch）周りの基本回転行列"""
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    """Y軸（Yaw）周りの基本回転行列"""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg):
    """Z軸（Roll）周りの基本回転行列"""
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "yaw, pitch, roll",
    [
        pytest.param(45.0, 0.0, 20.0, id="Yaw + Roll"),
        pytest.param(0.0, 20.0, 20.0, id="Pitch + Roll"),
        pytest.param(30.0, 15.0, 15.0, id="Yaw + Pitch + Roll"),
        pytest.param(-60.0, -10.0, 10.0, id="Large yaw with mixed signs"),
        pytest.param(80.0, 0.0, -5.0, id="Near yaw 90 deg"),
    ],
)
def test_decomposition_order_is_yaw_pitch_roll(yaw, pitch, roll):
    """
    Yaw -> Pitch -> Roll の順に回転した姿勢が、同じ角度に分解されるか検証する。
    分解順序が異なると、Roll を含む組み合わせで他の軸に偽の角度が混入する。
    """
    # 準備: 基本回転を Yaw -> Pitch -> Roll の順に合成
    matrix = rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)

    # 実行
    result = rotation_matrix_to_euler_angles(matrix)

    # 検証
    assert result == pytest.approx((yaw, pitch, roll), abs=1e-6)


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
        assert corrector.add_calibration_sample(mat)
    assert corrector.get_calibration_sample_len() == len(samples)
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
    assert pose[X] == pytest.approx(input_x, abs=1e-6)
    assert pose[Y] == pytest.approx(0.0, abs=1e-6)
    assert pose[Z] == pytest.approx(0.0, abs=1e-6)
    assert pose[YAW] == pytest.approx(0.0, abs=1e-6)
    assert pose[PITCH] == pytest.approx(0.0, abs=1e-6)
    assert pose[ROLL] == pytest.approx(0.0, abs=1e-6)


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


def test_distance_scale_applies_to_depth_only(corrector):
    """距離スケールがカメラ座標系の Z（奥行き）にのみ適用されるか検証する。"""
    # 準備: スケール2倍でキャリブレーション
    scale = 2.0
    corrector.set_distance_scale(scale)
    perform_standard_calibration(corrector)
    input_mat = create_transform_matrix(5.0, 3.0, -50.0)

    # 実行
    pose = corrector.correct(input_mat)

    # 検証: X, Y はそのまま、Z のみ2倍（opentrack 向けに符号反転）
    assert pose[X] == pytest.approx(5.0, abs=1e-6)
    assert pose[Y] == pytest.approx(3.0, abs=1e-6)
    assert pose[Z] == pytest.approx(50.0 * scale, abs=1e-6)


# MediaPipe の推定奥行き / 真の奥行き（例: 垂直画角 43° のカメラを 63° と仮定した場合）
DEPTH_RATIO = 0.65


def mediapipe_estimated_matrix(rotation, true_position):
    """画角の仮定の違いにより、奥行きだけが DEPTH_RATIO 倍に縮んだ MediaPipe の推定結果を模擬する。"""
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = np.asarray(true_position) * [1.0, 1.0, DEPTH_RATIO]
    return mat


@pytest.mark.parametrize("cam_yaw", [20.0, 40.0, -30.0])
def test_calibration_with_depth_compressed_estimate(corrector, cam_yaw):
    """
    奥行きだけが縮んで推定される状況で、斜めに設置したカメラでも正しく補正できるか検証する。
    距離スケールを X, Y, Z 一様にかけると、移動軸が歪んでカメラ角度と軸間の混入が生じる。
    """
    # 準備: 頭をカメラに対して cam_yaw だけ斜めに向けたまま、頭の正面方向へ前後移動してキャリブレーション
    neutral_rot = rot_y(cam_yaw)
    lateral, up, forward = neutral_rot[:, 0], neutral_rot[:, 1], neutral_rot[:, 2]
    base_pos = np.array([-10.0, -5.0, -60.0])
    corrector.set_distance_scale(1.0 / DEPTH_RATIO)
    for d in (-10.0, -5.0, 0.0, 5.0, 10.0):
        assert corrector.add_calibration_sample(mediapipe_estimated_matrix(neutral_rot, base_pos + forward * d))
    corrector.calibrate()
    base_pose = np.array(corrector.correct(mediapipe_estimated_matrix(neutral_rot, base_pos))[:3])

    # 実行: 頭の向きはそのまま、横・上・前へ 10cm ずつ平行移動
    def moved(direction):
        mat = mediapipe_estimated_matrix(neutral_rot, base_pos + direction * 10.0)
        return np.array(corrector.correct(mat)[:3]) - base_pose

    # 検証: 各移動が対応する1軸にだけ現れる（Z は opentrack 向けに符号反転）
    np.testing.assert_allclose(moved(lateral), [10.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(moved(up), [0.0, 10.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(moved(forward), [0.0, 0.0, -10.0], atol=1e-3)

    # 検証: カメラの設置角（頭に対する向き）が正しく推定される
    assert corrector.get_cam_pose()[YAW] == pytest.approx(-cam_yaw, abs=1e-3)


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

    np.testing.assert_allclose(pose_orig, pose_new, atol=1e-6)


def perform_calibration(corrector, neutral_rot, moving_axis, base_pos):
    """頭の姿勢を neutral_rot に保ったまま、moving_axis 方向へ前後移動するキャリブレーションを実行するヘルパー"""
    moving_axis = np.asarray(moving_axis, dtype=float)
    moving_axis /= np.linalg.norm(moving_axis)
    for d in (-10.0, -5.0, 0.0, 5.0, 10.0):
        mat = np.eye(4)
        mat[:3, :3] = neutral_rot
        mat[:3, 3] = np.asarray(base_pos) + moving_axis * d
        assert corrector.add_calibration_sample(mat)
    corrector.calibrate()


# キャリブレーション時の頭部姿勢と移動方向の組み合わせ（カメラの設置状況の違いを表す）
CALIBRATION_SETUPS = [
    pytest.param(rot_y(30.0), rot_y(30.0)[:, 2], id="Camera to the side"),
    pytest.param(rot_x(-20.0) @ rot_y(10.0), (rot_x(-20.0) @ rot_y(10.0))[:, 2], id="Camera above, looking down"),
    pytest.param(rot_y(15.0) @ rot_z(3.0), [0.3, -0.2, 1.0], id="Moving axis differs from head forward"),
]


@pytest.mark.parametrize("neutral_rot, moving_axis", CALIBRATION_SETUPS)
def test_rotation_output_is_relative_to_neutral_head(corrector, neutral_rot, moving_axis):
    """
    回転出力が、キャリブレーション時の頭部姿勢を基準とした（頭部座標系での）相対回転になるか検証する。
    カメラの設置角や移動方向には依存しない。
    """
    # 準備
    perform_calibration(corrector, neutral_rot, moving_axis, [-10.0, 5.0, -55.0])
    yaw, pitch, roll = 25.0, -10.0, 8.0
    input_mat = np.eye(4)
    input_mat[:3, :3] = neutral_rot @ rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)
    input_mat[:3, 3] = [-10.0, 5.0, -55.0]

    # 実行
    pose = corrector.correct(input_mat)

    # 検証: opentrack 向けに符号を反転した相対回転が出力される
    assert pose[YAW] == pytest.approx(-yaw, abs=1e-3)
    assert pose[PITCH] == pytest.approx(-pitch, abs=1e-3)
    assert pose[ROLL] == pytest.approx(-roll, abs=1e-3)


@pytest.mark.parametrize("neutral_rot, moving_axis", CALIBRATION_SETUPS)
def test_save_load_state_with_tilted_setup(corrector, neutral_rot, moving_axis):
    """カメラが斜めに設置されている場合も、保存・復元したキャリブレーションで同じ出力になるか検証する。"""
    # 準備: キャリブレーションし、状態を別のインスタンスに復元（Config の保存・読み込みを模倣）
    corrector.set_distance_scale(1.5)
    perform_calibration(corrector, neutral_rot, moving_axis, [-10.0, 5.0, -55.0])
    restored = PoseCorrector()
    restored.set_distance_scale(corrector.get_distance_scale())
    restored.set_calibrated_data(corrector.get_cam_pose(), corrector.get_offset_angles())
    input_mat = np.eye(4)
    input_mat[:3, :3] = neutral_rot @ rot_y(-20.0) @ rot_x(12.0) @ rot_z(-6.0)
    input_mat[:3, 3] = [3.0, -4.0, -48.0]

    # 実行
    pose_orig = corrector.correct(input_mat)
    pose_restored = restored.correct(input_mat)

    # 検証: オイラー角を経由して復元するため、float32 の丸め誤差を考慮した許容誤差とする
    np.testing.assert_allclose(pose_orig, pose_restored, atol=1e-4)


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
    # 角度は float32 の行列演算から算出されるため、丸め誤差を考慮した許容誤差とする
    assert pose[YAW] == pytest.approx(-mp_yaw, abs=1e-4)
    assert pose[PITCH] == pytest.approx(-mp_pitch, abs=1e-4)
    assert pose[ROLL] == pytest.approx(-mp_roll, abs=1e-4)


@pytest.mark.parametrize("cam_yaw", [0.0, 30.0, -30.0])
def test_correct_no_pitch_crosstalk_on_yaw_and_roll(corrector, cam_yaw):
    """
    横を向いて首を傾けた（Yaw + Roll）ときに、偽の Pitch が出力されないことを検証する。
    カメラを横に設置した場合（キャリブレーション時の顔がカメラに対して斜め）も含む。
    """
    # 準備: 頭をカメラに対して cam_yaw だけ斜めに向けたまま、頭の正面方向へ前後移動してキャリブレーション
    neutral_rot = rot_y(cam_yaw)
    forward = neutral_rot[:, 2]
    base_pos = np.array([-25.0, 0.0, -45.0])
    for i in range(3):
        mat = np.eye(4)
        mat[:3, :3] = neutral_rot
        mat[:3, 3] = base_pos + forward * 10.0 * i
        assert corrector.add_calibration_sample(mat)
    corrector.calibrate()

    # 準備: 中立姿勢から頭のローカル軸で Yaw 45° -> Roll 20° 回転
    input_mat = np.eye(4)
    input_mat[:3, :3] = neutral_rot @ rot_y(45.0) @ rot_z(20.0)
    input_mat[:3, 3] = base_pos

    # 実行
    pose = corrector.correct(input_mat)

    # 検証: opentrack 向けに符号を反転した Yaw / Roll のみが出力される
    assert pose[YAW] == pytest.approx(-45.0, abs=1e-4)
    assert pose[PITCH] == pytest.approx(0.0, abs=1e-4)
    assert pose[ROLL] == pytest.approx(-20.0, abs=1e-4)
