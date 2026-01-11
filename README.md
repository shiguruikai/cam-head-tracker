# CamHeadTracker

Webカメラを使用して頭の動き（6DoF: X, Y, Z, Yaw, Pitch, Roll）をトラッキングし、姿勢データを[opentrack](https://github.com/opentrack/opentrack)に連携するためのアプリ。

## 特徴

* **キャリブレーション機能:** 姿勢データのサンプルをもとにカメラの設置位置（高さ、角度）を算出して補正する。
* **ヘッドトラッキング:** 姿勢推定には MediaPipe Face landmark detection を使用し、オフラインで高精度なトラッキングが可能。
* **UDP送信:** [opentrack](https://github.com/opentrack/opentrack)の入力として使用可能。

## 動作環境

* Windows OS
* Webカメラ

## 使用方法

TODO: 後で書く

## 開発環境の構築とビルド方法

パッケージマネージャーの [uv](https://github.com/astral-sh/uv) を使用する。

1. **リポジトリのクローン**

   ```
   git clone [https://github.com/YOUR_USERNAME/cam-head-tracker.git](https://github.com/YOUR_USERNAME/cam-head-tracker.git)
   cd cam-head-tracker
   ```

2. **依存関係の同期**

   ```
   uv sync
   ```

3. **アプリケーションの実行**

   ```
   uv run python src/cam_head_tracker/main.py
   ```

4. **MediaPipeモデルの更新**

   [Face landmark detection guide | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)から最新の`face_landmarker.task`をダウンロードし、`assets/face_landmarker.task`に配置する。

5. **`ffmpeg.exe`の更新**

   dockerを使用できる環境に`ffmpeg-builder/Dockerfile`を配置し、ビルドする。

   ```
   docker build -t ffmpeg-builder -o out .
   ```

   ビルド環境に生成された`out/fmpeg.exe`を`assets/ffmpeg.exe`に配置する。

6. **アプリケーションのビルド**

   ```
   uv run pyinstaller -y --clean build.spec
   ```

   完了すると`dist\CamHeadTracker`フォルダが生成される。
