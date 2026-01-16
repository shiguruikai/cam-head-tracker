# CamHeadTracker

Web カメラを使用して頭の動き（6DoF: X, Y, Z, Yaw, Pitch, Roll）をトラッキングし、姿勢データを opentrack に連携するためのアプリ。

<img src="docs/app.jpg" width="600" alt="application screenshot">

## 特徴

- **ヘッドトラッキング:** オフラインで高精度な姿勢推定ができる MediaPipe Face landmark detection を使用。
- **キャリブレーション機能:** 水平移動したときの姿勢データのサンプルをもとに、カメラの設置位置（高さ、角度）を算出して補正する。
- **UDP 送信:** 姿勢データを UDP で送信。[opentrack](https://github.com/opentrack/opentrack) の入力として使用可能。

## 動作環境

- Windows OS
- Web カメラ<br>
  ※代わりに、USB Web カメラとして使用できる機能を備えた Android 14 以降のスマートフォンでも利用可能。<br>
  ※映像に乱れや遅延が発生したときは、USB 3.0 以上の規格に対応した高速なケーブルを使用すること。

## 使用方法

1. [Releases](https://github.com/shiguruikai/cam-head-tracker/releases) から最新の`CamHeadTracker_vX.X.X.zip`をダウンロードし、任意の場所に展開します。
2. Web カメラをモニターの上部または下部に設置します。
3. \[Camera Device\] で使用する Web カメラを選択します。
4. \[Distance Scale\] のスライダーを動かして、推定された X および Z の距離が実際の距離と近くなるように調整します。
5. \[Calibration\] ボタンをクリックしてキャリブレーションを開始します。<br>
   キャリブレーション中は、モニターの正面（垂直の中心線）を向き、頭を水平に保ちながら前方または後方へゆっくり移動します。<br>
   水平移動時のデータが一定数収集されるとキャリブレーションが完了します。<br>
   キャリブレーション後、推定された Y および Pitch の誤差が小さければキャリブレーションは成功です。
6. プレビュー映像が不要な場合、\[Enable Preview\] のチェックを外してください。CPU 負荷が減ります。

#### opentrack との連携方法

1. opentrack を起動し、\[Input\] で UDP over network を選択します。
2. 右隣の設定ボタンをクリックすると、opentrack のポート番号（デフォルト: 4242）が表示されます。
3. CamHeadTracker の \[Port\] に opentrack のポート番号を入力し、\[Apply\] ボタンをクリックします。

## 開発環境の構築とビルド方法

パッケージマネージャーの [uv](https://github.com/astral-sh/uv) を使用します。

1. **リポジトリのクローン**

   ```
   git clone https://github.com/shiguruikai/cam-head-tracker.git
   cd cam-head-tracker
   ```

2. **依存関係の同期**

   ```
   uv sync --frozen
   ```

3. **アプリケーションの実行**

   ```
   uv run python -m cam_head_tracker.main
   ```

4. **アプリケーションのビルド**

   ```
   uv run pyinstaller -y --clean build.spec
   ```

   ビルド成果物は`dist\CamHeadTracker`ディレクトリに出力されます。

### MediaPipe モデルの更新

[Face landmark detection guide | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) から最新の`face_landmarker.task`をダウンロードし、`cam-head-tracker/assets/face_landmarker.task`に配置します。

### `ffmpeg.exe`の更新

本アプリは、Web カメラの情報および映像取得に`ffmpeg.exe`を使用しており、必要最小限の機能のみを有効化したカスタムビルド版を同梱しています。

**自動更新**

GitHub Actions のワークフローで FFmpeg の最新安定版を定期的にチェックします。新しいバージョンがリリースされている場合、自動でビルドしてプルリクエストを作成します。

**手動更新**

1. Docker を実行できる環境に`ffmpeg-builder`ディレクトリを配置します。
2. ビルド対象のバージョン（FFmpeg のリポジトリにおけるタグまたはブランチ名）を`ffmpeg-builder/version`に記載します。
3. スクリプトを実行してビルドします。
   ```sh
   cd ffmpeg-builder
   chmod +x build.sh
   ./build.sh
   ```
4. 生成された`out/ffmpeg.exe`を`cam_head_tracker/assets/ffmpeg.exe`に配置します。

## バグ報告と貢献

個人使用の目的で公開したのですが、興味を持っていただきありがとうございます。<br>
個人開発のため、PR 等は対応にお時間をいただく場合があります。<br>
バグ報告やご質問等あれば Issues でご連絡ください。

## ライセンス

[MIT License](LICENSE.txt)

本アプリの動作には以下のライブラリが含まれています。

- FFmpeg (Custom build): [LGPL v2.1](ffmpeg-builder/LICENSE_FFMPEG.txt)
- Third-Party Notices: [NOTICE.txt](NOTICE.txt)
