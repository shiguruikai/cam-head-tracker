# CamHeadTracker

Webカメラを使用して頭の動き（6DoF: X, Y, Z, Yaw, Pitch, Roll）をトラッキングし、姿勢データをopentrackに連携するためのアプリ。

<img src="docs/app.jpg" width="600" alt="application screenshot">

## 特徴

* **ヘッドトラッキング:** オフラインで高精度な姿勢推定ができる MediaPipe Face landmark detection を使用。
* **キャリブレーション機能:** 水平移動したときの姿勢データのサンプルをもとに、カメラの設置位置（高さ、角度）を算出して補正する。
* **UDP送信:** 姿勢データをUDPで送信。[opentrack](https://github.com/opentrack/opentrack) の入力として使用可能。

## 動作環境

* Windows OS
* Webカメラ
    * 代わりに、USB Webカメラとして使用できる機能を備えたAndroid 14以降のスマートフォンでも利用可能。<br>
    * 映像に乱れや遅延が発生したときは、USB 3.0以上の規格に対応した高速なケーブルを使用すること。

## 使用方法

1. [Releases](https://github.com/shiguruikai/cam-head-tracker/releases) から最新の`CamHeadTracker_vX.X.X.zip`をダウンロードし、任意の場所に展開します。
2. Webカメラをモニターの上部または下部に設置します。
3. \[Camera Device\] で使用するWebカメラを選択します。
4. \[Distance Scale\] のスライダーを動かして、推定された X および Z の距離が実際の距離と近くなるように調整します。
5. \[Calibration\] ボタンをクリックしてキャリブレーションを開始します。<br>
   キャリブレーション中は、モニターの正面（垂直の中心線）を向き、頭を水平に保ちながら前方または後方へゆっくり移動します。<br>
   水平移動時のデータが一定数収集されるとキャリブレーションが完了します。<br>
   キャリブレーション後、推定された Y および Pitch の誤差が小さければキャリブレーションは成功です。
6. プレビュー映像が不要な場合、\[Enable Preview\] のチェックを外してください。CPU負荷が減ります。

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
   uv run python src/cam_head_tracker/main.py
   ```

4. **MediaPipe モデルの更新**

   [Face landmark detection guide | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)から最新の`face_landmarker.task`をダウンロードし、`cam-head-tracker/assets/face_landmarker.task`に配置する。

5. **`ffmpeg.exe`の更新**

   docker を使用可能な環境に`ffmpeg-builder/Dockerfile`を配置し、ビルドする。<br>
   ※再配布可能でWebカメラを使用するための最小構成のカスタムビルドです。

   ```
   docker build -t ffmpeg-builder -o out .
   ```

   ビルド環境に生成された`out/fmpeg.exe`を`cam-head-tracker/assets/ffmpeg.exe`に配置する。

6. **アプリケーションのビルド**

   ```
   uv run pyinstaller -y --clean build.spec
   ```

   完了すると`dist\CamHeadTracker`フォルダが生成される。

## バグ報告と貢献

個人使用の目的で公開したのですが、興味を持っていただきありがとうございます。<br>
現在、個人開発・学習中のため、PR等は対応にお時間をいただく場合があります。<br>
バグを見つけた場合は、Issues でお知らせください。忘れた頃には修正されているかもしれません。<br>
その他、ご質問等あれば Issues でご連絡ください。

## ライセンス

[MIT License](LICENSE.txt)

本アプリの動作には以下のライブラリが含まれています。

* FFmpeg (Custom build): [LGPL v2.1](ffmpeg-builder/LICENSE_FFMPEG.txt)
* Third-Party Notices: [NOTICE.txt](NOTICE.txt)
