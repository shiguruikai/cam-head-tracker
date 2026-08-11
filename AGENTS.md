# CamHeadTracker プロジェクト

## プロジェクト概要

CamHeadTracker は、Windows 向けのカメラベースのヘッドトラッキングアプリケーションです。
Web カメラを使用して、ユーザーの頭部の動きを 6DoF（X、Y、Z、ヨー、ピッチ、ロール）でトラッキングし、そのデータを UDP で送信します。

## 技術スタック

- Python (v3.13), `uv`, Tkinter, NumPy, Pillow, pytest, Ruff, PyInstaller
- 頭の姿勢推定: MediaPipe Face Landmarker
- カメラ入力: `ffmpeg.exe`（[Dockerfile](ffmpeg-builder/Dockerfile) による独自ビルド版）
- CI: GitHub Actions

## コマンド

- **アプリ実行**: `task run`
- **クリーンビルド**: `task build`（出力先: `dist/CamHeadTracker/`）
- **リント**: `task lint`
- **フォーマット**: `task format`
- **テスト**: `task test`

## コーディングスタイル

- 既存のコーディングスタイルに従うこと。
- **ソース中のコメントは日本語**で記述せよ。
- **UIおよびログ出力のメッセージは英語**で記述せよ。

## テストルール

- 新機能の追加や重要なロジックの変更を行う場合は、対応するテストコードを `tests` ディレクトリに追加・更新せよ。
- AAA (Arrange, Act, Assert) パターンに従い、準備・実行・検証を明示せよ。
- 同じロジックでデータのみが異なる場合は `@pytest.mark.parametrize` を活用せよ。
- インスタンスの生成など、共通の準備処理は `@pytest.fixture` にまとめよ。
- 浮動小数点の比較には `pytest.approx` または `np.testing.assert_allclose` を使用し、適切な許容誤差を設定せよ。
- 数学的ロジックについては、エッジケースやコーナーケースを網羅せよ。

## 依存関係

- 新しい依存関係が必要な場合は、その理由をユーザーに説明せよ。

## Gitルール

- **事前検証**: コミット前に必ず `task lint format test build` を実行し成功させること。
- **コミットメッセージ**: `Conventional Commits` 形式（`<type>: <description>`）に準拠し、`<description>` は簡潔な日本語で記述すること。
  - **type**:
    - **build**: ビルドシステムまたは外部依存関係の変更（uv, PyInstaller, FFmpeg ビルドなど）
    - **ci**: CI 設定の変更（GitHub Actions など）
    - **docs**: ドキュメントやAIエージェント設定の変更
    - **feat**: 新機能の追加
    - **fix**: バグ修正
    - **perf**: パフォーマンスを向上させる変更
    - **refactor**: 機能追加もバグ修正も行わないコード変更
    - **test**: テストの追加または既存テストの修正
- **ユーザー確認**: ユーザーの明示的な指示がない限り、自動で `git commit` や `git push` を行わず事前に許可を得ること。

## 注意事項

- 破壊的変更を行う場合は、必ず事前に警告せよ。
- 仕様が不明瞭な場合や曖昧な点がある場合は、推測に基づいて変更を行う前に、ユーザーに確認または説明を求めよ。
