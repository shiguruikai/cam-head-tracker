# CamHeadTracker プロジェクト

## プロジェクト概要

CamHeadTrackerは、Windows向けのカメラベースのヘッドトラッキングアプリケーションです。
Webカメラを使用して、ユーザーの頭部の動きを6DoF（X、Y、Z、ヨー、ピッチ、ロール）でトラッキングし、そのデータをUDP経由で`opentrack`に送信します。

## 使用している主要技術

- 言語: Python 3.13
- パッケージ管理: `uv`
- GUI: `tkinter`
- 頭の姿勢推定: `MediaPipe Face Landmarker`
- カメラ入力: `ffmpeg.exe`
- 数値演算: `NumPy`
- 画像処理: `Pillow`
- ビルド: `PyInstaller`の単一フォルダビルド（`--onedir`）
- テストフレームワーク: `pytest`
- リンターおよびフォーマッター: `Ruff`
- CI: `GitHub Actions`, `Dockerfile`

## 一般的な指示

- **ユーザーには日本語で応答せよ**。
- 実行環境は、Windowsの`PowerShell 5.1`である。複数のコマンドを`&&`で繋いで実行することはできないため、**必ず1コマンドずつ個別に実行せよ。**
- 破壊的変更を行う場合は、必ず事前に警告せよ。
- リモートリポジトリに変更を加える場合は、必ずユーザーに確認を求めよ。
- GitHubの操作は、`gh`コマンドを使用せよ。
- 長文の技術解説などを提示する場合は、`.tmp`フォルダに一時ファイル（例: `description.md`）として出力し、そのファイル名と概要をユーザーに提示せよ。
- 仕様が不明瞭な場合や曖昧な点がある場合は、推測に基づいて変更を行う前に、ユーザーに確認または説明を求めよ。

## 実行とビルド

- ソースコードから実行: `uv run python -m cam_head_tracker.main`
- クリーンビルド: `uv run pyinstaller -y --clean build.spec`

## コーディングスタイル

- 既存のコーディングスタイルに従うこと。
- **ソース中のコメントは日本語**で記述せよ。
- **UIやログ出力のメッセージは英語**で記述せよ。
- Pythonコードは、`ruff`でリントおよびフォーマットせよ。
    - 単一ファイル: `uv run ruff check --fix <file>`, `uv run ruff format <file>`
    - 全ファイル: `uv run ruff check --fix .`, `uv run ruff format .`

## テスト

新機能の追加や重要なロジックの変更を行う場合は、対応するテストコードを`tests`ディレクトリに追加・更新せよ。

- **実行方法**: `uv run pytest`
- **書き方**:
    - AAA (Arrange, Act, Assert) パターンに従い、準備・実行・検証を明示せよ。
    - 同じロジックでデータのみが異なる場合は`@pytest.mark.parametrize`を活用せよ。
    - インスタンスの生成など、共通の準備処理は`@pytest.fixture`にまとめよ。
    - 浮動小数点の比較には`pytest.approx`または`np.testing.assert_allclose`を使用し、適切な許容誤差を設定せよ。
    - 数学的ロジックについては、エッジケースやコーナーケースを網羅せよ。

## 依存関係

- 絶対に必要な場合を除き、新しい外部依存関係の導入は避けよ。
- 新しい依存関係が必要な場合は、その理由をユーザーに説明せよ。

## コミットメッセージ規約

```
<type>: <short summary>

[optional body]
```

- **type**:
    - **build**: ビルドシステムまたは外部依存関係の変更（uv, PyInstaller, FFmpeg ビルドなど）
    - **ci**: CI 設定の変更（GitHub Actionsなど）
    - **docs**: ドキュメントやAIエージェント設定の変更
    - **feat**: 新機能の追加
    - **fix**: バグ修正
    - **perf**: パフォーマンスを向上させる変更
    - **refactor**: 機能追加もバグ修正も行わないコード変更
    - **test**: テストの追加または既存テストの修正
- **short summary**: 日本語で簡潔な概要を記述せよ。末尾の句点は禁止。
- **optional body**: 冗長さを排除し、絶対に必要な場合にのみ記述せよ。
