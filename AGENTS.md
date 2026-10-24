# CamHeadTracker

Web カメラで頭部の動きを 6DoF（X、Y、Z、ヨー、ピッチ、ロール）でトラッキングし、UDP で送信する Windows 向けアプリ。

- Python 3.13、uv、Tkinter、NumPy、Pillow、pytest、Ruff、PyInstaller
- 頭の姿勢推定: MediaPipe Face Landmarker
- カメラ入力: 同梱の `ffmpeg.exe`（[Dockerfile](ffmpeg-builder/Dockerfile) による独自ビルド版）
- CI: GitHub Actions

## コマンド

`task setup`（依存関係のインストール）、`task run`、`task lint`、`task format`、`task test`、`task build`（クリーンビルド。出力先は `dist/CamHeadTracker/`）

## 規約

- UI とログのメッセージは英語で書く。
- 新しい依存関係が必要なときは、理由をユーザーに説明する。
- 依存関係を更新するときは、`pyproject.toml` の下限も `uv.lock` に合わせて引き上げる（`uv lock --upgrade` は `uv.lock` しか更新しない）。
- 旧バージョンの `config.ini` との互換性は考慮しない（移行処理もアプリ上の通知も不要）。
- 破壊的変更は事前に警告する。仕様が曖昧なときは、推測で変更せずユーザーに確認する。

## テスト

- 新機能や重要なロジックの変更には、`tests/` のテストを追加・更新する。
- AAA パターンで書き、準備・実行・検証をコメントで明示する。データだけが違う場合は `@pytest.mark.parametrize`、共通の準備は `@pytest.fixture` を使う。
- 浮動小数点は `pytest.approx` か `np.testing.assert_allclose` で、適切な許容誤差を設定して比較する。
- 数学的なロジックは、エッジケースやコーナーケースも網羅する。

## Git

- コミット前に `task lint format test build` を成功させる。
- コミットメッセージは Conventional Commits（`<type>: <簡潔な日本語>`）で書く。type は `build`（uv、PyInstaller、FFmpeg のビルドなど）、`ci`、`docs`（AI エージェントの設定を含む）、`feat`、`fix`、`perf`、`refactor`、`test` のいずれか。
- ユーザーの明示的な指示がない限り、`git commit` や `git push` の前に許可を得る。
- PR は GitHub 上で Squash and merge でマージする。

## 言語プロトコル

- Think and reason in English.
- Write files under `.claude/` in English.
- ソースコメント、ドキュメントは日本語で記述する。
- ユーザーには日本語で応答・報告する。
