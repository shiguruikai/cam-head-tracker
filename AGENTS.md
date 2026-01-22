# CamHeadTracker プロジェクト

## プロジェクト概要

CamHeadTrackerは、Windows向けのカメラベースのヘッドトラッキングアプリケーションです。
Webカメラを使用して、ユーザーの頭部の動きを6DoF（X、Y、Z、ヨー、ピッチ、ロール）でトラッキングし、そのデータをUDP経由で`opentrack`に送信します。

## 使用している主要技術

- 言語: Python 3.13
- パッケージ管理: `uv`
- GUI: `tkinter`
- 姿勢推定: `MediaPipe`のFace landmark detection
- カメラ入力: `ffmpeg.exe`
- 数値演算: `numpy`
- 画像処理: `Pillow`
- ビルド: `PyInstaller`の単一フォルダビルド（`--onedir`）
- テストフレームワーク: `pytest`
- リンターおよびフォーマッター: `ruff`
- CI: `GitHub Actions`、`Docker`

## 一般的な指示

- ユーザーには日本語で応答せよ。
- 破壊的変更を行う場合は、必ず事前に警告せよ。
- リモートリポジトリに変更を加える場合は、必ずユーザーに確認を求めよ。
- IssueやPull Requestの本文は、`.gemini/tmp`フォルダに一時ファイル（例: `pr_body.md`）として出力し、そのファイル名と概要をユーザーに提示せよ。
- 仕様が不明瞭な場合や曖昧な点がある場合は、推測に基づいて変更を行う前に、ユーザーに確認または説明を求めよ。

## ビルドと実行

### ソースコードから実行

```
uv run python -m cam_head_tracker.main
```

### クリーンビルド

```
uv run pyinstaller -y --clean build.spec
```

ビルド成果物は`dist\CamHeadTracker`ディレクトリに出力される。

## バージョニング

本プロジェクトは、セマンティックバージョニング（SemVer）を使用する。

### バージョンアップ手順

1. **前提条件の確認**
    1. **`master`ブランチの同期**: ローカルの`master`ブランチをリモートの最新の状態に更新する。
    2. **現行バージョンの確認**: `pyproject.toml`およびリポジトリのタグから現在のバージョン番号を特定する。
    3. **次期バージョンの決定**: 変更履歴（コミットログ）に基づき、次期バージョンをユーザーに提案する。
2. **リリース作業**
    1. **作業ブランチの作成**: `build/vX.X.X`の形式のブランチを作成する。
    2. **ファイルの更新**: `pyproject.toml`の`version`を変更する。
    3. **ロックファイルの更新**: `uv lock`を実行し、`uv.lock`を更新する。
    4. **変更内容の確認**: `uv.lock`内の自プロジェクトのバージョンが正しく更新されていることを確認する。
    5. **コミット**: `build: bump version to vX.X.X`の形式でコミットする。
    6. **ブランチのプッシュとPRの作成**: 作業ブランチをプッシュし、PRを作成する。
    7. **PRのマージ（ユーザー作業）**: ユーザーがレビュー後、PRをマージする。
    8. **`master`ブランチの同期**: ローカルの`master`ブランチをリモートの最新の状態に更新する。
    9. **タグの作成とプッシュ**: `vX.X.X`の形式のタグを作成し、リモートにプッシュする。
    10. **リリース（ユーザー作業）**: タグのプッシュをトリガーにGitHub Actionsで自動作成されたReleaseドラフトをユーザーが確認し、問題なければリリースする。
    11. **後片付け**: 作業ブランチや一時ファイルを削除する。

## コーディングスタイル

- 既存のコーディングスタイルに従え。
- ソースコメントは、複雑な仕様やロジックを説明する場合にのみ記述せよ。
- ソースコメントは日本語で、UIやログメッセージは英語で記述せよ。
- 最新の型ヒントを積極的に活用せよ。
- ソースコードの変更後は、リントおよびフォーマットを実行せよ。
    - 単一ファイル: `uv run ruff check --fix <file>`, `uv run ruff format <file>`
    - 全ファイル: `uv run ruff check --fix .`, `uv run ruff format .`

## テスト

新機能の追加や重要なロジックの変更を行う場合は、対応するテストコードを `tests` ディレクトリに追加・更新せよ。

- **テストの実行方法**: `uv run pytest`
- **テストの書き方**:
    - AAA (Arrange, Act, Assert) パターンに従い、準備・実行・検証を明示せよ。
    - 同じロジックでデータのみが異なる場合は `@pytest.mark.parametrize` を活用せよ。
    - インスタンスの生成など、共通の準備処理は `@pytest.fixture` にまとめよ。
    - 浮動小数点の比較には `pytest.approx` または `np.testing.assert_allclose` を使用し、適切な許容誤差を設定せよ。
    - 数学的ロジックについては、エッジケースやコーナーケースを網羅せよ。

## 依存関係

- 絶対に必要な場合を除き、新しい外部依存関係の導入は避けよ。
- 新しい依存関係が必要な場合は、その理由を明記せよ。

## Commit Message Conventions

- 以下の形式で記述せよ。

    ```
    <type>: <short summary>

    [optional body]
    ```

- `type`には、以下のいずれかを使用せよ。
    - `build`: ビルドシステムまたは外部依存関係の変更（uv, PyInstaller, FFmpeg ビルドなど）
    - `ci`: CI 設定の変更（GitHub Actionsなど）
    - `docs`: ドキュメントのみの変更
    - `feat`: 新機能の追加
    - `fix`: バグ修正
    - `perf`: パフォーマンスを向上させる変更
    - `refactor`: 機能追加もバグ修正も行わないコード変更
    - `test`: テストの追加または既存テストの修正
- `short summary`は、日本語で簡潔に概要を記述し、末尾に句点を付けないこと。
- `optional body`は、冗長さを排除し、どうしても説明が必要な場合にのみ記述せよ。

## Issue Conventions

- **タイトル（title）**: 簡単な概要を記述せよ。
- **本文（body）**: 説明、再現手順、期待する動作、原因、対応案などを自由に記述せよ。
- **ラベル（label）**: **build, ci, docs, feat, bug, perf, refactor, test, question** のいずれかを使用せよ。（複数選択可）

## Pull Request Conventions

- **ブランチ（head）**:
    - 通常の場合: `<type>/<summary>`（例: `fix/coordinate-calculation`）
    - Issue 関連の場合: `issues/<issue number>-<type>-<summary>`（例: `issues/3-fix-coordinate-calculation`）
    - 40文字以内、すべて小文字
- **タイトル（title）**: `<type>: <short summary>`の形式で記述せよ。
- **本文（body）**: 以下の形式で記述せよ。

    ```
    ## 概要
    <!-- PRの背景、目的、概要など -->

    ## 変更内容
    <!-- 変更内容 -->

    ## 影響範囲
    <!-- 他の機能への影響など -->

    ## 検証内容
    <!-- テスト方法、テスト結果 -->

    ## 関連Issue
    <!-- 関連するIssue ※クローズする場合は close #1 の形式 -->
    ```

- **ラベル（label）**: タイトルの`type`と同じ英単語のラベルを追加せよ。
- **type**: ブランチ名およびタイトルの`type`には、 **build, ci, docs, feat, fix, perf, refactor, test** のいずれかを使用せよ。

## Checklist before creating a Pull Request

プルリクエストを作成する前に、必ず以下を完了せよ。

1. リントを実行: `uv run ruff check --fix .`
2. フォーマットを実行: `uv run ruff format .`
3. 全テストがパスすることを確認: `uv run pytest`
