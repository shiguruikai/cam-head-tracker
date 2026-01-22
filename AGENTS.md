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
- 数値演算: `NumPy`
- 画像処理: `Pillow`
- ビルド: `PyInstaller`の単一フォルダビルド（`--onedir`）
- テストフレームワーク: `pytest`
- リンターおよびフォーマッター: `Ruff`
- CI: `GitHub Actions`、`Docker`

## 一般的な指示

- ユーザーには日本語で応答せよ。
- Gemini CLIの実行環境は、Windowsの`PowerShell 5.1`である。複数のコマンドを`&&`で繋いで実行することはできないため、**必ず1コマンドずつ個別に実行せよ。**
- 破壊的変更を行う場合は、必ず事前に警告せよ。
- リモートリポジトリに変更を加える場合は、必ずユーザーに確認を求めよ。
- GitHubの操作は、`gh`コマンドを使用せよ。
- IssueやPull Requestの本文は、`.gemini/tmp`フォルダに一時ファイル（例: `pr_body.md`）として出力し、そのファイル名と概要をユーザーに提示せよ。
- 仕様が不明瞭な場合や曖昧な点がある場合は、推測に基づいて変更を行う前に、ユーザーに確認または説明を求めよ。

## ビルドと実行

- ソースコードから実行: `uv run python -m cam_head_tracker.main`
- クリーンビルド: `uv run pyinstaller -y --clean build.spec`

## バージョニング

セマンティックバージョニング（SemVer）を使用する。

### バージョンアップ手順

1. **前提条件の確認**
    1. **`master`ブランチの同期**:
        1. `git checkout master`
        2. `git pull origin master`
    2. **現行バージョンの確認**: 現在のバージョン番号を特定せよ。
        1. `git describe --tags --abbrev=0`（現在のコミットから到達可能な最新のタグを取得）
        2. `git tag --sort=-v:refname`（既存のタグ一覧を取得）
        3. `Get-Content pyproject.toml | Select-String "^version"`（`pyproject.toml`の`version`で始まる行を取得）
    3. **次期バージョンの決定**: `git log <current tag>..HEAD --oneline`で取得した変更履歴に基づき、次期バージョンをユーザーに提案せよ。
2. **リリース作業**
    1. **作業ブランチの作成**: `git checkout -b build/vX.X.X`
    2. **pyproject.tomlの更新**: `pyproject.toml`の`version`を変更せよ。
    3. **uv.lockの更新**: `uv lock`
    4. **確認**:
        1. `git add pyproject.toml uv.lock`
        2. `git diff --cached`
    5. **コミット**: `git commit -m "build: bump version to vX.X.X"`
    6. **ブランチのプッシュとPRの作成**:
        1. `git push origin HEAD`
        2. `gh pr create --title "build: bump version to vX.X.X" --body-file .gemini/tmp/pr_body.md --label build`
    7. **PRのマージ（ユーザー作業）**: ユーザーがPRをマージするのを待て。
    8. **`master`ブランチの同期**:
        1. `git checkout master`
        2. `git pull origin master`
    9. **タグのプッシュ**:
        1. `git tag vX.X.X`
        2. `git push origin vX.X.X`
    10. **リリース（ユーザー作業）**: 自動作成されたReleaseドラフトをユーザーが`gh release view vX.X.X --web`で確認してリリースするまで待て。
    11. **`master`ブランチの同期**:
        1. `git checkout master`
        2. `git pull origin master`
    12. **後片付け**: `git branch -d build/vX.X.X`で作業用ブランチを削除せよ。

## コーディングスタイル

- 既存のコーディングスタイルに従え。
- ソースコメントは、複雑な仕様やロジックを説明する場合にのみ記述せよ。
- ソースコメントは日本語で、UIやログメッセージは英語で記述せよ。
- 最新の型ヒントを積極的に活用せよ。
- ソースコードの変更後は、リントおよびフォーマットを実行せよ。
    - 単一ファイル: `uv run ruff check --fix <file>`, `uv run ruff format <file>`
    - 全ファイル: `uv run ruff check --fix .`, `uv run ruff format .`

## テスト

新機能の追加や重要なロジックの変更を行う場合は、対応するテストコードを`tests`ディレクトリに追加・更新せよ。

- **テストの実行方法**: `uv run pytest`
- **テストの書き方**:
    - AAA (Arrange, Act, Assert) パターンに従い、準備・実行・検証を明示せよ。
    - 同じロジックでデータのみが異なる場合は`@pytest.mark.parametrize`を活用せよ。
    - インスタンスの生成など、共通の準備処理は`@pytest.fixture`にまとめよ。
    - 浮動小数点の比較には`pytest.approx`または`np.testing.assert_allclose`を使用し、適切な許容誤差を設定せよ。
    - 数学的ロジックについては、エッジケースやコーナーケースを網羅せよ。

## 依存関係

- 絶対に必要な場合を除き、新しい外部依存関係の導入は避けよ。
- 新しい依存関係が必要な場合は、その理由をユーザーに説明せよ。

## Git Conventions

### Commit Message Format

```
<type>: <short summary>

[optional body]
```

- **type**:
    - **build**: ビルドシステムまたは外部依存関係の変更（uv, PyInstaller, FFmpeg ビルドなど）
    - **ci**: CI 設定の変更（GitHub Actionsなど）
    - **docs**: ドキュメントのみの変更（README.md, .gemini/settings.json など）
    - **feat**: 新機能の追加
    - **fix**: バグ修正
    - **perf**: パフォーマンスを向上させる変更
    - **refactor**: 機能追加もバグ修正も行わないコード変更
    - **test**: テストの追加または既存テストの修正
- **short summary**: 日本語で簡潔な概要を記述せよ。末尾の句点は禁止。
- **optional body**: 冗長さを排除し、絶対に必要な場合にのみ記述せよ。

### Issue Format

- **title**: タイトルだけで内容が推測できるような簡潔かつ具体的な概要
- **body**: 自由形式（概要、再現手順、期待する動作、原因、ログ、一時的な回避策、修正案など）
- **label**: build, ci, docs, feat, bug, perf, refactor, test, question のいずれか（複数選択可）
    - バグ報告の場合、`fix`ではなく`bug`を使用せよ。

### Issue Workflow

Issueの作成手順:

1. **現状の分析**: 既存のIssueを検索し、重複がないかチェックせよ。
    1. **一覧取得**: `gh issue list --state all --limit 100`（直近のIssueを確認）
    2. **詳細検索**: `gh issue list --state all --limit 100 --search <query>`（関連しそうな単語で検索）
    3. **内容の確認**: 重複が疑われる場合、`gh issue view "<issue number>"`でタイトルと本文を取得して精査せよ。
2. **本文とタイトルの作成**: 本文は`.gemini/tmp/issue_body.md`に出力し、ユーザーに確認を求めよ。
3. **Issueの作成**: `gh issue create --title "<title>" --body-file .gemini/tmp/issue_body.md --label <label>`

### Pull Request Format

- **type**: build, ci, docs, feat, fix, perf, refactor, test のいずれか1つ
- **branch name**: `<type>/<summary>`（40文字以内、すべて小文字、例: `fix/coordinate-calculation`）
- **title**: `<type>: <日本語で簡潔かつ具体的な概要>`
- **label**: `<type>`
- **body**:
    ```
    ## 概要
    <!-- PRの背景、目的、概要など -->

    ## 変更内容
    <!-- 変更内容 -->

    ## 影響範囲
    <!-- （省略可）他の機能への影響など -->

    ## 検証内容
    <!-- （省略可）テスト方法、テスト結果 -->

    ## 関連Issue
    <!-- （省略可）関連するIssue ※クローズする場合は close #1 の形式 -->
    ```

### Pull Request Workflow

PRの作成手順:

1. **コードフォーマット**: `uv run ruff check --fix .`, `uv run ruff format .`
2. **テスト**: `uv run pytest`を実行し、結果の要約をユーザーに提示せよ。
3. **コミット履歴の分析**: `git diff <base branch>...HEAD`
4. **本文とタイトルの作成**: 本文は`.gemini/tmp/pr_body.md`に出力し、ユーザーに確認を求めよ。
5. **プッシュとPRの作成**: 許可を得たら、ブランチをプッシュし、PRを作成せよ。
    1. `git push origin HEAD`
    2. `gh pr create --title "<title>" --body-file .gemini/tmp/pr_body.md --label <type>`
