# CamHeadTracker プロジェクト

## プロジェクト概要

CamHeadTrackerは、Windows向けのカメラベースのヘッドトラッキングアプリケーションです。
Webカメラを使用して、ユーザーの頭部の動きを6DoF（X、Y、Z、ヨー、ピッチ、ロール）でトラッキングし、そのデータをUDP経由で`opentrack`に送信します。

## 使用している主要技術

- 言語: Python 3.13
- GUI: `tkinter`
- 姿勢推定: `MediaPipe`のFace landmark detection
- カメラ入力: `ffmpeg.exe`
- 数値演算: `numpy`
- 画像処理: `Pillow`
- ビルド: `PyInstaller`の単一フォルダビルド（`--onedir`）
- リンターおよびフォーマッター: `ruff`
- パッケージ管理: `uv`
- CI: `GitHub Actions`、`Docker`

## 一般的な指示

- ユーザーには日本語で応答せよ。
- 破壊的変更を行う場合は、必ず事前に警告せよ。
- GitHub CLIの`gh`コマンドを使用可能。リモートリポジトリに変更を加える場合は、必ずユーザーに確認を求めよ。
- コミットメッセージやPRの本文、技術解説などが長文になる場合は、`.gemini/tmp`フォルダに一時ファイル（例: `commit_msg.md` や `description.md`）として出力し、そのファイル名と概要をユーザーに提示せよ。
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

1. **作業ブランチの作成:** `build/vX.X.X`の形式のブランチを作成する。
2. **ファイルの更新:** `pyproject.toml`の`version`を変更する。
3. **ロックファイルの更新:** `uv lock`を実行し、`uv.lock`を更新する。
4. **変更内容の確認:** `uv.lock`内の自プロジェクトのバージョンが正しく更新されていることを確認する。
5. **コミット:** `build: bump version to vX.X.X`の形式でコミットする。
6. **タグ付けとプッシュ:** `vX.X.X`の形式のタグを付けて、ブランチとタグをプッシュする。
7. **PRの作成:** プルリクエストを作成する。
8. **リリース:** GitHub Actionsで自動作成されたReleaseドラフトをユーザーが確認し、問題なければリリースする。
9. **後片付け:** 作業ブランチを削除する。

## コーディングスタイル

- 既存のコーディングスタイルに従え。
- ソースコメントは、複雑な仕様やロジックを説明する場合にのみ記述せよ。
- ソースコメントは日本語で、UIやログメッセージは英語で記述せよ。
- 最新の型ヒントを積極的に活用せよ。
- コードの変更後やコミットの直前には、リントおよびフォーマットを実行せよ。
  - 単一ファイル: `uv run ruff check --fix <file>`, `uv run ruff format <file>`
  - 全ファイル: `uv run ruff check --fix .`, `uv run ruff format .`

## テスト

現在、本プロジェクトには自動化されたテストスイートは無い。

もし新機能を追加する際には、`pytest`のようなテストフレームワークを使用して、対応するテストを追加することを検討せよ。

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
- `short summary`には、日本語または英語で簡潔に記述し、末尾にピリオドや句点を付けないこと。
- `optional body`は、冗長さを排除し、`short summary`のみで内容が理解できる場合は省略せよ。

## Issue Conventions

- **タイトル（title）:** 簡単な概要を記述せよ。
- **本文（body）:** 説明、再現手順、期待する動作、原因、対応案などを自由に記述せよ。
- **ラベル（label）:** **build, ci, docs, feat, bug, perf, refactor, test, question**のいずれかを使用せよ。（複数選択可）

## Pull Request Conventions

- **ブランチ（head）:**
  - 通常の場合: `<type>/<summary>`（例: `fix/coordinate-calculation`）
  - Issue 関連の場合: `issues/<issue number>-<type>-<summary>`（例: `issues/3-fix-coordinate-calculation`）
  - 40文字以内、すべて小文字
- **タイトル（title）:** `<type>: <short summary>`の形式で記述せよ。
- **本文（body）:**
  以下の形式で、必要な項目のみ記述せよ。Issueをクローズする場合は、`関連 Issue`に`close #1`の形式で記述せよ。

  ```markdown
  ## 内容

  ### 変更理由

  ### 実装内容

  ### 影響範囲

  ### 検証内容

  ## 関連 Issue
  ```

- **ラベル（label）:** タイトルの`type`と同じ英単語をラベルとして使用せよ。
- **type**: ブランチ名およびタイトルの`type`には、**build, ci, docs, feat, fix, perf, refactor, test**のいずれかを使用せよ。
