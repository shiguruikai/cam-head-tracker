---
name: release-creator
description:
    次期バージョンのリリースを依頼された際に使用するスキルです。
    リポジトリの基準に従って、バージョンアップを行い、新しいリリースが作成されるようにします。
---

# Release Creator

次期バージョンのリリース作業をサポートします。

- セマンティックバージョニング（SemVer）を使用します。
- タグのプッシュにより、GitHub Actionsのワークフローがリリースのドラフトを作成します。

## 進め方

### 1. 前提条件の確認

1. **`master`ブランチの同期**:
    - `git checkout master`
    - `git pull origin master`
2. **現行バージョンの確認**:
    - `git describe --tags --abbrev=0`（現在のコミットから到達可能な最新のタグを取得）
    - `git tag --sort=-v:refname`（既存のタグ一覧を取得）
    - `Get-Content pyproject.toml | Select-String "^version"`（`pyproject.toml`の`version`で始まる行を取得）
3. **次期バージョンの決定**: `git log <current_tag>..HEAD --oneline`で取得した変更履歴に基づき、次期バージョンをユーザーに提案する。

### 2. リリース作業

1. **作業ブランチの作成**: `git checkout -b build/vX.X.X`
2. **pyproject.tomlの更新**: `pyproject.toml`の`version`を次期バージョンに変更する。
3. **uv.lockの更新**: `uv lock`
4. **更新内容の確認**:
    - `git add pyproject.toml uv.lock`
    - `git diff --staged`
5. **コミット**: `git commit -m "build: bump version to vX.X.X"`
6. **ブランチのプッシュとPRの作成**:
    - `git push origin HEAD`
    - `gh pr create --title "build: bump version to vX.X.X" --body-file .gemini/tmp/pr_body.md --label build --label skip-changelog`
7. **PRのマージ（ユーザー作業）**: ユーザーがPRをマージするまで待つ。
8. **`master`ブランチの同期**:
    - `git checkout master`
    - `git pull origin master`
9. **タグのプッシュ**:
    - `git tag vX.X.X`
    - `git push origin vX.X.X`
10. **リリース（ユーザー作業）**: GitHub Actionsのワークフローで自動作成されたReleaseドラフトをユーザーがブラウザで確認し、リリースするまで待つ。
    - `gh run view --web`（ワークフローの実行状況をブラウザで確認）
    - `gh release view vX.X.X --web`（Releaseドラフトをブラウザで確認）
11. **`master`ブランチの同期**:
    - `git checkout master`
    - `git pull origin master`
12. **クリーンアップ**: `git branch -d build/vX.X.X`で作業用ブランチを削除する。

## 原則

- **ユーザー作業の待機**: ユーザー作業を無視して次の手順に進まないこと。
