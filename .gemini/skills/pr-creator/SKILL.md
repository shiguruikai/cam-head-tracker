---
name: pr-creator
description:
    Pull Request (PR) の作成を依頼された際に使用するスキルです。
    リポジトリのテンプレート基準に従ってPRを作成します。
---

# Pull Request Creator

## 進め方

1. **ブランチ管理**
    - **未コミットの確認**: `git status -s`で未コミットがある場合、どうするべきかユーザーに確認する。
    - **ブランチの切り替え**: `git branch --show-current`で現在のブランチが`master`の場合、`git switch -c <tmp_branch_name>`で一時的なブランチを作成する。
    - **最新情報の取得**: `git fetch origin`
2. **変更内容の分析**:
    - **履歴**: `git log origin/master..HEAD`
    - **差分**: `git diff origin/master...HEAD`
3. **(Optional) 事前検証**: ソースコード（`*.py`, `pyproject.toml`, `uv.lock`, `build.spec`など）に変更がある場合にのみ実行する。失敗時は即座に停止し、ユーザーに報告する。
    - **リント**: `uv run ruff check --fix .`
    - **フォーマット**: `uv run ruff format .`
    - **テスト**: `uv run pytest -q`
    - **ビルド**: `uv run pyinstaller -y --clean --log-level=WARN build.spec`
4. **typeの確定**: 変更内容に基づき、`build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`からいずれか1つ選択する。
5. **(Optional) ブランチ名の変更**: `<type>/<short_summary>`にリネーム（`git branch -m <branch_name>`）する。
6. **本文とタイトルの下書き**:
    - **本文**: `.github/pull_request_template.md`のテンプレートを使用して`.tmp/pr_body.md`に書き込む。
    - **タイトル**: `<type>: <日本語で簡潔かつ具体的な概要>`
7. **プッシュとPR作成**: ユーザーの承認を得た後、PRを作成する。
    - **プッシュ**: `git push -u origin HEAD`
    - **作成**: `gh pr create --title "<タイトル>" --body-file .tmp/pr_body.md --label <type>`

## 原則

- **テンプレート遵守**: PRのテンプレートを無視しないこと。
- **正確性**: 完了していないタスクのチェックボックスをオンにしないこと。
- **明確性**: PRのタイトルと本文は、明確かつ具体的で、他人がコンテキストを理解するのに十分な情報を提供すること。
