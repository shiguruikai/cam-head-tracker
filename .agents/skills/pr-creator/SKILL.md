---
name: pr-creator
description:
    Pull Request (PR) の作成を依頼された際に使用するスキルです。
    リポジトリのテンプレート基準に従って PR を作成します。
---

# Pull Request Creator

## 進め方

1. **ブランチ管理**
    - **未コミットの確認**: `git status -s` で未コミットがある場合、どうするべきかユーザーに確認する。
    - **ブランチの切り替え**: `git branch --show-current` で現在のブランチが `master` の場合、`git switch -c <tmp_branch_name>` で一時的なブランチを作成する。
    - **最新情報の取得**: `git fetch origin`
2. **変更内容の分析**:
    - **履歴**: `git log origin/master..HEAD`
    - **差分**: `git diff origin/master...HEAD`
3. **事前検証**: ソースコード（`*.py`, `pyproject.toml`, `uv.lock`, `build.spec` など）に変更がある場合は `AGENTS.md` に従い事前検証を実行すること。失敗時は即座に停止し、ユーザーに報告する。
4. **type の確定**: 変更内容に基づき、`build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test` からいずれか1つ選択する。
5. **（Optional）ブランチ名の変更**: `<type>/<short_summary>` にリネーム（`git branch -m <branch_name>`）する。
6. **本文とタイトルの下書き**:
    - **本文**: `.github/pull_request_template.md` のテンプレートを使用して `.tmp/pr_body.md` に書き込む。コメント（`<!-- -->`）や未入力の見出しは削除すること。
    - **タイトル**: `<type>: <日本語で簡潔かつ具体的な概要>`
7. **プッシュと PR 作成**: ユーザーの承認を得た後、PR を作成する。
    - **プッシュ**: `git push -u origin HEAD`
    - **作成**: `gh pr create --title "<タイトル>" --body-file .tmp/pr_body.md --label <type>`

## 原則

- **テンプレート遵守**: PR のテンプレートを無視しないこと。
- **正確性**: 完了していないタスクのチェックボックスをオンにしないこと。
- **明確性**: PR のタイトルと本文は、明確かつ具体的で、他人がコンテキストを理解するのに十分な情報を提供すること。
- **規約準拠**: `AGENTS.md` の記述ルールに準拠すること。
