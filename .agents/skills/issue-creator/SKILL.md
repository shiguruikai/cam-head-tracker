---
name: issue-creator
description:
    Issue の作成を依頼された際に使用するスキルです。
    リポジトリの基準に従って Issue が作成されるようにします。
---

# Issue Creator

## 進め方

1. **問題の理解**: どんな Issue を作成するのか、不明点がなくなるまでユーザーと相談する。
2. **重複回避**: 重複がないか確認する。
    - **最近の Issue 一覧**: `gh issue list --state all --limit 100`
    - **関連 Issue の検索**: `gh issue list --state all --limit 100 --search "<関連するキーワード>"`
    - **内容確認**: 重複の可能性のある Issue が見つかった場合、`gh issue view "<issue_number>"` で内容を精査し、重複しているか確認する。
3. **本文とタイトルの下書き**:
    - **本文**: 自由形式（`概要`、`再現手順`、`期待する動作`、`原因`、`ログ`、`一時的な回避策`、`修正案`などの見出し）で `.tmp/issue_body.md` に書き込む。
    - **タイトル**: タイトルだけで内容を推測できるような、簡潔かつ具体的な概要
4. **ラベルの確定**: Issue の内容に基づき、`build`, `ci`, `docs`, `feat`, `bug`, `perf`, `refactor`, `test`, `question` からいずれか1つ選択する。
5. **Issue の作成**: ユーザーの承認を得た後、Issue を作成する。
    - **作成**: `gh issue create --title "<タイトル>" --body-file .tmp/issue_body.md --label <label>`

## 原則

- **明確性**: Issue のタイトルと本文は明確かつ具体的で、他人がコンテキストを理解するのに十分な情報を提供すること。
- **規約準拠**: `AGENTS.md` の記述ルールに準拠すること。
