---
name: issue-creator
description:
    Issue の作成を依頼された際に使用するスキルです。
    リポジトリの基準に従って Issue が作成されるようにします。
---

# Issue Creator

リポジトリの基準に準拠した高品質なIssueの作成をサポートします。

## 進め方

1. **問題の理解**: どのような本文のIssueを作成するのか、不明点がなくなるまでユーザーと相談します。
2. **既存Issueの分析**: 新しいIssueを作成する前に、重複がないか確認します。
    - **最近のIssue一覧**: `gh issue list --state all --limit 100`
    - **関連Issueの検索**: `gh issue list --state all --limit 100 --search "<関連するキーワード>"`
    - **確認**: 重複の可能性のあるIssueが見つかった場合、`gh issue view "<issue_number>"`を実行して内容を精査し、重複しているか確認します。
3. **本文とタイトルの下書き**:
    - **本文**: 自由形式（`概要`、`再現手順`、`期待する動作`、`原因`、`ログ`、`一時的な回避策`、`修正案`などの見出し）で`./tmp/issue_body.md`に出力し、ユーザーに確認を求めます。
    - **タイトル**: タイトルだけで内容を推測できるような、簡潔かつ具体的な概要
4. **ラベルの確定**: Issueの内容に基づき、`build`, `ci`, `docs`, `feat`, `bug`, `perf`, `refactor`, `test`, `question`からいずれか1つ選択します。
5. **Issueの作成**: ユーザーの承認を得た後、Issueを作成します。
    - **作成**: `gh issue create --title "<タイトル>" --body-file ./tmp/issue_body.md --label <label>`

## 原則

- **重複回避**: 新しいIssueを作成する前に、常に既存のIssueを検索すること。
- **明確性**: Issueのタイトルと本文は、明確かつ具体的で、他人がコンテキストを理解するのに十分な情報を提供すること。
