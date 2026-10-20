---
name: pr-creator
description: Use when creating a pull request, and when syncing the local repository after the user merges it.
---

The user squash-merges PRs on GitHub. The squash commit takes the PR title (or the commit message if the PR has only one commit) and lists the commit messages in its body; release notes use the PR title. Keep the title and the commit messages consistent.

- If there are uncommitted changes, ask the user how to handle them. If on `master`, create a `<type>/<summary>` branch.
- If code, dependencies, or build inputs changed (e.g. `*.py`, `pyproject.toml`, `uv.lock`, `build.spec`, `cam_head_tracker/assets/`), run `task lint format test build`. If it fails, stop and report to the user.
- Title: `<type>: <specific summary in Japanese>`. Pick one type for the main change, and add the same type as a label.
- Body: write `.tmp/pr_body.md` in Japanese, following `.github/pull_request_template.md`.
  - The file may be left over from an earlier PR; do not reuse its content.
  - Remove comments and empty headings. Keep every checklist item, and check only the completed ones.
  - Describe breaking changes under 影響範囲.
  - Ask the user whether a related Issue should close on merge (`close #N`) or only be listed under 関連 Issue.
- After the user approves, run `git push -u origin HEAD` and `gh pr create --title "<title>" --body-file .tmp/pr_body.md --label <type>`. If the user wants to run them, show the commands instead.
- Ask the user to merge the PR and delete the remote branch (it is not deleted automatically), and wait until they do.
- After the merge:
  - Pull `master` and run `git fetch --prune`.
  - Confirm with `gh pr view <branch> --json state,headRefOid` that the PR is `MERGED` and that `headRefOid` equals `git rev-parse <branch>`, then delete the branch with `git branch -D <branch>` (`-d` fails after a squash merge).
  - If the PR changed `pyproject.toml` or `uv.lock`, `update-notice.yml` runs on `master` and commits `NOTICE.txt` only when it changes. Wait for the run to finish (`gh run list --workflow update-notice.yml --limit 1`), then pull again.
