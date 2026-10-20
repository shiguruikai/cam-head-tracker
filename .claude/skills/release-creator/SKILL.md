---
name: release-creator
description: Use when releasing the next version, from choosing the version number to publishing the Release and following up on related Issues.
---

Pushing a `v*` tag makes `build-release.yml` build the exe and create a draft Release. Release notes are generated from PR titles and labels using the categories in `.github/release.yml`; PRs labeled `skip-changelog` are excluded.

1. Update `master`. If a PR was just merged, finish pr-creator's after-merge steps first, including the `update-notice.yml` wait.
2. Propose the next version based on `git log <latest tag>..HEAD`. Use SemVer; in 0.x, bump the minor version for breaking changes or new features, and the patch version otherwise.
3. On a `build/vX.Y.Z` branch, update `version` in `pyproject.toml` and run `uv lock`. Use `git grep` to make sure the version number is not hard-coded anywhere else. After the user approves, commit as `build: bump version to vX.Y.Z`.
4. Create the PR following pr-creator, with the labels `build` and `skip-changelog`, and follow its after-merge steps.
5. On `master`, confirm that HEAD is `build: bump version to vX.Y.Z (#N)`. After the user approves, create the tag on it with `git tag -a vX.Y.Z -m "Release vX.Y.Z"` and push it. Find the run with `gh run list --workflow build-release.yml --branch vX.Y.Z`, wait with `gh run watch <id> --exit-status`, and check the draft body and the attached zip with `gh release view vX.Y.Z`.
6. Before the user publishes the draft:
   - If there are breaking changes, draft a `## ⚠️ アップデート時の注意` section in Japanese for users, to go above `## What's Changed`.
   - Ask the user to download the zip from the draft and check that the app works.
7. Related Issues are those listed under 関連 Issue in the PRs since the previous tag. For each one, draft a short reply in Japanese: thanks for the report, what was improved, a link to the release, and any caveats. The user posts it and closes the Issue. If the reporter does not respond, suggest closing it as completed after about two weeks.

Do not move on until the user has finished their steps (merging and publishing).
