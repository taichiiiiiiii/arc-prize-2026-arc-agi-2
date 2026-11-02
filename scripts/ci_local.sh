#!/usr/bin/env bash
# Local stand-in for the GitHub Actions `tests` job (Actions is blocked by account billing).
# Runs ruff + pytest on a clean checkout of the PR head and posts the result as a commit
# status named `tests` on that commit, so `gh pr checks` shows it.
#   scripts/ci_local.sh <pr-number>
set -eu
PR=$1
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
# the PR head straight from git (gh pr view can lag a just-pushed commit)
BR=$(gh pr view "$PR" --json headRefName -q .headRefName)
SHA=$(git ls-remote origin "refs/heads/$BR" | cut -f1)
DIR=$(mktemp -d "${TMPDIR:-/tmp}/ci-XXXXXX")
trap 'git worktree remove --force "$DIR" >/dev/null 2>&1 || true' EXIT
git fetch -q origin
git worktree add -q --detach "$DIR" "$SHA"
gh api -X POST "repos/$REPO/statuses/$SHA" -f state=pending -f context=tests \
  -f description="local: ruff + pytest running" >/dev/null
if (cd "$DIR" && uv sync -q --group dev && uv run ruff check . && uv run pytest -q) > "$DIR.log" 2>&1; then
  STATE=success; DESC="local: $(grep -oE '[0-9]+ passed' "$DIR.log" | tail -1), ruff clean"
else
  STATE=failure; DESC="local: failed (see PR comment)"
fi
gh api -X POST "repos/$REPO/statuses/$SHA" -f state=$STATE -f context=tests -f description="$DESC" >/dev/null
echo "tests: $STATE ($DESC) on ${SHA:0:7}"
[ "$STATE" = success ] || { gh pr comment "$PR" --body "$(printf 'ローカル CI 失敗（%s）\n```\n%s\n```' "${SHA:0:7}" "$(tail -40 "$DIR.log")")"; exit 1; }
