# ARC Prize 2026 - ARC-AGI-2

応答・ドキュメント・Issue・実験台帳は日本語で書く。コード・識別子・コミットメッセージは英語で書く。

## 境界

- コンペデータ (`data/`) はコミット・公開・再配布しない（ルール 2.4.b）。リポジトリは private のまま運用する。
- `~/.kaggle/` の認証情報は読まない・表示しない・送信しない。
- `kaggle kernels push` と `kaggle competitions submit` は、その操作ごとにユーザーの明示的な承認を得てから親エージェントだけが行う。**提出は 1 日 1 回まで**なので、同じ日に 2 回目は送らない。
- 読み取り専用の Kaggle API（files / submissions / leaderboard / kernels list・pull・status・output / topics）は承認なしで使ってよい。認証エラーが出たら再試行せずにユーザーへ報告する。
- 提出ノートブックは private にし、インターネットを無効にする。外部データとモデルは、無料で公開されていてオープンソースライセンスのものに限る（入賞時は CC BY 4.0 で公開する義務がある）。
- このマシン (M4 Pro / 48 GB) はユーザーの Docker や他の学習ジョブと共有している。重い処理の前に `memory_pressure` を確認し、他プロジェクトのプロセスには触れない。

## Git

- コミットの前に `uv run ruff check .` と `uv run pytest -q` が通ることを確認する。
- 作業ブランチで作業し、main と develop へ直接 push しない。force-push しない。

## Git 運用（1 つの作業 = 1 つの Issue = 1 つのブランチ）
1. **Issue を立てる**: 日本語で書く。ラベルは `arc-loop` と種類（`bug` / `enhancement`）。オーナーの判断が要るものには `needs-decision` を付ける。追跡用の親は #1。
2. **ブランチを切る**: 最新の `origin/develop` から `<種類>/<Issue番号>-<内容>` で切る（例: `enhancement/5-git-workflow`）。
3. **コミットする**: 1 つの論理単位ごとにコミットし、本文に `Refs #番号` を書く。メッセージは `feat(...)` / `fix(...)` / `docs(...)` / `ci(...)` の形。
4. **PR を作る**: `gh pr create --base develop`。本文に、変更内容、数値（テスト件数、評価の値など）、`Fixes #番号` を書く。
5. **CI を通す**: `tests` が pass になるまでマージしない。GitHub Actions が Billing で止まっている間（#4）は、`scripts/ci_local.sh <PR番号>` で commit status `tests` を付ける。
6. **マージする**: Issue の受け入れ条件を満たしたら `gh pr merge --merge --delete-branch`（マージコミットを残す）。
7. **確認と記録**: マージ後の `develop` でテストや測定をやり直し、結果を Issue にコメントして、#1 のチェックを更新する。
- 実験台帳（`analysis/experiment_ledger.md`）だけの追記は、その時点で進めている作業の Issue ブランチに含める。進行中の作業がないときは、日ごとの `docs/1-ledger-YYYYMMDD` ブランチにまとめて PR にする（`Refs #1`）。
- Kaggle の push と提出は、マージ後の develop のコードから行う。提出には、使った commit をメッセージに書く。
- 認証情報、コンペデータ、生成物（submission.json、重み）はステージしない。

## 評価

- ローカルでの評価は `arc-agi_evaluation_*`（120 タスク / 172 出力）を基準にする。`arc-agi_test_challenges.json` はプレースホルダーで、中身は training の 240 タスクであり、評価には使えない。
- Public LB は隠しテスト 240 タスクの一部なので、LB の小さな差だけで判断しない。
