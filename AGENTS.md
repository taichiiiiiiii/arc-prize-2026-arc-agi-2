# ARC Prize 2026 - ARC-AGI-2

応答・ドキュメント・Issue・実験台帳は日本語で書く。コード・識別子・コミットメッセージは英語で書く。

## 境界

- コンペデータ (`data/`) はコミット・公開・再配布しない（ルール 2.4.b）。
- **このリポジトリは公開を前提にする**。コンペ期間中に外部へ公開すると、Kaggle の forum か notebook でも全参加者に共有する義務が生じる（Rules 6.a/b）。そのため、public にするのは締め切り（2026-11-02 23:59 UTC）の後にする。手順は `docs/publication.md`。それまでも、**いつでも公開できる状態を保つ**:
  - 追跡するファイルに、個人の絶対パス（`/Users/...`、`~/.hermes/...`）、ユーザー名に依存するパス、認証情報、トークン、コンペデータ（タスクのグリッドそのもの）を書かない。場所は環境変数（`ARC2_SCRATCH`、`ARC2_MLX_DIR`、`KAGGLE_PY`）か、リポジトリからの相対パス（`work/` は gitignore）にする。
  - 他者のコードを取り込むときは、出典とライセンスを `NOTICE` とファイル先頭の docstring に書く（Kaggle の公開ノートブックは Apache 2.0）。
  - コード・コメント・コミットメッセージは英語。README は英語の概要と日本語の詳細を並べる。
- `~/.kaggle/` の認証情報は読まない・表示しない・送信しない。
- `kaggle kernels push` と `kaggle competitions submit` は、その操作ごとにユーザーの明示的な承認を得てから親エージェントだけが行う。**提出は 1 日 1 回まで**なので、同じ日に 2 回目は送らない。
- 読み取り専用の Kaggle API（files / submissions / leaderboard / kernels list・pull・status・output / topics）は承認なしで使ってよい。認証エラーが出たら `kaggle auth login`（ブラウザの OAuth）で再認証する。
- 提出ノートブックは private にし、インターネットを無効にする。外部データとモデルは、無料で公開されていてオープンソースライセンスのものに限る（入賞時は CC BY 4.0 で公開する義務がある）。
- ローカルで重い処理（MLX の評価など）を回すときは、同じマシンの他のジョブを止めない。事前に空きメモリを確認する（macOS では `memory_pressure`）。

## Git

- コミットの前に `uv run ruff check .` と `uv run pytest -q` が通ることを確認する。
- 作業ブランチで作業し、main と develop へ直接 push しない。force-push しない。

## Git 運用（1 つの作業 = 1 つの Issue = 1 つのブランチ）
1. **Issue を立てる**: 日本語で書く（テンプレートは `.github/ISSUE_TEMPLATE/`）。ラベルは `arc-loop` と種類（`bug` / `enhancement` / `documentation`）。必要に応じて `submission`（Kaggle の提出・push）、`evaluation`（評価カーネル・ローカル評価）、`hypothesis`（`analysis/hypotheses.md` の Hn）を付ける。リポジトリ管理者の判断が要るものには `needs-decision` を付ける。追跡用の親は #1。
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
