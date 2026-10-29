# ARC Prize 2026 - ARC-AGI-2

Kaggle: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

## Overview (English)

A Kaggle-notebook solution for ARC-AGI-2 under the competition limits (4x L4, 12 h, no internet).
It combines two public approaches and merges their answers per test output:

1. **NVARC** – a 4B grid model (`sorokin/qwen3_4b_grids15_sft139`) fine-tuned per task at test
   time (LoRA TTT) and decoded with augmented DFS; run first on every task, shortest first.
2. **Qwen3.8-27B (FP8, vLLM, TP4)** on the remaining time, in one of two modes:
   - `direct` – long `xhigh` reasoning that writes the output grid itself (two samples per task,
     with a no-thinking "commit now" rescue when a reply is cut off);
   - `repl` – a program-synthesis agent with `python` / `submit` tools, answers verified on the
     training pairs.
3. **Merge** – NVARC's top-1 is always kept; the second attempt is the agreeing Qwen answer,
   NVARC's top-2 when NVARC is confident, otherwise the Qwen answer. A REPL program that
   reproduces every training pair goes first.

Public LB so far: 27.36 (NVARC copy) → 24.44 (NVARC 7 h + REPL) → **30.56** (NVARC 10.5 h + REPL).
The experiment ledger (Japanese) records every run, decision rule and result:
[analysis/experiment_ledger.md](analysis/experiment_ledger.md).

Competition data is **not** included (rule 2.4.b). Third-party code and its licences are listed
in [NOTICE](NOTICE). Licence: Apache-2.0 ([LICENSE](LICENSE)).

## 概要（日本語）

作業ルールは [AGENTS.md](AGENTS.md)、コンペ概要とルールの要点は [docs/competition_overview.md](docs/competition_overview.md)、
実験記録は [analysis/experiment_ledger.md](analysis/experiment_ledger.md)、改善ループの手順は
[analysis/reports/loop.md](analysis/reports/loop.md)、公開手順は [docs/publication.md](docs/publication.md)。

## セットアップ

```bash
uv sync --group dev          # Python 3.11 + numpy / pytest / ruff
./scripts/download_data.sh   # data/ にコンペデータを展開（gitignore 済み。ルールへの同意が必要）
uv run pytest -q
uv run ruff check .
```

## 提出ノートブックのビルド

```bash
# 提出版（save run は約 1 分で終わり、本処理は competition rerun でのみ動く）
uv run python notebooks/hybrid_v1/build.py --out work/build \
  --set agent_mode='"direct"' --set max_model_len=131072 --set max_num_seqs=16 --set save_mode='"minimal"'
# エージェントだけを evaluation 120 タスクで測る評価カーネル
uv run python notebooks/hybrid_v1/build.py --out work/build_eval --id <user>/arc2-agent-eval \
  --set agent_eval_h=2.0 --set agent_mode='"direct"' --set max_model_len=131072 --set max_num_seqs=16
kaggle kernels push -p work/build
```

## ローカル評価

```bash
# 提出ファイルを evaluation（120 タスク / 172 出力）で採点
uv run python scripts/evaluate.py path/to/submission.json --split evaluation
# Mac（MLX）の小型 Qwen でエージェントの挙動を比べる（mlx_lm.server を :8765 で起動しておく）
uv run python scripts/local_eval.py --name base --tasks 40 --hours 2 --split training
```

## 構成

```
notebooks/hybrid_v1/   提出ノートブックの生成器（build.py）とモジュール（src/）
  src/hybrid.py        全体の流れ（NVARC → vLLM → direct / REPL → 統合）
  src/direct.py        直接回答モード
  src/repl_agent.py    REPL エージェント
  src/arc_*.py         NVARC（TTT + DFS）
notebooks/frontier_copy_3389/  公開 NVARC ノートブックのコピー（S1）
src/arc2/              データ読み込み・評価指標・提出形式
scripts/               評価、ローカル評価、Kaggle の監視・ログ取得、ローカル CI
tests/                 統合・エージェント・各モードのテスト（偽 LLM、GPU 不要）
analysis/              実験台帳・調査・振り返り・ループ手順
docs/                  コンペ概要・公開手順
data/                  コンペデータ（コミットしない）
```
