# ARC Prize 2026 – ARC-AGI-2（日本語）

[English README](README.md) · [解説（英語）](docs/writeup.md) · [実験台帳](analysis/experiment_ledger.md)

Kaggle の [ARC Prize 2026 – ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) の解法です。コンペの制約（L4×4、12 時間、インターネットなし）の中で、2 つの公開手法を組み合わせ、テスト出力ごとに答えを統合します。

- **NVARC**: 4B のグリッドモデルを、タスクごとにテスト時学習（LoRA TTT）して、拡張 DFS で答えを出します。短いタスクから順に処理します。
- **Qwen3.8-27B**: vLLM（FP8、TP4）で、残りの時間に動かします。方式は 2 つです。
  - `direct`: xhigh で長く考えて、グリッドを直接書きます。
  - `repl`: `python` と `submit` のツールで、プログラムを合成します。
- **統合**: NVARC の 1 位の答えを必ず残します。2 つ目の答えは、検証済みのプログラム、NVARC と一致した Qwen の答え、NVARC の 2 位などの中から選びます。

## 結果

最良の Public は **30.56** でした（メダル圏は約 33、1 位は 88.06）。同じコードでも 30.56 と 28.06 の差が出たので、3 点以内の差はノイズとみなします。全提出の表は [README.md](README.md#results) にあります。精度が伸びなかった原因は、[解説](docs/writeup.md) と台帳の 2026-11-02 の総括に書きました。主な原因は次のとおりです。

- 27B に残りの時間しか回さなかった。
- 主流の方式（direct）に気づくのが遅れた。
- 評価の検出力が足りなかった。

## ドキュメント

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
export KAGGLE_USERNAME=<自分のユーザー名>   # カーネル ID は $KAGGLE_USERNAME/arc2-hybrid-v1
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
scripts/               評価、ローカル評価、Kaggle の提出監視・ログ取得、ローカル CI
tests/                 統合・エージェント・各モードのテスト（偽 LLM、GPU 不要）
analysis/              実験台帳・調査・振り返り・ループ手順
docs/                  コンペ概要・公開手順
data/                  コンペデータ（コミットしない）
```
