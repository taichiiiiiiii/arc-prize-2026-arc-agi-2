# ARC Prize 2026 - ARC-AGI-2

Kaggle: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

作業ルールは [AGENTS.md](AGENTS.md)、コンペ概要とルールの要点は [docs/competition_overview.md](docs/competition_overview.md)、
実験記録は [analysis/experiment_ledger.md](analysis/experiment_ledger.md)。

## セットアップ

```bash
uv sync                      # Python 3.11 + numpy / pytest / ruff
./scripts/download_data.sh   # data/ にコンペデータを展開 (gitignore 済み)
uv run pytest -q
uv run ruff check .
```

## ローカル評価

```bash
# 評価セット (120 タスク / 172 出力) でスコアを計算
uv run python scripts/evaluate.py path/to/submission.json --split evaluation
# 動作確認用のベースライン (テスト入力をそのまま返す)
uv run python scripts/evaluate.py --baseline identity --split evaluation
```

## 構成

```
src/arc2/        data.py (読み込み) / metric.py (評価指標) / submission.py (生成・形式検証)
scripts/         download_data.sh, evaluate.py
tests/           指標と提出形式のテスト
notebooks/       Kaggle 提出用ノートブック
analysis/        実験台帳・レポート
docs/            コンペ概要
data/            コンペデータ (コミットしない)
```
