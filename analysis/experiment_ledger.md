# 実験台帳

## 2026-10-25 セットアップ
- リポジトリ・環境（uv / Python 3.11）・ローカル評価器（`scripts/evaluate.py`）を用意した。
- 恒等ベースライン（test 入力をそのままコピー）のスコアは evaluation 0.0000、training 0.0000。採点処理の動作確認としては想定どおり。
- 提出: なし。

## 2026-10-25 調査と方針
- 詳細は [reports/research_2026-10-25.md](reports/research_2026-10-25.md)。公開ノートブックの最高は 33.89（NVARC 4B TTT）。上位は Qwen3.8-27B の REPL エージェントが主流と推測。本線は NVARC と Qwen3.8-27B エージェントの組み合わせで、Public 45 以上を目標にする。
- S1: `taichiiiii/arc2-frontier-copy-3389` の v1 を push した（ユーザー承認済み）。中身は `koushikrudra/failed-in-aimo` の V1 で、ipynb は cmp で一致することを確認済み。is_private=True。完了後に自動で提出し、スコアを監視する（`scripts/watch_frontier_copy.sh`）。期待値は 33.89 付近（銀ラインの近く）。
