# 実験台帳

## 2026-10-25 セットアップ
- リポジトリ・環境（uv / Python 3.11）・ローカル評価器（`scripts/evaluate.py`）を用意した。
- 恒等ベースライン（test 入力をそのままコピー）のスコアは evaluation 0.0000、training 0.0000。採点処理の動作確認としては想定どおり。
- 提出: なし。

## 2026-10-25 調査と方針
- 詳細は [reports/research_2026-10-25.md](reports/research_2026-10-25.md)。公開ノートブックの最高は 33.89（NVARC 4B TTT）。上位は Qwen3.8-27B の REPL エージェントが主流と推測。本線は NVARC と Qwen3.8-27B エージェントの組み合わせで、Public 45 以上を目標にする。
- S1: `taichiiiii/arc2-frontier-copy-3389` の v1 を push した（ユーザー承認済み）。中身は `koushikrudra/failed-in-aimo` の V1 で、ipynb は cmp で一致することを確認済み。is_private=True。完了後に自動で提出し、スコアを監視する（`scripts/watch_frontier_copy.sh`）。期待値は 33.89 付近（銀ラインの近く）。
- S1 v1 は保存時の実行で失敗した。原因は Kaggle Models のマウント先が `/kaggle/input/models/sorokin/...` に変わっていたことで、NVARC がモデルを見つけられなかった。**今の公開フォークも、パスを直さないと同じ理由で落ちる可能性がある。** v2 ではモデルのパスを 1 行だけ直して push した（diff は 1 行）。完了したら自動で提出する。
- hybrid_v1 の実装は `notebooks/hybrid_v1/` にある。NVARC を 5 時間回したあと、Qwen3.8-27B FP8 の REPL エージェントを vLLM で動かす（vLLM は同梱の CPython 3.12 で起動）。統合ルール: エージェントが検証済みの答えを最優先にし、次に NVARC の上位、最後にエージェントの未検証答えを使う。テスト 13 件が通過（統合ロジック、偽 LLM でのエージェントループ、NVARC デコード）。Kaggle Dataset の作成と、動作確認のための push はユーザー承認待ち。
- S1 v2 を 2026-10-24 19:11 UTC に提出（ref 57604463、PENDING）。保存時の実行（evaluation 4 タスク）の Reload score は 2.5。
- NVARC の所要時間は 1 タスクあたり、小さいタスクで 107〜350 秒、大きいタスクで 536〜1216 秒（smoke と S1 のログより）。
- hybrid_v1 v1 の smoke 結果: NVARC は 3/11 で、エージェントは動かなかった。原因は CPython 3.12 の pip が utility script の PYTHONPATH を引き継ぎ、読み取り専用の場所にある zipp を消そうとして失敗したこと。v2 では環境変数を空にして実行し、vllm の import 確認と /kaggle/working の後片付けを追加した。出力ファイルが大量にあると CLI の output 取得が 429 になるので、ログだけを取る `scripts/fetch_log.py` を作った。
- hybrid v2 の smoke（evaluation 8 タスク / 11 出力）: Python 3.12 環境の準備に 263 秒、vLLM の起動に 450 秒（TRITON_ATTN + FP8 KV + MTP2 の設定で起動）。NVARC は 5/11、エージェントは verified 2/8 だが test は 0/11 正解、統合後も 5/11。生成速度は同時 1 本で約 34 tok/s、同時 8 本で約 245 tok/s とほぼ比例して伸びる。エージェントの 6/8 は submit を一度もせずに 60k トークンの上限を使い切った（考えすぎ）。
- v3: NVARC を 7 時間に延ばし、短いタスクから処理（smoke の実測から見て、ほぼ全タスクを処理できる見込み）。エージェントは force_submit_after=5、thinking を 1 ターン 6144 トークンまで、同時 32 本、1 回の試行の上限を 50k トークンに変更。S2 として、smoke の結果が出て、PENDING の提出がなく、その日の UTC の枠が空いていれば自動で提出する（`scripts/watch_submit_hybrid.sh`）。
- 事前に決めておく判定ルール: S2 の Public が S1 より 0.5 以上低ければ、NVARC を縮めたことが原因とみて、NVARC をフル（11 時間）に戻す。エージェントは、NVARC が処理を終えたあとの残り時間だけで回す。
