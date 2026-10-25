# 改善ループの手順書（cron から毎回これを読んで 1 周する）

目的: 締め切り 2026-11-02 23:59 UTC までに Private で最も高いスコアを出すこと。提出は 1 日 1 本で、毎日の枠を使い切る。Public への過学習は防ぐ（ledger の「過学習を防ぐルール」に従う）。

## 毎周の手順
1. **状態を読む**: `analysis/experiment_ledger.md` の末尾、`kaggle competitions submissions arc-prize-2026-arc-agi-2 --page-size 20`、`date -u`、`ps aux | grep -E 'watch_submit|kernels status' | grep -v grep`（動いている監視スクリプト）。認証エラーが出たら何もせず、`kaggle auth login` の依頼だけを報告する。
2. **新しいスコアを記録する**: まだ ledger にない Public スコアがあれば、ref・内容・スコアを追記する。そのとき、事前に書いたルール（S2 と S1 の比較など）で判定し、結論を書く。
3. **次の候補を 1 つ決める**（1 周につき 1 つまで）。候補の優先順位:
   a. 判定ルールで決まった次の手（例: S2 が S1 より 0.5 以上低ければ、NVARC をフル（11h）に戻す）
   b. 仕組みで説明できる改善を 1 つ（例: force_submit_turns による「submit しない失敗」の削減。改善点のリストは「## 候補リスト」を参照）
   c. 何もなければ、現在の最良構成の別シード（sampling seed）。最終枠の分散を減らすためで、Public の点数を探すためではない。
4. **build → push → smoke**: `uv run python notebooks/hybrid_v1/build.py --out <scratch> --id taichiiiii/arc2-hybrid-v1`、続けて `kaggle kernels push -p <scratch>`。push する前に `uv run ruff check .` と `uv run pytest -q` を通し、kernel-metadata の `is_private: true` を確認する。設定値は build.py の CONFIG を 1 か所だけ変える。
5. **自動提出の監視を張る**: `scripts/watch_submit_hybrid.sh taichiiiii/arc2-hybrid-v1 <version> "<message>"` を、terminal の background と persist_on_release で起動する。このスクリプトは、smoke の目印（"server ready" と "SMOKE score merged"）がログにあり、その日の UTC の枠が空いているときだけ提出する。同じ version 用の監視スクリプトがすでに動いていれば、新しく起動しない。
6. **ledger に追記して commit・push する**: 作業ブランチ `feat/research-frontier-copy` に、何を変えたか、なぜか、smoke の結果を書く。
7. **報告**: 日本語で 5 行以内。新しいスコア、判定、次に出す候補、提出予定時刻。

## 制約
- 提出は 1 日 1 本（UTC）。採点中の提出があっても、新しい提出は受け付けられる（10/25 に確認）。
- GPU クォータ: smoke 1 回で L4×4 を約 1.5h 使う（クォータは約 3h 分）。週 30h なので、1 日 1 回の push に抑える。
- 締め切りが近いとき: 11/2 の提出は 23:59 UTC までに。smoke と push に約 2h かかるので、11/2 20:00 UTC までに push する。
- 最終の 2 本の選択は 11/2 に行う。ルールは ledger に書いてある。選択は最終日に、ユーザーに報告したうえで行う。

## 候補リスト（仕組みで説明できるもの、上ほど優先）
1. force_submit_turns=4（v4 の候補。実装済み、push はまだ）
2. NVARC の時間配分: 7h と 11h（判定ルールで決める）
3. エージェントの試行を 2 回にする: NVARC の確信度が低いタスクだけ、2 回目の attempt を出す（未実装）
4. 分割方式（NVARC を GPU 2 枚、vLLM を TP2 で同時に動かす。実装済み、smoke は未実施）
