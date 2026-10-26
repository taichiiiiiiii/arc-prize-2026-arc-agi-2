# 改善ループの手順書（cron から毎回これを読んで 1 周する）

目的: 締め切り 2026-11-02 23:59 UTC までに Private で最も高いスコアを出すこと。Public への過学習は防ぐ（ledger の「過学習を防ぐルール」に従う）。
ループは 2 本立て: **(A) エージェント評価ループ**（evaluation 120 タスクでエージェントの変更を 1 つずつ測る）と、**(B) 提出ループ**（評価を通った構成だけを本番に入れ、1 日 1 本提出する）。

## 毎周の手順
1. **状態を読む**: `analysis/experiment_ledger.md` の末尾（特に「エージェント評価」表）、`kaggle competitions submissions arc-prize-2026-arc-agi-2 --page-size 20`、`kaggle kernels status taichiiiii/arc2-agent-eval`、`kaggle kernels status taichiiiii/arc2-hybrid-v1`、`date -u`、`ps aux | grep -E 'watch_submit' | grep -v grep`。認証エラーが出たら何もせず、`kaggle auth login` の依頼だけを報告する。
2. **新しい結果を記録する**:
   - Public スコア: まだ ledger にないものを追記し、事前登録した判定ルールで判定する。
   - エージェント評価: `arc2-agent-eval` が COMPLETE で、まだ記録していない version があれば、`scripts/fetch_log.py taichiiiii/arc2-agent-eval <out>` でログを取り、`EVAL {...}` の行（attempted / verified / correct / false_verified / outputs）を ledger の「エージェント評価」表に追記する。
3. **(A) 次の評価を 1 つ回す**（評価カーネルが動いていなければ）:
   - 「エージェント改善候補」の上から、まだ測っていないものを 1 つ選ぶ。変える設定は 1 つだけ。比べる相手は直前の採用構成（baseline）。
   - `uv run python notebooks/hybrid_v1/build.py --out <scratch> --id taichiiiii/arc2-agent-eval --set agent_eval_h=2.0 --set <key>=<json>` でビルドし、`kaggle kernels push -p <scratch>`。コードを変える候補は、先にテストを書いて ruff と pytest を通す。
   - 1 回あたり GPU クォータを約 5h 使う。週の残りが 8h を切ったら評価を止め、提出用の smoke にクォータを残す。
4. **採用の判定**（事前登録、変えない）: 候補が baseline に比べて **correct が +5 タスク以上**、かつ **false_verified が増えていない**なら採用する。それ以外は不採用。不採用でも、仕組み上の理由が分かったら ledger に書く。
5. **(B) 提出**: その日の UTC の枠が未使用なら、「現在の本番構成」に、採用済みのエージェント変更を反映して `arc2-hybrid-v1` に push する（`build.py --id taichiiiii/arc2-hybrid-v1`、CONFIG を直接編集してコミット）。続けて `scripts/watch_submit_hybrid.sh taichiiiii/arc2-hybrid-v1 <version> "<message>"` を background と persist_on_release で起動する。前回の提出から本番構成が変わっていなければ、同じ構成を別シードで出してよい（最終 2 本のぶれを測るため）。ただし同じ構成を 3 本以上は出さない。
6. **ledger に追記して commit・push する**: 作業ブランチ `feat/research-frontier-copy`。新しい落とし穴や手順が分かったら、スキル `arc-agi-2-repo` に追記する。
7. **報告**: 日本語で 5 行以内。新しいスコアや評価結果、判定、次に回すもの。

## 制約
- 提出は 1 日 1 本（UTC）。採点中の提出があっても、新しい提出は受け付けられる。
- GPU クォータ（週 30h、L4×4 は 2 倍の速さで消費）: smoke 1 回で約 2h、エージェント評価 1 回で約 5h。
- 11/2 の提出は 23:59 UTC 締め切り。11/2 20:00 UTC までに push する。
- 最終の 2 本の選択は 11/2 に、ユーザーに報告したうえで行う（ルールは ledger）。
- evaluation のタスクは学習データに含まれている疑いがあるので、評価の絶対値は高めに出る。使うのは baseline との比較だけ。

## 現在の本番構成
- v6（NVARC 10.5h 優先、エージェントは残り時間、agent_reserve_h 1.0、force_submit_turns 4）。10/26 13:07 UTC に提出（ref 57667359）。
- v6 の Public で決める次の手（事前登録）: v6 ≥ 26.4 なら NVARC 優先の構成を維持する。v6 < 26.4 なら `skip_agent: true`（NVARC のみ）にする。

## エージェント改善候補（上ほど優先。1 回の評価で 1 つ）
0. baseline: v6 のエージェント設定のまま（評価 v1: correct 6, false_verified 1, 58 タスク着手）
1. [測定済み・不採用] `force_submit_turns: 0`（評価 v2: correct 4、false_verified 2。token 上限での失敗が 28 件に増えた）
2. [次の評価。週の GPU クォータがリセットされるまで保留] `thinking_token_budget: 12288`（思考を長くする。上位は長い思考で解いているという報告がある）
3. `attempts: 2`（余り時間で未解決タスクに 2 回目の試行）
4. `temperature: 1.0`（試行の多様性を増やす。上位の duck harness は高温を使っている）
5. `max_cap: 80000`（1 回の試行のトークン上限を広げる。評価の 2h で全タスクを回り切るかを確かめてから）
