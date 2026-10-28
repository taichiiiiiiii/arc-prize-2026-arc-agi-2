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
- **GPU クォータが切れると `kaggle kernels push` が「Maximum weekly GPU quota of 30.00 hours reached」で拒否される（10/27 に発生）。** その間は push も評価もしない。既存 version の提出だけはできる（competition rerun はクォータを使わない）。同じ構成の提出は 2 本までなので、v6 はもう出さない。リセットされたかは push が通るかで確かめる（試すのは 1 日 1 回）。
- 提出は 1 日 1 本（UTC）。採点中の提出があっても、新しい提出は受け付けられる。
- GPU クォータ（週 30h、L4×4 は 2 倍の速さで消費）: smoke 1 回で約 2h、エージェント評価 1 回で約 5h。
- 11/2 の提出は 23:59 UTC 締め切り。11/2 20:00 UTC までに push する。
- 最終の 2 本の選択は 11/2 に、ユーザーに報告したうえで行う（ルールは ledger）。
- evaluation のタスクは学習データに含まれている疑いがあるので、評価の絶対値は高めに出る。使うのは baseline との比較だけ。

## 現在の本番構成
- v6（NVARC 10.5h 優先、エージェントは残り時間、agent_reserve_h 1.0、force_submit_turns 4）。**Public 30.56**（ref 57667359）。同じコードの複製を 10/27 に提出（ref 57701545、ノイズ幅の実測用）。
- v6 の Public で決める次の手（事前登録）: v6 ≥ 26.4 なら NVARC 優先の構成を維持する。v6 < 26.4 なら `skip_agent: true`（NVARC のみ）にする。

## エージェント改善候補（上ほど優先。1 回の評価で 1 つ）
- **v7 に入れることが確定しているもの**（どちらも commit 済み、テスト 21 件通過）: ① 強制 submit で思考を切る（client.py、HTTP 500 の修正）。② 強制 submit がエラーになっても試行を終わらせない（repl_agent.py）。クォータがリセットされたら最初に push する提出版 = v6 のコード + この 2 つ（CONFIG は v6 のまま）。
- 次に評価する候補（優先）: `max_num_seqs: 40` と `concurrency: 40`（同時 32 本で KV cache 63%。48 本だと約 95% で preempt の危険）。2 つは同じ仕組みなので、1 変更として同時に変える。
- その次: MTP の `num_speculative_tokens` を 2 → 3（受理率 0.93/0.85）。server_ladder の rung 0/1 の MTP2 を変える。
0. baseline: v6 のエージェント設定のまま（評価 v1: correct 6, false_verified 1, 58 タスク着手）
1. [測定済み・不採用] `force_submit_turns: 0`（評価 v2: correct 4、false_verified 2。token 上限での失敗が 28 件に増えた）
2. [次の評価。週の GPU クォータがリセットされるまで保留] `thinking_token_budget: 12288`（思考を長くする。上位は長い思考で解いているという報告がある）
3. `attempts: 2`（余り時間で未解決タスクに 2 回目の試行）
4. `temperature: 1.0`（試行の多様性を増やす。上位の duck harness は高温を使っている）
5. `max_cap: 80000`（1 回の試行のトークン上限を広げる。評価の 2h で全タスクを回り切るかを確かめてから）

## GPU クォータ切れの間の手順（1 周につき 1 項目、終わったら [x] を付ける）
クォータ切れの間（push が「Maximum weekly GPU quota」で拒否される間）は、(A) の評価と (B) の push はできない。代わりに下の項目を上から 1 つ進める。どの項目もローカル（CPU、Mac）だけで完結させる。GPU を使う作業や Kaggle への push はしない。Mac の MLX サーバーは、項目で必要なときだけ立ち上げ、終わったら止める（空きメモリを 30% 以上に保つ）。
- [x] （10/27 済み、余裕約 1h。下の ledger を参照）本番 rerun の時間予算を監査する: 12h 制限に対して、hard_end（11.5h）、NVARC の grace 900 s、decode、VALIDATE セルまで足して余裕があるかを、v6 のログ（`scripts/fetch_log.py taichiiiii/arc2-hybrid-v1`、save run の時刻）から見積もる。余裕が 15 分を切るなら reserve_s を増やす変更を v7 に入れる。
- [x] （10/27 済み、テスト 23 件）本番 rerun の失敗経路を監査する: NVARC が例外で落ちた場合、vLLM が起動しない場合、エージェントが例外を出した場合のそれぞれで、submission.json が NVARC の答え（なければ fallback）で必ず書かれることを、テストで確かめる（なければ追加する）。
- [x] （10/27 済み。force_submit_after は今は評価しない。リセット後は v7 で評価 baseline を取り直す）評価 v1 のログ（`~/.hermes/profiles/arc2/cache/scratch/eval_v1.log`）と v2 を突き合わせ、強制 submit の前後で verified した試行の特徴（ターン数、python 回数）を表にする。候補「force_submit_after の値」を評価に回す価値があるかを判断する。
- [ ] 最終 2 本の選択手順を確定する: v6 の複製（57701545）のスコアが出たら、同じコード 2 本の差を「ノイズ幅」の実測値として ledger のルールに反映する。
- [x] （10/28 済み。scratch/v7_push と scratch/eval_v3。scratch は 24h で消えるので、使う直前に再ビルドする）リセット後すぐに push できるよう、v7 の提出版と評価 v3（concurrency 40）のビルドを scratch に作っておき、ビルドが通ることを確かめる（push はしない）。

## 10/28〜11/2 の計画（ユーザー情報: GPU クォータは土曜まで戻らない）
- **10/28〜10/30（GPU なし）**: push は試さない（土曜まで戻らないと分かったので、毎日の push 試行はやめる）。提出枠は見送る。既存 version で出せるのは v6（2 本出し済み、上限）と、劣ると分かっている v3〜v5、S1 だけで、出しても情報も得点も増えないため。ローカルの作業リストを進める。
- **10/31（土）**: クォータのリセット後（00:00 UTC 以降の最初の周で push を試す）。
  1. v7（v6 のコード + 強制 submit の 2 つの修正、CONFIG は v6 のまま）を push し、smoke を通して 10/31 の枠で提出する（約 2h）。
  2. 評価 v3（concurrency 40 / max_num_seqs 40）を push する（約 5h）。
- **11/1（日）**: 評価 v3 が採用なら、v8 = v7 + concurrency 40 を push して提出。不採用なら、評価 v4（MTP 3）を回し、提出は v7 の複製（ぶれの測定）にする。
- **11/2（月、最終日）**: 20:00 UTC までに、その時点の最良構成を push して提出する。そのあと、ledger のルールで最終 2 本を選び、ユーザーに報告する。
- クォータの見積もり: smoke 3 回（約 6h）+ 評価 2 回（約 10h）= 約 16h。30h に収まる。

## ローカル評価ループ（GPU なしの間の (A) の代わり）
- 毎周、`work/local_eval/runs.jsonl` の新しい行を ledger の「ローカル評価」に転記し、事前登録したローカルの判定ルールで判定する。
- **10/28 以降、REPL エージェントのローカル比較キューは止めている（優先度を下げた）。キューを立ち上げ直さないこと。** 理由: 上位は直接回答（direct）方式で、REPL の細かい設定より direct への切り替えのほうが効果が大きい（research 追補を参照）。
- （以前の手順）キュー（`work/local_eval/queue.txt`）が空になりそうなら、候補を 1 つ足す。1 行 1 run で、`<name> --set key=json`。キューの実行プロセスが止まっていたら、terminal の background で `scripts/local_eval_queue.sh` を立ち上げ直す。同時に 2 つは動かさない。
- 有望な候補は、10/31 のクォータのリセット後に Kaggle の評価カーネルで確かめる候補の先頭に入れる。

## direct 方式（10/28 実装、最優先）
- `notebooks/hybrid_v1/src/direct.py` と、hybrid の `agent_mode: "direct"`。NVARC の後の残り時間で、Qwen3.8 に xhigh で直接グリッドを書かせる（1 タスク 2 サンプル、NVARC の確信度が低いタスクから）。統合のルール: NVARC top1 は必ず残す。2 つ目は、direct が一致していれば（2 サンプルが同じか、NVARC top2 と同じ）direct、NVARC が自信あり（conf ≥ 0.75）なら NVARC top2、それ以外は direct。
- direct では max_model_len 131072、max_num_seqs 16（madarshbb の direct-b98k-c16 と同じ値）。
- 10/31 のクォータのリセット後の最初の評価は、評価カーネル（`--set agent_eval_h=2.0 --set agent_mode='"direct"' --set max_model_len=131072 --set max_num_seqs=16`）で行う。REPL の baseline（correct 6）と比べる。
