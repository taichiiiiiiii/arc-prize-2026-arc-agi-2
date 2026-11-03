# リポジトリ公開の手順と監査記録

## 方針
- 締め切り（2026-11-02 23:59 UTC）までは private のままにする。期間中に外部へ公開する場合は、Kaggle の forum か notebook でも全参加者に共有する義務があるため（Rules 6.a「Private Code Sharing」、6.b「Public Sharing」）。
- 締め切り後に public にする。入賞した場合は、Winner License（CC BY 4.0）とオープンソース化の要件（7 日以内に Solution Writeup に添付）に従う。
- 自分のコードは Apache-2.0（`LICENSE`）。他者のコードの出典は `NOTICE` に書く。

## 監査記録（2026-10-29、develop の enhancement/7 時点）
| 項目 | 方法 | 結果 |
|---|---|---|
| コンペデータが履歴にない | `git log --all --name-only` で `data/`・`*challenges*`・`*solutions*`・`submission.json` を探した | 0 件 |
| グリッドのデータが埋め込まれていない | 履歴の追加行から、10 個以上の数字が並ぶ `[[...]]` を探した | 0 件（テストは合成した 1×2 程度のタスクだけ） |
| 認証情報 | 履歴全体から `KGAT_`、`ghp_`、`github_pat_`、`sk-…`、`hf_…`、`AKIA…`、秘密鍵ヘッダを探した | 0 件。`kaggle.json`、`.kaggle/`、`.env` は gitignore |
| 個人の絶対パス（HEAD） | `git grep '/Users/\|\.hermes'` | 0 件（AGENTS.md のルールの説明文を除く）。スクリプトは `ARC2_SCRATCH`、`ARC2_MLX_DIR`、`KAGGLE_PY`、相対パスを使う |
| 個人の絶対パス・個人のメール（履歴） | `git log --all -p` と author/committer を確認 | 2026-11-03 に `git filter-repo` で履歴を書き換えた（#58）。マージコミット 9 件の個人メールを noreply アドレスに、ローカルのユーザー名を含むパスを `/Users/<user>` に置き換えた。書き換え後は 0 件。古い SHA と新しい SHA の対応は `docs/commit-map.tsv` |
| 他者のコード | koushikrudra（NVARC）、madarshbb（Qwen エージェント、direct）、Tufa Labs の duck harness | `NOTICE` に出典を書いた。Kaggle の公開ノートブックは Apache 2.0 |
| モデルの重み・wheel | `*.safetensors`、`*.bin`、`*.gguf`、`kaggle_datasets/` は gitignore | 履歴に 0 件 |

## 再監査（2026-11-02、#43）
上の表の項目を develop の最新で確認し直した。コンペデータ・グリッドの埋め込み・認証情報・個人の絶対パス（HEAD）・Discord などの個人 ID・500 KB 超のファイルは、いずれも 0 件だった。外部の人が再現できるように、次の点も整えた。
- カーネル ID を `$KAGGLE_USERNAME` から作るようにした（`build.py`）。
- 英語の README と解説（`docs/writeup.md`）を書き、日本語版は `README.ja.md` に移した。
- `analysis/README.md` に台帳の索引を置いた。

## 公開する日の手順（11/3 以降）
1. この監査を develop の最新でやり直す（上の表のコマンド）。差分があれば、Issue と PR で直す。
2. README の結果欄を、最終の Public / Private スコアに更新する（PR）。
3. `develop` を `main` へ PR でマージする（`--merge`）。
4. `gh repo edit --visibility public --accept-visibility-change-consequences`。
5. public になると GitHub Actions が無料になるので、`.github/workflows/tests.yml` の `on:` を push / pull_request に戻す（#4 を閉じる）。
6. 入賞圏なら、Kaggle の Solution Writeup にリポジトリと notebook のリンクを付ける。

## 公開の記録（2026-11-03）
- 最終レビュー（外部レビュー担当、読み取り専用）で指摘ゼロを確認してから、リポジトリを public にした。
- GitHub Actions の tests を push / pull_request で自動実行に戻した（#65）。
