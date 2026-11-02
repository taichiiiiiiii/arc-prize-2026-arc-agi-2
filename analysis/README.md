# analysis/ — 実験の記録（日本語）

The files in this directory are written in Japanese. For an English summary, see [docs/writeup.md](../docs/writeup.md).

| ファイル | 中身 |
|---|---|
| [experiment_ledger.md](experiment_ledger.md) | 実験台帳。全実行・提出の設定とスコア、事前登録した判定ルール、「うまくいかなかったこと」を時系列で記録している。最後に総括（2026-11-02）がある |
| [hypotheses.md](hypotheses.md) | 仮説バックログ（期待上積み × 確率 ÷ コスト）と、上位との差の表 |
| [reports/loop.md](reports/loop.md) | 改善ループの手順書 v2（上積みを取りに行くループ） |
| [reports/loop_2026-10.md](reports/loop_2026-10.md) | 旧手順書（悪化を防ぐことが中心のループ。比較用に残している） |
| [reports/research_2026-10-25.md](reports/research_2026-10-25.md) | 技術調査（LB の構造、公開ノートブック、NVARC、27B direct） |
| [reports/retrospective_2026-10-28.md](reports/retrospective_2026-10-28.md) | 中間の振り返り |

これらは作業中にそのまま書いた生の記録で、後から整えていない。運用の細部（監視スクリプト、定期実行のスケジューラ、ローカルの作業用ディレクトリなど）もそのまま残している。

台帳には、自動の改善ループ（エージェントが定期実行）の記録と、親セッション（人の指示を受けた作業）の記録が混ざっている。見出しの「ループ」は自動実行、「親セッション」は対話中の作業を指す。
