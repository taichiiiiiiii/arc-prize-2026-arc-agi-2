---
name: 評価
about: 評価カーネル（arc2-agent-eval）かローカル評価を 1 本回す
title: "評価: "
labels: [arc-loop, evaluation, enhancement]
---

親: #1

## 何を測るか
<!-- 変える設定は 1 つ。比べる相手（baseline） -->

## 条件
- タスク: <!-- 全 120 / 固定 seed の部分集合 N -->
- 予算: <!-- 時間、1 サンプルのトークン数（方式が前提とする予算か） -->
- ビルド: `build.py --id taichiiiii/arc2-agent-eval --set agent_eval_h=... --set ...`

## 検出力と判定（事前登録）
<!-- 期待効果が 2σ ≈ 2√(n·p·(1−p)) を超えるか。結果がどうなれば判断が変わるか -->

## 受け入れ条件
- [ ] ログで `tasks=N` を確かめた（空振りでない）
- [ ] EVAL の結果を ledger に書き、判定した
