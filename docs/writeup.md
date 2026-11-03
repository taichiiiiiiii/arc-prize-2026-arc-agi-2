# ARC Prize 2026 – ARC-AGI-2: write-up

Best Public LB: **30.56** (medal zone ≈ 33, 1st place 88.06). This write-up covers what we built,
what we measured, and why the score stayed flat. The detailed day-by-day record, in Japanese, is in
[analysis/experiment_ledger.md](../analysis/experiment_ledger.md).

## 1. Setting

- 240 hidden tasks. Each test output gets two attempts, and only an exact match scores.
- The run must finish within 12 h on 4× L4 (96 GB) with no internet. One submission is allowed per day.
- Public notebooks at the start (late October):
  - The NVARC copies scored about 30–34.
  - A Qwen3.8-27B "direct answer" notebook scored in the mid-30s.
  - Teams combining both reported about 45. The top teams (45–88) made the 27B model the main component.

## 2. System

| Component | What it does | Source |
|---|---|---|
| NVARC | Per-task LoRA test-time training of a 4B grid model, then augmented DFS decoding. Tasks run shortest first. | NVARC (ARC Prize 2025, 1st), via koushikrudra's notebook |
| Qwen3.8-27B server | vLLM 0.19, FP8, TP4, MTP speculative decoding (acceptance 0.93 / 0.85) | madarshbb's notebooks; Tufa Labs' Duck harness |
| `repl` agent | Multi-turn `python` / `submit` tool use. A program is "verified" if it reproduces all training pairs. | madarshbb, extended here |
| `direct` mode | `xhigh` reasoning writes the grid. Up to ~100k thinking tokens and two samples. A no-thinking rescue runs when a reply is cut off. | madarshbb, re-implemented here |
| Merge | NVARC top-1 is always attempt 1. Attempt 2 is chosen in this order: verified program, then the Qwen answer that agrees with NVARC, then NVARC top-2 when NVARC is confident, then the Qwen answer. | this repo |

Engineering details that were worth getting right:

- **The save run does nothing heavy.** The notebook writes a valid fallback submission in about a
  minute and runs the real pipeline only in the competition rerun. Without this, our smoke tests
  on save runs used up the weekly GPU quota.
- **vLLM 0.19 rejects a forced tool call combined with thinking** (HTTP 500,
  `assert content is not None`). A forced `submit` therefore turns thinking off.
- **An evaluation kernel** (`agent_eval_h`) runs only the Qwen part on the 120 public evaluation
  tasks and prints `EVAL {attempted, verified, correct, false_verified}`. After every push, check
  the log for `tasks=N`. Once a default change made the evaluation exit in one minute.
- **Local proxy on Apple Silicon.** `scripts/local_eval.py` runs the agent against a 4–9B Qwen
  under `mlx_lm.server`, which is useful for comparing modes. A local comparison once favoured
  short budgets, but that result did *not* carry over to the 27B model on the evaluation set.

## 3. Results

| Version | Change | Public |
|---|---|---|
| S1 | public NVARC copy | 27.36 |
| S2 | NVARC 7 h + REPL agent | 24.44 |
| v6 | NVARC 10.5 h + REPL agent | 30.56, and 28.06 on a second run of the same code |
| v7 | + shallow direct (~20k tokens/sample) on low-confidence tasks | 30.14 |
| v8 | REPL + forced-submit fixes | 29.72 |
| v9 | + deep direct (~100k tokens/sample) | 30.56 |

v9 put deep direct answers into the ~2.5 h left after NVARC and tied v6, which is consistent
with the analysis in section 4: at the current generation speed, deep direct and the last hours
of NVARC earn about the same points per hour.

Running the same code twice gave 30.56 and 28.06, so any Public difference under about 3 points is noise.

Agent-only evaluation (public evaluation set):

| Run | Budget | Correct |
|---|---|---|
| REPL (6k thinking per turn, 50k per attempt) | 2 h, 120 tasks | 6/120 |
| REPL + forced-submit fixes | 2 h, 120 tasks | 2/120 |
| direct, shallow (~20k tokens/sample) | 2 h, 120 tasks | 2/120 |
| direct, deep (~106k tokens/sample) | 2 h, 24 tasks | **9/24** |

## 4. Why the score did not move

1. **The 27B model only got leftover time.** S2 cut NVARC to 7 h *and* added a weak REPL agent,
   and it dropped by 2.9 points. We read that as "NVARC time is precious". From then on every
   configuration gave Qwen only about 2.5 h. That single comparison mixed two changes, and it was
   also within about one noise width.
2. **We tuned the wrong mode for three days.** We kept tuning a REPL agent that cut its thinking
   into 6k-token turns. The strong public notebooks and the ARC Prize verification
   (Qwen3.8-27B: 42.4% on ARC-AGI-2) both use long direct reasoning.
3. **We tested direct mode at a budget where it cannot work.** Spreading 2 h over 120 tasks gives
   about 20k tokens per sample, and the result (2/120) nearly made us drop the mode. At its
   intended budget it solved 9/24. That measurement came on the last day, when no submission slot
   was left to try a 27B-centred time split.
4. **Our evaluations could not detect the effects we cared about.** With 2–6 correct answers out
   of 120, differences of a few tasks are noise. The pre-registered rule "+5 correct" was almost
   never reachable.
5. **Operations took too much effort.** We built monitoring, CI, watchers and a cron loop, and the
   score was still flat. Smoke runs and a duplicate push used up the GPU quota, and nothing could
   be measured on Kaggle for three days.

The improvement loop was built to *avoid regressions*: pre-registration, noise bands and a merge
that always keeps NVARC top-1. It did that, but it never looked for gains.

## 5. What we would do differently

- **Week 1: measure every candidate method at its intended budget** on a fixed task subset, and
  record accuracy and correct answers per GPU-hour. Split the 12 h by equalising marginal points
  per hour between components, not by one LB comparison.
- **Keep a hypothesis backlog** ranked by expected gain × probability ÷ cost, and refill it from
  three sources: a gap table against top solutions, failure analysis, and our own ceiling
  measurements ([analysis/hypotheses.md](../analysis/hypotheses.md)).
- **Size each evaluation so that the expected effect exceeds about 2σ of count noise.**
- **Make one in three submissions a high-upside bet.** One of the two final picks stays the safe
  configuration, so the bet cannot lower the final result.
- **Cap infrastructure work at about 20% of the effort.**

The redesigned loop is in [analysis/reports/loop.md](../analysis/reports/loop.md). The two
highest-ranked open hypotheses are:

- **H1, a 27B-centred split:** NVARC for 6–7 h, then deep direct for 5 h or more.
- **H2, doubling long-context generation speed:** it currently drops from 370 to 210 tok/s.

## Credits

NVARC (I. Sorokin, J.-F. Puget); Kaggle notebooks by koushikrudra and madarshbb; Tufa Labs' Duck
harness; the Qwen team; and the authors of the Kaggle datasets and models listed in
[NOTICE](../NOTICE).
