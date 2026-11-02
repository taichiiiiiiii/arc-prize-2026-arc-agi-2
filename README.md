# ARC Prize 2026 – ARC-AGI-2 (Kaggle)

[日本語版 README](README.ja.md) · [Solution write-up](docs/writeup.md) · [Experiment ledger (JA)](analysis/experiment_ledger.md)

A Kaggle code-competition solution for [ARC Prize 2026 – ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
that runs inside the competition limits (4× NVIDIA L4, 12 h, no internet). It combines two public
approaches and merges their answers per test output. The repository also keeps the full record of
how the solution evolved: every run, each pre-registered decision rule, the scores, and the post-mortem.

**Best Public LB: 30.56.** The medal zone was about 33 and first place was 88.06. No change moved
the score outside the run-to-run noise (about 3 points) of our NVARC-first baseline, and nothing
reached the 33.89 shown by the public NVARC notebook we started from. This is an honest record of
that, written up so the lessons can be reused.

## Approach

1. **NVARC** runs first on every task, shortest task first. NVARC is the 1st-place solution of
   ARC Prize 2025. A 4B grid model (`sorokin/qwen3_4b_grids15_sft139`) is fine-tuned per task at
   test time with LoRA TTT, then decoded with augmented DFS.
2. **Qwen3.8-27B** (FP8, vLLM 0.19, tensor parallel 4) uses the remaining time in one of two modes:
   - `direct`: long `xhigh` reasoning that writes the output grid itself. It draws two samples per
     task. When a reply is cut off, a no-thinking "commit now" rescue still gets an answer.
   - `repl`: a program-synthesis agent with `python` and `submit` tools. Its answers are verified
     against the training pairs.
3. **Merge**: NVARC's top-1 answer is always kept. For the second attempt the merge uses, in order
   of preference:
   - a REPL program that reproduces every training pair;
   - a Qwen answer that agrees with NVARC;
   - NVARC's top-2, when NVARC is confident;
   - otherwise, the Qwen answer.

## Results

Public LB scores. The same code scored 30.56 and 28.06 on two runs, so differences under about
3 points are noise.

| Version | Configuration | Public |
|---|---|---|
| S1 | Copy of the public NVARC notebook | 27.36 |
| S2 | NVARC 7 h (shortest first) + REPL agent | 24.44 |
| v6 | NVARC 10.5 h + REPL agent (two runs of the same code) | **30.56** / 28.06 |
| v7 (`sub-v7`) | v6 + shallow direct answers (~20k tokens/sample) | 30.14 |
| v8 (`sub-v8`) | v6 + forced-submit fixes | 29.72 |
| v9 (`sub-v9`) | v6 + deep direct answers (~100k tokens/sample) | pending |

On a fixed 24-task evaluation subset, deep direct answers solved 9/24 tasks (37.5%). NVARC alone
scores about 28–30%. The final configuration, however, gave the 27B model only the time NVARC
left over. The [write-up](docs/writeup.md) explains why the score stayed flat and what we would
change.

## Repository layout

```
notebooks/hybrid_v1/            Generator for the submission notebook (build.py) and its modules
  src/hybrid.py                 Orchestration: NVARC → vLLM → direct / REPL → merge
  src/direct.py                 Direct-answer mode
  src/repl_agent.py             REPL agent (python / submit tools)
  src/arc_*.py, starter.py      NVARC (TTT + DFS)
notebooks/frontier_copy_3389/   Copy of the public NVARC notebook (S1)
src/arc2/                       Data loading, metric, submission format
scripts/                        Local scoring, local MLX evaluation, log fetching, submit watcher, local CI
tests/                          Merge / agent / mode tests with a fake LLM (no GPU, no competition data)
analysis/                       Experiment ledger, research notes, retrospectives, loop runbook (Japanese)
docs/                           Write-up, competition overview, publication checklist
```

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), a Kaggle account that has accepted the competition
rules, and the `kaggle` CLI.

```bash
uv sync --group dev
./scripts/download_data.sh          # competition data into data/ (git-ignored, never committed)
uv run pytest -q && uv run ruff check .

# Build the submission notebook (kernel id defaults to $KAGGLE_USERNAME/arc2-hybrid-v1)
export KAGGLE_USERNAME=<you>
uv run python notebooks/hybrid_v1/build.py --out work/build \
  --set agent_mode='"direct"' --set max_model_len=131072 --set max_num_seqs=16 \
  --set save_mode='"minimal"'
kaggle kernels push -p work/build    # then submit the notebook from the competition page

# Agent-only evaluation kernel on the 120 public evaluation tasks
uv run python notebooks/hybrid_v1/build.py --out work/build_eval --id $KAGGLE_USERNAME/arc2-agent-eval \
  --set agent_eval_h=2.0 --set agent_mode='"direct"' --set max_model_len=131072 --set max_num_seqs=16

# Score a submission.json locally (evaluation split: 120 tasks / 172 outputs)
uv run python scripts/evaluate.py path/to/submission.json --split evaluation
```

`save_mode="minimal"` writes a valid fallback submission and exits in about a minute. The full
pipeline runs only in the competition rerun. The notebook attaches public Kaggle models and
datasets; they are listed in [NOTICE](NOTICE).

`scripts/fetch_log.py` and `scripts/watch_submit_hybrid.sh` call the Kaggle API, so run them
with the Python that has the `kaggle` package installed (the one the `kaggle` CLI uses), not with `uv run`.

For a cheap local comparison of agent modes, `scripts/local_eval.py` runs the agent against a
small Qwen served by `mlx_lm.server` on Apple Silicon. Set `ARC2_MLX_DIR` and `ARC2_SCRATCH`;
see the script header.

## Licence and credits

Apache-2.0 ([LICENSE](LICENSE)). This repository builds on NVARC (Sorokin & Puget), the public
Kaggle notebooks by koushikrudra and madarshbb, and Tufa Labs' Duck harness. [NOTICE](NOTICE)
lists every third-party source, model and runtime. Competition data is **not** included
(competition rule 2.4.b).
