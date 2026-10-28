"""Local agent evaluation on the evaluation set against an OpenAI-compatible server
(mlx_lm.server on the Mac with a small Qwen model).

Compares *harness* settings (force-submit rules, budgets, prompts) cheaply. The model is not the
27B used on Kaggle, so only paired differences against a local baseline are meaningful, and an
adopted change is still confirmed with the Kaggle agent-eval kernel.

    uv run python scripts/local_eval.py --name base --tasks 40 --hours 2
    uv run python scripts/local_eval.py --name fst0 --tasks 40 --hours 2 --set force_submit_turns=0

Results: work/local_eval/<name>.json (summary + per task) and one line in
work/local_eval/runs.jsonl.
"""
import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "hybrid_v1" / "src"
BUILD = ROOT / "notebooks" / "hybrid_v1" / "build.py"
sys.path.insert(0, str(SRC))

import hybrid
from client import ChatClient
from repl_agent import AGENT_OPTS, solve_repl

LOCAL_DEFAULTS = {          # small local model: smaller turns, fewer parallel requests
    "concurrency": 8, "turn_max_tokens": 4096, "max_cap": 16000, "context_budget": 16000,
    "max_turns": 20,
}


def load_config():
    spec = importlib.util.spec_from_file_location("build", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG)


def pick_tasks(ch, n):
    """The n shortest evaluation tasks (deterministic; small models only solve short ones)."""
    return sorted(ch, key=lambda t: (len(json.dumps(ch[t])), t))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--model", default="default_model")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    a = ap.parse_args()

    cfg = {**load_config(), **LOCAL_DEFAULTS}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    ch = json.loads((ROOT / "data" / "arc-agi_evaluation_challenges.json").read_text())
    sols = json.loads((ROOT / "data" / "arc-agi_evaluation_solutions.json").read_text())
    keys = pick_tasks(ch, a.tasks)
    tasks = {k: ch[k] for k in keys}

    client = ChatClient(a.url, a.model, max_tokens=cfg["turn_max_tokens"],
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=True,
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))
    out = ROOT / "work" / "local_eval"
    out.mkdir(parents=True, exist_ok=True)
    results, lock = {}, threading.Lock()
    alog = open(out / f"{a.name}.attempts.jsonl", "w")  # noqa: SIM115 (closed after solve_repl)

    def on_done(rec):
        with lock:
            results.setdefault(rec["task_id"], []).append(rec)
            slim = {k: v for k, v in rec.items() if k not in ("transcript", "programs")}
            alog.write(json.dumps(slim) + "\n")
            alog.flush()

    t0 = time.time()
    solve_repl(tasks, client, attempts=cfg.get("attempts", 1), concurrency=cfg["concurrency"],
               deadline=t0 + a.hours * 3600, stop_after=1, max_turns=cfg["max_turns"],
               max_gen_tokens=cfg["max_cap"], context_budget=cfg["context_budget"],
               nudge_every=cfg.get("nudge_every", 6), on_done=on_done,
               **{k: cfg[k] for k in AGENT_OPTS if k in cfg})
    alog.close()
    empty = {t: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[t]["test"]] for t in keys}
    agent_sub = {t: hybrid.merge_task(tasks[t], empty[t], hybrid.agent_ranked(results.get(t, []), tasks[t]))
                 for t in keys}
    rep = hybrid.eval_report(tasks, results, agent_sub, {t: sols[t] for t in keys})
    rep["summary"].update(name=a.name, overrides=a.set, hours=round((time.time() - t0) / 3600, 2),
                          url=a.url, finished=sum(1 for t in keys for r in results.get(t, [])
                                                  if r["status"] != "stopped"))
    (out / f"{a.name}.json").write_text(json.dumps(rep, indent=1))
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(rep["summary"]) + "\n")
    print("LOCAL_EVAL " + json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
