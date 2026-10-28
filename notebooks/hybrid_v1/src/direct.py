"""Direct-answer mode: Qwen3.8 reasons at length (reasoning_effort xhigh) and writes the test
output grid(s) itself, no tools. Several samples per task; the hybrid merge picks two answers.

Adapted from madarshbb/arc-agent-sub-q38-direct-b98k-c16 `run_direct.py` (Apache 2.0). This is
how headline ARC-AGI-2 numbers for chat models are measured (ARC Prize verified Qwen3.8-27B at
42.4%), and the REPL agent with short thinking turns scored far below it in our evaluation.
"""
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from grids import show

DIRECT_SYSTEM = (
    "You are an expert at solving ARC-AGI puzzles. Each puzzle has a few input->output "
    "example pairs that share one hidden transformation rule. Grids are 2D arrays of digits "
    "0-9 (colours); 0 is usually the background."
)

DIRECT_INSTRUCTIONS = """Solve this ARC puzzle. The examples share one hidden transformation rule. Work out the rule, check it against every example, then apply it to the test input(s).

At the very end of your answer, give each test output grid as rows of digits inside its own ```grid code block, in test order, with nothing else in the block."""

# A reply cut off before any grid gets ONE short follow-up without thinking: the tail of its own
# reasoning and an order to commit. An unanswered task scores nothing; a committed guess may.
RESCUE = """{prompt}

## Your notes so far (they were cut off)
{notes}

Stop analysing now. Apply the rule you were converging on and give the final test output grid(s): each as rows of digits in its own ```grid code block, in test order, with nothing else in the block."""


def direct_prompt(task):
    parts = [DIRECT_INSTRUCTIONS, ""]
    for i, p in enumerate(task["train"], 1):
        parts += [f"## Example {i}", show(p["input"], "Input"), show(p["output"], "Output"), ""]
    for j, t in enumerate(task["test"], 1):
        parts += [f"## Test input {j}", show(t["input"], "Input"), ""]
    return "\n".join(parts).rstrip()


def parse_grids(text, n_tests):
    """The last n_tests digit-only code blocks (```grid preferred, any fence accepted)."""
    blocks = re.findall(r"```(?:grid)?[^\n]*\n(.*?)```", text or "", re.S)
    grids = []
    for b in blocks:
        rows = [r.strip().replace(" ", "").replace(",", "") for r in b.strip().splitlines() if r.strip()]
        if (rows and all(r.isdigit() for r in rows) and len({len(r) for r in rows}) == 1
                and len(rows) <= 30 and len(rows[0]) <= 30):
            grids.append([[int(c) for c in r] for r in rows])
    return grids[-n_tests:] if len(grids) >= n_tests else None


def answer(client, task, seed, deadline, max_tokens=None, rescue_tokens=8192, notes_chars=16000):
    """One direct answer. rec["grids"] is the list of test grids, or None."""
    n = len(task["test"])
    prompt = direct_prompt(task)

    def timeout():
        return max(60.0, min(4 * 3600, deadline - time.time() + 60))

    mt = max_tokens or client.max_tokens
    for _ in range(3):
        r = client.chat([{"role": "system", "content": DIRECT_SYSTEM},
                         {"role": "user", "content": prompt}],
                        seed=seed, max_tokens=mt, timeout=timeout())
        if not (r.get("context_overflow") and mt > 8192):
            break
        mt = int(mt * 0.75)                    # a long prompt: shrink the reply, never give up
    if "error" in r:
        return {"status": "error: " + r["error"][:200], "grids": None, "gen_tokens": 0}
    rec = {"finish": r["finish_reason"], "gen_tokens": r["completion_tokens"]}
    grids = parse_grids(r["content"], n)
    if grids:
        return dict(rec, status="answered", grids=grids)
    notes = ((r["reasoning"] or "") + "\n" + (r["content"] or "")).strip()
    if notes and deadline - time.time() > 120:
        r2 = client.chat([{"role": "system", "content": DIRECT_SYSTEM},
                          {"role": "user", "content": RESCUE.format(prompt=prompt,
                                                                    notes=notes[-notes_chars:])}],
                         seed=seed + 1, max_tokens=rescue_tokens, timeout=timeout(),
                         template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
        if "error" not in r2:
            rec["gen_tokens"] += r2["completion_tokens"]
            # Without thinking, a reasoning parser may still file the reply as reasoning.
            grids = parse_grids(r2["content"], n) or parse_grids(r2["reasoning"], n)
            if grids:
                return dict(rec, status="rescued", grids=grids)
    return dict(rec, status="no grid", grids=None)


def run_direct(client, tasks, order, deadline, samples=2, concurrency=16, on_done=None, log=print,
               rescue_tokens=8192):
    """Sample-major: every task in `order` gets sample 0 before any gets sample 1, so a short
    run covers the most tasks. Returns {task_id: [record, ...]}."""
    results, lock, t0 = {}, threading.Lock(), time.time()
    jobs = [(t, k) for k in range(samples) for t in order]

    def work(job):
        t, k = job
        if time.time() > deadline - 120:
            return
        ts = time.time()
        rec = {"task_id": t, "sample": k,
               **answer(client, tasks[t], 1000 * k + 7, deadline, rescue_tokens=rescue_tokens)}
        rec["wall"] = round(time.time() - ts, 1)
        with lock:
            results.setdefault(t, []).append(rec)
            if on_done:
                on_done(rec)
            log(f"[{time.time() - t0:6.0f}s] direct {t}#{k} {rec['status']:9s} "
                f"tok {rec.get('gen_tokens', 0):6d} finish {rec.get('finish')}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work, jobs))
    return results


def direct_candidates(records, j):
    """Grids for test output j from the direct samples, most frequent first."""
    c = Counter()
    first = {}
    for r in records or []:
        g = r.get("grids")
        if g and j < len(g):
            key = repr(g[j])
            c[key] += 1
            first.setdefault(key, g[j])
    return [first[k] for k, _ in c.most_common()], c
