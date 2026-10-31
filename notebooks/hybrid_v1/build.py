"""Build the hybrid kernel (ipynb + kernel-metadata.json) from src/.

    uv run python notebooks/hybrid_v1/build.py --out <dir> [--id taichiiiii/arc2-hybrid-v1]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
FILES = ["arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py",
         "grids.py", "sandbox.py", "backend.py", "agent.py", "repl.py", "client.py",
         "repl_agent.py", "direct.py", "hybrid.py"]

ATTN_ENV = {"VLLM_FORCE_ATTN_BACKEND": "TRITON_ATTN"}
REASON = ["--reasoning-config", json.dumps({"reasoning_start_str": "<think>",
                                            "reasoning_end_str": "</think>"})]
MTP2 = ["--speculative-config", json.dumps({"method": "qwen3_5_mtp", "num_speculative_tokens": 2})]
CONFIG = {
    "py312_glob": "/kaggle/input/**/cpython-3.12*-x86_64-linux-install_only.tar.gz",
    # public mirror of the upstream python-build-standalone archive, split in 7 parts;
    # sha256 = upstream release 20260814 asset digest (verified)
    "py312_parts_glob": "/kaggle/input/**/arc3-cpython312-runtime/**/cpython-3.12.14-part-0*.bin",
    "py312_sha256": "3297691ae34f75fed81ac424e040145fccb0bafe8e581cd5cadbddfa1c0766c0",
    "model_glob": "/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json",
    "overlay": "/kaggle/input/**/taaf-duck-qwen38-serving-v1/serving_wheels/*.whl",
    "parsers": ["--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3"],
    # most ambitious first; the last rung is plain serving
    "server_ladder": [
        {"args": REASON + ["--kv-cache-dtype", "fp8", "--language-model-only",
                           "--attention-backend", "TRITON_ATTN"] + MTP2, "env": ATTN_ENV},
        {"args": REASON + ["--kv-cache-dtype", "fp8", "--language-model-only"] + MTP2},
        {"args": REASON + ["--kv-cache-dtype", "fp8"]},
        {"args": REASON},
        {"args": []},
    ],
    "max_model_len": 131072,  # direct needs long reasoning (v7)
    "max_num_seqs": 16,
    "server_start_s": 1500,
    "nvarc_hours": 10.5,
    "agent_reserve_h": 1.0,   # S2 (NVARC 7h) 24.44 < S1 (NVARC-only) 27.36: NVARC first, agent gets leftovers
    "nvarc_grace_s": 900,
    "reserve_s": 30 * 60,
    "confident_nvarc": 0.75,
    # agent (repl_agent.run_attempt options)
    "attempts": 1,
    "concurrency": 32,
    "max_turns": 40,
    "context_budget": 24000,
    "nudge_every": 6,
    "turn_max_tokens": 8192,
    "thinking_token_budget": 6144,
    "force_submit_after": 5,
    "force_submit_turns": 4,
    "in_python": True,
    "effort_ladder": ["low"],
    "timeout_as_cut": True,
    "degenerate_guard": True,
    "template_kwargs": {"reasoning_effort": "xhigh"},
    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    "tok_per_s": 800, "min_cap": 15000, "max_cap": 50000,
    # direct-answer mode (agent_mode "direct"): long xhigh reasoning, grids written by the model.
    # Needs max_model_len 131072 and max_num_seqs ~16 (madarshbb direct-b98k-c16 settings).
    "agent_mode": "direct",  # v7: REPL 6/120 on eval; top public runs answer directly
    # "minimal": the save run skips everything (about a minute of GPU quota); "smoke" runs the
    # short evaluation smoke test (about 2 h of quota).
    "save_mode": "minimal",
    "direct_samples": 2, "direct_concurrency": 16, "direct_max_tokens": 106496,
    "direct_thinking_budget": 98304, "direct_temperature": 1.0,
    "direct_template_kwargs": {"reasoning_effort": "xhigh"}, "rescue_tokens": 8192,
    # per-sample cap = direct_tok_per_s * time left / tasks to do, clamped to [min, max];
    # tasks with NVARC conf >= direct_skip_conf get no direct samples.
    "direct_tok_per_s": 350, "direct_min_tokens": 16384, "direct_skip_conf": 0.75,
    # save-run smoke test on evaluation tasks
    "smoke_tasks": ["e8686506", "20270e3b", "28a6681f", "dbff022c",
                    "7b5033c1", "78332cb0", "36a08778", "aa4ec2a5"],
    "smoke_nvarc_min": 15,
    "smoke_agent_min": 25,
}

HEAD = """# ARC-AGI-2 hybrid: NVARC (Qwen3-4B TTT) + Qwen3.8-27B REPL agent

* NVARC pipeline from `koushikrudra/failed-in-aimo` (Apache 2.0), itself built on NVARC (ARC Prize 2025).
* REPL agent (python + submit tools) from `madarshbb/arc-qwen-3-8-27b-agent` (Apache 2.0),
  modelled on the Tufa Labs duck harness.
* Orchestration and merge: `hybrid.py`. Outside a competition rerun the notebook runs a short smoke
  test on evaluation tasks and prints the scores."""

FALLBACK = """# A valid submission exists before anything heavy runs.
import glob, json, os, time
from pathlib import Path
START = time.time()
Path("/kaggle/working/start_epoch.txt").write_text(str(START))
os.makedirs("/kaggle/working/hybrid", exist_ok=True)
try:
    ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
    json.dump({t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in task["test"]]
               for t, task in ch.items()}, open("/kaggle/working/submission.json", "w"))
    print("fallback written for", len(ch), "tasks; rerun =", os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
except Exception as e:
    print("fallback FAILED:", repr(e))"""

VALIDATE = """# Last line of defence: malformed rows are replaced by the fallback.
import glob, json
try:
    ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
    sub = json.load(open("/kaggle/working/submission.json"))
    def ok_grid(g):
        return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30
                and len(r) == len(g[0]) for r in g)
                and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))
    bad = [t for t, task in ch.items()
           if t not in sub or len(sub[t]) != len(task["test"])
           or not all(ok_grid(r.get("attempt_1")) and ok_grid(r.get("attempt_2")) for r in sub[t])]
    for t in bad:
        sub[t] = [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in ch[t]["test"]]
    sub = {t: sub[t] for t in ch}
    json.dump(sub, open("/kaggle/working/submission.json", "w"))
    print(f"submission.json: {len(sub)} tasks, {len(bad)} rows replaced by fallback")
except Exception as e:
    print("validation error:", repr(e))"""


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(out: Path, kernel_id: str):
    cells = [cell(HEAD, "markdown"), cell(FALLBACK)]
    for name in FILES:
        cells.append(cell(f"%%writefile /kaggle/working/hybrid/{name}\n" + (SRC / name).read_text()))
    run = ("import json, subprocess, sys\n"
           f"cfg = json.loads(r'''{json.dumps(CONFIG)}''')\n"
           "rc = subprocess.run([sys.executable, '/kaggle/working/hybrid/hybrid.py', json.dumps(cfg)],"
           " cwd='/kaggle/working/hybrid').returncode\n"
           "print('hybrid exit code', rc)")
    cells += [cell(run), cell(VALIDATE)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    out.mkdir(parents=True, exist_ok=True)
    slug = kernel_id.split("/")[1]
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": kernel_id, "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3",
                            "keithtyser/taaf-duck-qwen38-serving-v1",
                            "destinyfire1/arc3-cpython312-runtime"],
        "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                          "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
        # same image as the scored NVARC notebook (Python 3.11); vLLM runs on bundled CPython 3.12
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                        "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    for c in cells:                       # every code cell must at least parse
        src = "".join(c["source"])
        if c["cell_type"] == "code" and not src.startswith("%%"):
            compile(src, "<cell>", "exec")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", default="taichiiiii/arc2-hybrid-v1")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a CONFIG entry, e.g. --set agent_eval_h=2.0")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in CONFIG or k in ("agent_eval_h", "agent_eval_tasks", "skip_agent"), k
        CONFIG[k] = json.loads(v)
    print(build(Path(a.out), a.id))
