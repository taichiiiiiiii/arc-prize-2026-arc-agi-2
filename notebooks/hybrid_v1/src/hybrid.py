"""Hybrid ARC-AGI-2 run: NVARC (Qwen3-4B TTT) first, then a Qwen3.8-27B REPL agent, merged.

    python hybrid.py '<json config>'

Phases (each guarded; a failing phase leaves the previous submission in place):
  0. background: unpack CPython 3.12 + install the vLLM wheelhouse (CPU, overlaps phase 1)
  1. NVARC on all tasks until cfg.nvarc_end  -> submission.json (NVARC only)
  2. vLLM server for Qwen3.8-27B FP8 on 4xL4 (option ladder)
  3. REPL agent (python + submit tools), tasks ordered by NVARC uncertainty
  4. merge per test output: agent-verified answer + NVARC top-1, else NVARC top-2

Outside a competition rerun this is a SMOKE run on a few evaluation tasks with short
budgets, which prints NVARC / agent / merged scores against the evaluation solutions.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
PY312 = WORK / "py312"
LOG = WORK / "vllm-server.log"
SUB = WORK / "submission.json"
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"


def log(*a):
    print(f"[hybrid {time.strftime('%H:%M:%S')}]", *a, flush=True)


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def write_atomic(obj, path=None):
    path = path or SUB
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def challenges_path(rerun):
    name = "arc-agi_test_challenges.json" if rerun else "arc-agi_evaluation_challenges.json"
    p = Path(COMP) / name
    return p if p.exists() else one(f"/kaggle/input/**/{name}")


# ---------------------------------------------------------------- phase 0: runtime
def py312_env():
    """Environment for the bundled CPython 3.12: none of the notebook's Python paths (the
    unsloth utility script on PYTHONPATH made pip try to uninstall packages there)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "PIP_", "VIRTUAL_ENV", "CONDA"))}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def prepare_runtime(cfg, state):
    try:
        t = time.time()
        hits = sorted(glob.glob(cfg["py312_glob"], recursive=True))
        parts = sorted(glob.glob(cfg.get("py312_parts_glob", "/nonexistent"), recursive=True))
        if not hits and parts:                 # split upstream archive: concat + sha256 check
            import hashlib
            tgz = WORK / "cpython312.tar.gz"
            h = hashlib.sha256()
            with open(tgz, "wb") as out:
                for p in parts:
                    data = Path(p).read_bytes()
                    h.update(data)
                    out.write(data)
            if h.hexdigest() != cfg["py312_sha256"]:
                raise RuntimeError(f"cpython archive sha256 mismatch: {h.hexdigest()}")
            hits = [str(tgz)]
        if hits:
            with tarfile.open(hits[0]) as tf:
                tf.extractall(WORK / "py312_tmp")
            shutil.move(str(WORK / "py312_tmp" / "python"), str(PY312))
        else:                                  # Kaggle may have unpacked the archive
            exe = one("/kaggle/input/**/python/bin/python3.12")
            shutil.copytree(exe.parent.parent, PY312, symlinks=True)
            for p in (PY312 / "bin").iterdir():
                p.chmod(0o755)
        py = PY312 / "bin" / "python3"
        wh = one("/kaggle/input/**/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock").parent
        subprocess.run([str(py), "-m", "pip", "install", "--no-index", "--find-links", str(wh),
                        "--requirement", str(wh / "requirements.lock"), "--only-binary", ":all:",
                        "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                       check=True, env=py312_env())
        if cfg.get("overlay"):
            wheels = sorted(glob.glob(cfg["overlay"], recursive=True))
            if wheels:
                subprocess.run([str(py), "-m", "pip", "install", "--no-index", "--no-deps",
                                "--upgrade", "--force-reinstall", "--no-compile",
                                "--disable-pip-version-check", "-q", *wheels], check=True, env=py312_env())
        subprocess.run([str(py), "-c", "import vllm, torch; print('vllm', vllm.__version__, 'torch', "
                        "torch.__version__, 'cuda', torch.cuda.is_available())"],
                       check=True, env=py312_env())
        state["py"] = str(py)
        log(f"runtime ready in {time.time() - t:.0f}s: {py}")
    except BaseException:
        state["error"] = traceback.format_exc()
        log("runtime FAILED:\n" + state["error"])


# ---------------------------------------------------------------- phase 1: NVARC
def run_nvarc(cfg, keys, chal_path, end_time, gpus="0,1,2,3"):
    kf = WORK / "nvarc_keys.json"
    kf.write_text(json.dumps(keys))
    env = dict(os.environ, UNSLOTH_DISABLE_STATISTICS="1",
               TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas", OMP_NUM_THREADS="12",
               PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, str(HERE / "starter.py"), "--end-time", str(end_time),
           "--keys", str(kf), "--challenges", str(chal_path), "--gpus", gpus]
    log("NVARC:", " ".join(cmd))
    t = time.time()
    proc = None
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(HERE), start_new_session=True)
        proc.wait(timeout=max(60, end_time - time.time() + cfg.get("nvarc_grace_s", 900)))
    except subprocess.TimeoutExpired:
        log("NVARC overran its grace period -> killed")
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    log(f"NVARC finished in {(time.time() - t) / 60:.1f} min")


def decode_nvarc(chal_path, sol_path=None, store="/kaggle/inference_outputs"):
    """Ranked NVARC candidates per test output + a confidence in [0, 1]."""
    sys.path.insert(0, str(HERE))
    from arc_decoder import ArcDecoder
    from arc_loader import ArcDataset
    data = ArcDataset.from_file(str(chal_path))
    if sol_path:
        data = data.load_replies(str(sol_path))
    dec = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    if os.path.isdir(store):
        dec.load_decoded_results(store)
    ranked = dec.run_selection_algo()            # basekey "tid_j" -> [np grids], best first
    sub = data.get_submission(ranked)
    meta = {}
    for bk, samples in dec.decoded_results.items():
        sols = [json.dumps(s["solution"].tolist()) for s in samples.values()]
        top = json.dumps(ranked[bk][0].tolist()) if ranked.get(bk) else None
        meta[bk] = {"n": len(sols), "conf": (sols.count(top) / len(sols)) if sols else 0.0,
                    "cands": [g.tolist() for g in ranked.get(bk, [])[:2]]}
    return sub, meta


# ---------------------------------------------------------------- phase 2: vLLM
def gpus_free(limit_mb=1500, wait_s=180, which=None):
    t = time.time()
    out = []
    while time.time() - t < wait_s:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        used = [int(u) for u in out if u.strip().isdigit()]
        if which is not None:
            used = [u for i, u in enumerate(used) if i in which]
        if used and all(u < limit_mb for u in used):
            return True
        time.sleep(5)
    log("GPUs still busy:", out)
    return False


def kill_server(proc, which=None):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=60)
    except Exception:
        pass
    gpus_free(which=which)


def start_server(cfg, py, rung, deadline):
    model = one(cfg["model_glob"]).parent
    env = dict(py312_env(), USE_TF="0", TRANSFORMERS_NO_TF="1", TRANSFORMERS_NO_TORCHVISION="1",
               VLLM_NO_USAGE_STATS="1",
               LIBRARY_PATH="/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", ""))
    env.update(rung.get("env", {}))
    tp = cfg.get("tp", 4)
    if cfg.get("vllm_gpus"):
        env["CUDA_VISIBLE_DEVICES"] = cfg["vllm_gpus"]
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
           "--served-model-name", "qwen", "--host", "127.0.0.1", "--port", "1234",
           "--tensor-parallel-size", str(tp), "--enable-auto-tool-choice", *cfg["parsers"],
           "--generation-config", "vllm", "--enable-prefix-caching",
           "--max-model-len", str(cfg["max_model_len"]), "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", str(cfg["max_num_seqs"]), *rung.get("args", [])]
    log(" ".join(cmd), "env", rung.get("env", {}))
    t = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=str(WORK))
    limit = min(t + cfg.get("server_start_s", 1500), deadline)
    while time.time() < limit:
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        if proc.poll() is not None or "EngineCore failed" in text:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
            log(f"server ready after {time.time() - t:.0f}s")
            return proc
        except Exception:
            time.sleep(10)
    log("server FAILED; tail:\n" + "\n".join(LOG.read_text(errors="replace").splitlines()[-40:]))
    kill_server(proc, which=vllm_gpu_ids(cfg))
    return None


def vllm_gpu_ids(cfg):
    return {int(g) for g in cfg["vllm_gpus"].split(",")} if cfg.get("vllm_gpus") else None


# ---------------------------------------------------------------- phase 3/4: agent + merge
def agent_ranked(records, task):
    """Per test input: (verified grids, other grids), each best first (run_submit.ranked_answers)."""
    out = []
    ntr = len(task["train"])
    for j in range(len(task["test"])):
        score, grids = {}, {}
        for r in records:
            for p in r["programs"]:
                g = p["test"][j] if j < len(p["test"]) else None
                if g is None:
                    continue
                k = json.dumps(g)
                grids[k] = g
                s = score.get(k, (0, 0, 0))
                score[k] = (s[0] + (p["n_ok"] == ntr), max(s[1], p["n_ok"]), s[2] + 1)
        order = sorted(score, key=lambda k: score[k], reverse=True)
        out.append(([grids[k] for k in order if score[k][0] > 0],
                    [grids[k] for k in order if score[k][0] == 0]))
    return out


def merge_task(task, nv_rows, agent_rows):
    """nv_rows[j] = {"attempt_1", "attempt_2"} from NVARC (or [[0]] when missing)."""
    rows = []
    for j, x in enumerate(task["test"]):
        nv = [g for g in (nv_rows[j]["attempt_1"], nv_rows[j]["attempt_2"]) if g != [[0]]]
        ver, unver = agent_rows[j] if agent_rows else ([], [])
        cands = []
        for g in ver[:1] + nv + ver[1:] + unver + [x["input"]]:
            if g not in cands:
                cands.append(g)
        rows.append({"attempt_1": cands[0], "attempt_2": cands[1] if len(cands) > 1 else cands[0]})
    return rows


def run_agent(cfg, tasks, order, nv_sub, deadline, results):
    sys.path.insert(0, str(HERE))
    from client import ChatClient
    from repl_agent import AGENT_OPTS, solve_repl
    client = ChatClient("http://127.0.0.1:1234/v1", "qwen", max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=True,
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))
    n = len(order)
    avail = max(0.0, deadline - time.time())
    cap = int(min(cfg.get("max_cap", 60000),
                  max(cfg.get("min_cap", 15000), cfg["tok_per_s"] * avail / max(n, 1) * 1.3)))
    log(f"agent: {n} tasks, {avail / 3600:.2f} h, per-attempt cap {cap} tokens")
    lock = threading.Lock()
    sub = json.load(open(SUB))
    alog = open(WORK / "agent_attempts.jsonl", "a")

    def on_done(rec):
        with lock:
            tid = rec["task_id"]
            results.setdefault(tid, []).append(rec)
            sub[tid] = merge_task(tasks[tid], nv_sub[tid], agent_ranked(results[tid], tasks[tid]))
            write_atomic(sub)
            slim = {k: v for k, v in rec.items() if k not in ("transcript", "programs")}
            slim["programs"] = [{"n_ok": p["n_ok"], "n_train": p["n_train"]} for p in rec["programs"]]
            alog.write(json.dumps(slim) + "\n")
            alog.flush()

    solve_repl({t: tasks[t] for t in order}, client, attempts=cfg.get("attempts", 1),
               concurrency=cfg.get("concurrency", 24), deadline=deadline, stop_after=1,
               max_turns=cfg.get("max_turns", 40), max_gen_tokens=cap,
               context_budget=cfg.get("context_budget", 24000), nudge_every=cfg.get("nudge_every", 6),
               log=log, on_done=on_done, **{k: cfg[k] for k in AGENT_OPTS if k in cfg})
    alog.close()


def eval_report(tasks, results, agent_sub, sols):
    """Per-task agent outcome on labelled tasks: attempted / verified / correct / false-verified."""
    rows = {}
    for t in tasks:
        recs = results.get(t, [])
        ver = any(r["status"] == "verified" for r in recs)
        ok, tot = score({t: agent_sub[t]}, {t: sols[t]})
        rows[t] = {"attempted": bool(recs), "verified": ver, "correct": ok == tot,
                   "tokens": sum(r.get("gen_tokens", 0) for r in recs)}
    v = [r for r in rows.values() if r["verified"]]
    summary = {"tasks": len(rows), "attempted": sum(r["attempted"] for r in rows.values()),
               "verified": len(v), "correct": sum(r["correct"] for r in rows.values()),
               "false_verified": sum(not r["correct"] for r in v),
               "outputs": score(agent_sub, sols)}
    return {"summary": summary, "tasks": rows}


def score(sub, sols):
    tot = ok = 0
    for t, truths in sols.items():
        for j, truth in enumerate(truths):
            tot += 1
            r = sub.get(t, [])
            if j < len(r) and truth in (r[j]["attempt_1"], r[j]["attempt_2"]):
                ok += 1
    return ok, tot


def start_server_ladder(cfg, state, agent_end):
    for i, rung in enumerate(cfg["server_ladder"]):
        if agent_end - time.time() < 45 * 60:
            break
        proc = start_server(cfg, state["py"], rung, agent_end - 30 * 60)
        if proc is not None:
            log(f"server up with rung {i}")
            return proc
    raise RuntimeError("no server")


def main(cfg):
    start = float((WORK / "start_epoch.txt").read_text())
    rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    split = cfg.get("mode") == "split"          # NVARC on GPUs 0,1 || vLLM TP2 on GPUs 2,3
    hard_end = start + 12 * 3600 - cfg.get("reserve_s", 30 * 60)
    chal_path = challenges_path(rerun)
    ch = json.load(open(chal_path))
    if rerun:
        keys = sorted(ch)
        if split:
            nvarc_end = hard_end - 25 * 60
        else:
            nvarc_end = min(start + cfg["nvarc_hours"] * 3600,
                            hard_end - cfg.get("agent_reserve_h", 3.0) * 3600)
        agent_end = hard_end
    elif cfg.get("agent_eval_h"):            # agent-only evaluation on the evaluation set
        keys = sorted(ch) if cfg.get("agent_eval_tasks", "all") == "all" else cfg["agent_eval_tasks"]
        nvarc_end = time.time()
        agent_end = time.time() + cfg["agent_eval_h"] * 3600 + 15 * 60
    else:
        keys = cfg["smoke_tasks"]
        nvarc_end = time.time() + cfg["smoke_nvarc_min"] * 60
        agent_end = (nvarc_end if not split else time.time()) + cfg["smoke_agent_min"] * 60 + 25 * 60
    tasks = {k: ch[k] for k in keys}
    by_len = sorted(keys, key=lambda t: len(json.dumps(tasks[t])))
    log(f"rerun={rerun} split={split} tasks={len(tasks)} nvarc_end in "
        f"{(nvarc_end - time.time()) / 60:.0f} min, agent_end in {(agent_end - time.time()) / 60:.0f} min")

    state = {}
    rt = threading.Thread(target=prepare_runtime, args=(cfg, state), daemon=True)
    rt.start()
    empty = {t: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[t]["test"]] for t in keys}
    nv = {"sub": empty, "meta": {}}

    def nvarc_phase():
        try:
            run_nvarc(cfg, by_len, chal_path, nvarc_end, gpus=cfg.get("nvarc_gpus", "0,1,2,3"))
            full_sub, meta = decode_nvarc(chal_path)
            nv["sub"], nv["meta"] = {t: full_sub[t] for t in keys}, meta
            (WORK / "nvarc_meta.json").write_text(json.dumps(meta))
        except BaseException:
            log("NVARC phase failed:\n" + traceback.format_exc())
        write_atomic(nv["sub"], WORK / "nvarc_submission.json")
        log(f"NVARC decoded {len(nv['meta'])} test outputs")

    def merge_all(results):
        sub = json.load(open(SUB))
        for t in keys:
            sub[t] = merge_task(tasks[t], nv["sub"][t],
                                agent_ranked(results[t], tasks[t]) if results.get(t) else None)
        write_atomic(sub)

    results = {}
    proc = None
    nv_thread = None
    if cfg.get("agent_eval_h") and not rerun:
        merge_all(results)
    elif split:
        nv_thread = threading.Thread(target=nvarc_phase, daemon=True)
        nv_thread.start()
    else:
        nvarc_phase()
        merge_all(results)
    try:
        if cfg.get("skip_agent"):              # NVARC-only candidate: no vLLM at all
            raise RuntimeError("skip_agent set")
        rt.join(timeout=max(0, agent_end - time.time() - 45 * 60))
        if "py" not in state:
            raise RuntimeError("runtime not ready: " + state.get("error", "timeout")[-500:])
        if not split:
            gpus_free()
        proc = start_server_ladder(cfg, state, agent_end)

        def conf(t):
            c = [nv["meta"].get(f"{t}_{j}", {}).get("conf", 0.0) for j in range(len(tasks[t]["test"]))]
            return min(c) if c else 0.0
        hi = cfg.get("confident_nvarc", 0.75)
        order = by_len if split else sorted(keys, key=lambda t: (conf(t) >= hi, len(json.dumps(tasks[t]))))
        run_agent(cfg, tasks, order, nv["sub"], agent_end - 5 * 60, results)
    except BaseException:
        log("agent phase failed:\n" + traceback.format_exc())
    finally:
        if proc is not None:
            kill_server(proc, which=vllm_gpu_ids(cfg))
    if nv_thread is not None:
        nv_thread.join(timeout=max(0, hard_end - time.time()))
    merge_all(results)
    # keep /kaggle/working small: thousands of runtime files make the output listing API 429
    for p in (PY312, WORK / "py312_tmp", WORK / "cpython312.tar.gz"):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    ver = sum(1 for t in results if any(r["status"] == "verified" for r in results[t]))
    log(f"agent verified {ver}/{len(keys)} tasks")
    if not rerun:
        sols = json.load(open(one("/kaggle/input/**/arc-agi_evaluation_solutions.json")))
        sols = {k: sols[k] for k in keys}
        final = json.load(open(SUB))
        agent_only = {t: merge_task(tasks[t], empty[t], agent_ranked(results.get(t, []), tasks[t]))
                      for t in keys}
        for name, s in (("nvarc", nv["sub"]), ("agent", agent_only), ("merged", final)):
            log(f"SMOKE score {name}: {score(s, sols)}")
        rep = eval_report(tasks, results, agent_only, sols)
        (WORK / "agent_eval.json").write_text(json.dumps(rep))
        log("EVAL " + json.dumps(rep["summary"]))


if __name__ == "__main__":
    try:
        main(json.loads(sys.argv[1]))
    except BaseException:
        log("hybrid main crashed:\n" + traceback.format_exc())
    sys.exit(0)
