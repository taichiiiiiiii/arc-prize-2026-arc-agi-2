"""Local checks of the hybrid kernel's merge logic and the REPL agent loop (no GPU, fake LLM)."""
import json
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks" / "hybrid_v1" / "src"
sys.path.insert(0, str(SRC))

import hybrid

TASK = {"train": [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 3]], "output": [[3, 2]]}],
        "test": [{"input": [[4, 5]]}]}
FLIP = "def transform(grid):\n    return [row[::-1] for row in grid]\n"
WRONG = "def transform(grid):\n    return grid\n"


def rec(*programs):
    return {"programs": [{"n_ok": n, "n_train": 2, "test": [g]} for n, g in programs]}


def test_agent_ranked_splits_verified():
    ranked = hybrid.agent_ranked([rec((2, [[5, 4]]), (1, [[4, 5]]))], TASK)
    assert ranked == [([[[5, 4]]], [[[4, 5]]])]


def test_merge_verified_first_then_nvarc_top1():
    nv = [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    rows = hybrid.merge_task(TASK, nv, hybrid.agent_ranked([rec((2, [[5, 4]]))], TASK))
    assert rows == [{"attempt_1": [[5, 4]], "attempt_2": [[1]]}]


def test_merge_without_agent_keeps_nvarc():
    nv = [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    assert hybrid.merge_task(TASK, nv, None) == nv
    # unverified agent output never displaces NVARC
    rows = hybrid.merge_task(TASK, nv, hybrid.agent_ranked([rec((1, [[9]]))], TASK))
    assert rows == nv


def test_merge_fills_missing_nvarc():
    nv = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    rows = hybrid.merge_task(TASK, nv, hybrid.agent_ranked([rec((1, [[9]]))], TASK))
    assert rows == [{"attempt_1": [[9]], "attempt_2": [[4, 5]]}]
    assert hybrid.merge_task(TASK, nv, None) == [{"attempt_1": [[4, 5]], "attempt_2": [[4, 5]]}]


def test_merge_dedupes_agent_equal_to_nvarc():
    nv = [{"attempt_1": [[5, 4]], "attempt_2": [[2]]}]
    rows = hybrid.merge_task(TASK, nv, hybrid.agent_ranked([rec((2, [[5, 4]]))], TASK))
    assert rows == [{"attempt_1": [[5, 4]], "attempt_2": [[2]]}]


class FakeClient:
    """Turn 1: a python call; turn 2: a wrong submit; turn 3: the right submit."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **kw):
        self.calls += 1
        n = sum(1 for m in messages if m["role"] == "assistant")
        name, code = [("python", "print(train[0])"), ("submit", WRONG), ("submit", FLIP)][min(n, 2)]
        return {"content": "", "reasoning": "r", "finish_reason": "tool_calls",
                "tool_calls": [{"id": f"c{n}", "name": name, "arguments": {"code": code}}],
                "prompt_tokens": 10, "completion_tokens": 5}


def test_repl_agent_end_to_end_with_fake_client():
    pytest.importorskip("numpy")
    from repl_agent import solve_repl
    got = []
    _, report = solve_repl({"t1": TASK}, FakeClient(), attempts=1, concurrency=1,
                           on_done=got.append, max_turns=6, max_gen_tokens=1000,
                           context_budget=8000, nudge_every=6, in_python=True)
    assert report["attempts"][0]["status"] == "verified"
    rows = hybrid.merge_task(TASK, [{"attempt_1": [[0]], "attempt_2": [[0]]}],
                             hybrid.agent_ranked(got, TASK))
    assert rows[0]["attempt_1"] == [[5, 4]]
    json.dumps(got)  # records must be serialisable for the attempts log


def test_force_submit_after_turns():
    from repl_agent import solve_repl

    class Thinker:
        def __init__(self):
            self.choices = []

        def chat(self, messages, tools=None, tool_choice=None, **kw):
            self.choices.append(tool_choice)
            if tool_choice == "submit":
                return {"content": "", "reasoning": "", "finish_reason": "tool_calls",
                        "tool_calls": [{"id": "s", "name": "submit", "arguments": {"code": FLIP}}],
                        "prompt_tokens": 1, "completion_tokens": 1}
            return {"content": "", "reasoning": "hmm", "finish_reason": "length", "tool_calls": [],
                    "prompt_tokens": 1, "completion_tokens": 1}

    c = Thinker()
    _, report = solve_repl({"t1": TASK}, c, attempts=1, concurrency=1, max_turns=10,
                           max_gen_tokens=1000, context_budget=8000, nudge_every=6,
                           force_submit_turns=3)
    assert c.choices == [None, None, None, "submit"]
    assert report["attempts"][0]["status"] == "verified"


def _fake_main_env(tmp_path, monkeypatch, split, **extra):
    work = tmp_path / "w"
    work.mkdir()
    ch = {"t1": TASK, "t2": TASK}
    (tmp_path / "ch.json").write_text(json.dumps(ch))
    (work / "start_epoch.txt").write_text("0")
    (work / "submission.json").write_text(
        json.dumps({t: [{"attempt_1": [[0]], "attempt_2": [[0]]}] for t in ch}))
    monkeypatch.setattr(hybrid, "WORK", work)
    monkeypatch.setattr(hybrid, "SUB", work / "submission.json")
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    monkeypatch.setattr(hybrid, "challenges_path", lambda rerun: tmp_path / "ch.json")
    monkeypatch.setattr(hybrid.time, "time", lambda: 1000.0)
    monkeypatch.setattr(hybrid, "prepare_runtime", lambda cfg, st: st.update(py="py"))
    monkeypatch.setattr(hybrid, "run_nvarc", lambda *a, **k: None)
    monkeypatch.setattr(hybrid, "decode_nvarc", lambda p: (
        {"t1": [{"attempt_1": [[1]], "attempt_2": [[2]]}], "t2": [{"attempt_1": [[3]], "attempt_2": [[4]]}]},
        {"t1_0": {"conf": 0.9}, "t2_0": {"conf": 0.1}}))
    monkeypatch.setattr(hybrid, "start_server_ladder", lambda *a: object())
    monkeypatch.setattr(hybrid, "kill_server", lambda *a, **k: None)
    monkeypatch.setattr(hybrid, "gpus_free", lambda *a, **k: True)
    seen = {}

    def fake_agent(cfg, tasks, order, nv_sub, deadline, results):
        seen["order"] = order
        results["t2"] = [{"status": "verified", "programs": [{"n_ok": 2, "n_train": 2, "test": [[[5, 4]]]}]}]
    monkeypatch.setattr(hybrid, "run_agent", fake_agent)
    cfg = {"mode": "split" if split else "sequential", "nvarc_hours": 5, "smoke_tasks": [], **extra}
    hybrid.main(cfg)
    return json.load(open(work / "submission.json")), seen


@pytest.mark.parametrize("split", [False, True])
def test_main_merges_nvarc_and_agent(tmp_path, monkeypatch, split):
    sub, seen = _fake_main_env(tmp_path, monkeypatch, split)
    assert sub["t1"] == [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    assert sub["t2"] == [{"attempt_1": [[5, 4]], "attempt_2": [[3]]}]
    if not split:
        assert seen["order"] == ["t2", "t1"]       # uncertain NVARC task first


def test_skip_agent_keeps_nvarc(tmp_path, monkeypatch):
    sub, seen = _fake_main_env(tmp_path, monkeypatch, False, skip_agent=True)
    assert "order" not in seen
    assert sub["t2"] == [{"attempt_1": [[3]], "attempt_2": [[4]]}]


def test_decode_nvarc_on_synthetic_outputs(tmp_path, monkeypatch):
    """NVARC pickles -> ranked submission + confidence, with transformers stubbed out."""
    import bz2
    import pickle
    import types

    np = pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=object))
    ch = {"t1": TASK, "t2": TASK}
    chal = tmp_path / "ch.json"
    chal.write_text(json.dumps(ch))
    store = tmp_path / "out"
    store.mkdir()
    good, bad = np.array([[5, 4]]), np.array([[1, 1]])
    samples = [{"beam_score": 0.1, "score_aug": [0.1, 0.2], "solution": good},
               {"beam_score": 0.5, "score_aug": [0.9, 0.9], "solution": bad}]
    with bz2.BZ2File(store / "t1_0.permuteX", "w") as f:
        pickle.dump(samples, f)
    with bz2.BZ2File(store / "t1_0.rot90", "w") as f:
        pickle.dump(samples[:1], f)
    sub, meta = hybrid.decode_nvarc(chal, store=str(store))
    assert sub["t1"][0]["attempt_1"] == [[5, 4]]
    assert sub["t1"][0]["attempt_2"] == [[1, 1]]
    assert sub["t2"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}   # never decoded
    assert meta["t1_0"]["n"] == 3 and abs(meta["t1_0"]["conf"] - 2 / 3) < 1e-9


def test_eval_report_counts_false_verified():
    tasks = {"a": TASK, "b": TASK, "c": TASK}
    sols = {"a": [[[5, 4]]], "b": [[[5, 4]]], "c": [[[5, 4]]]}
    sub = {"a": [{"attempt_1": [[5, 4]], "attempt_2": [[0]]}],
           "b": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
           "c": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}
    results = {"a": [{"status": "verified", "gen_tokens": 10}],
               "b": [{"status": "verified", "gen_tokens": 5}],
               "c": [{"status": "failed", "gen_tokens": 7}]}
    s = hybrid.eval_report(tasks, results, sub, sols)["summary"]
    assert (s["attempted"], s["verified"], s["correct"], s["false_verified"]) == (3, 2, 1, 1)
    assert s["outputs"] == (1, 3)


def test_agent_eval_mode_skips_nvarc(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(hybrid, "run_nvarc", lambda *a, **k: called.setdefault("nvarc", 1))
    work = tmp_path / "w"
    work.mkdir()
    ch = {"t1": TASK, "t2": TASK}
    (tmp_path / "ch.json").write_text(json.dumps(ch))
    (tmp_path / "sol.json").write_text(json.dumps({"t1": [[[5, 4]]], "t2": [[[5, 4]]]}))
    (work / "start_epoch.txt").write_text("0")
    (work / "submission.json").write_text(json.dumps({}))
    monkeypatch.setattr(hybrid, "WORK", work)
    monkeypatch.setattr(hybrid, "SUB", work / "submission.json")
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    monkeypatch.setattr(hybrid, "challenges_path", lambda rerun: tmp_path / "ch.json")
    monkeypatch.setattr(hybrid, "one", lambda pat: tmp_path / "sol.json")
    monkeypatch.setattr(hybrid, "prepare_runtime", lambda cfg, st: st.update(py="py"))
    monkeypatch.setattr(hybrid, "start_server_ladder", lambda *a: object())
    monkeypatch.setattr(hybrid, "kill_server", lambda *a, **k: None)
    monkeypatch.setattr(hybrid, "gpus_free", lambda *a, **k: True)

    def fake_agent(cfg, tasks, order, nv_sub, deadline, results):
        results["t1"] = [{"status": "verified", "gen_tokens": 3,
                          "programs": [{"n_ok": 2, "n_train": 2, "test": [[[5, 4]]]}]}]
    monkeypatch.setattr(hybrid, "run_agent", fake_agent)
    hybrid.main({"mode": "sequential", "nvarc_hours": 5, "smoke_tasks": [], "agent_eval_h": 1.0})
    rep = json.loads((work / "agent_eval.json").read_text())["summary"]
    assert "nvarc" not in called
    assert rep["correct"] == 1 and rep["tasks"] == 2


def test_forced_tool_call_disables_thinking(monkeypatch):
    import client as cl

    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent.clear()
        sent.update(json.loads(req.data.decode()))
        return Resp()

    monkeypatch.setattr(cl.urllib.request, "urlopen", fake_urlopen)
    c = cl.ChatClient("http://x/v1", "qwen", thinking_token_budget=6144, max_tokens=16384)
    tools = [{"type": "function", "function": {"name": "submit", "parameters": {}}}]
    c.chat([{"role": "user", "content": "hi"}], tools=tools)
    assert sent["chat_template_kwargs"]["enable_thinking"] is True and sent["thinking_token_budget"] == 6144
    c.chat([{"role": "user", "content": "hi"}], tools=tools, tool_choice="submit")
    assert sent["chat_template_kwargs"]["enable_thinking"] is False
    assert "thinking_token_budget" not in sent and sent["max_tokens"] == 8192
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "submit"}}


def test_forced_submit_error_does_not_end_attempt():
    from repl_agent import solve_repl

    class Flaky:
        def __init__(self):
            self.choices = []

        def chat(self, messages, tools=None, tool_choice=None, **kw):
            self.choices.append(tool_choice)
            if tool_choice == "submit":
                return {"error": "HTTP 500: ", "context_overflow": False}
            if len(self.choices) < 5:
                return {"content": "", "reasoning": "hmm", "finish_reason": "length", "tool_calls": [],
                        "prompt_tokens": 1, "completion_tokens": 1}
            return {"content": "", "reasoning": "", "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "s", "name": "submit", "arguments": {"code": FLIP}}],
                    "prompt_tokens": 1, "completion_tokens": 1}

    c = Flaky()
    _, report = solve_repl({"t1": TASK}, c, attempts=1, concurrency=1, max_turns=10,
                           max_gen_tokens=1000, context_budget=8000, nudge_every=6,
                           force_submit_turns=2)
    assert "submit" in c.choices
    assert report["attempts"][0]["status"] == "verified"


def test_nvarc_crash_still_writes_full_submission(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nvarc died")
    monkeypatch.setattr(hybrid, "run_nvarc", boom)
    monkeypatch.setattr(hybrid, "decode_nvarc", boom)
    real_setattr = monkeypatch.setattr

    def guarded(target, name, value=None, *rest):
        if target is hybrid and name in ("run_nvarc", "decode_nvarc"):
            return None                           # keep the crashing versions
        return real_setattr(target, name, value, *rest)
    monkeypatch.setattr = guarded
    sub, _ = _fake_main_env(tmp_path, monkeypatch, False)
    assert set(sub) == {"t1", "t2"}
    assert sub["t2"][0]["attempt_1"] == [[5, 4]]                     # agent verified answer kept
    assert sub["t1"][0]["attempt_1"] == TASK["test"][0]["input"]       # fallback, not [[0]]


def test_server_failure_keeps_nvarc(tmp_path, monkeypatch):
    def no_server(*a):
        raise RuntimeError("no server")
    real_setattr = monkeypatch.setattr

    def guarded(target, name, value=None, *rest):
        if target is hybrid and name == "start_server_ladder":
            return real_setattr(target, name, no_server)
        return real_setattr(target, name, value, *rest)
    monkeypatch.setattr = guarded
    sub, seen = _fake_main_env(tmp_path, monkeypatch, False)
    assert "order" not in seen
    assert sub["t1"] == [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    assert sub["t2"] == [{"attempt_1": [[3]], "attempt_2": [[4]]}]


def test_local_eval_harness_runs_end_to_end(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "local_eval", Path(__file__).resolve().parents[1] / "scripts" / "local_eval.py")
    le = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(le)

    class Solver:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages, tools=None, tool_choice=None, **kw):
            return {"content": "", "reasoning": "", "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "s", "name": "submit", "arguments": {"code": FLIP}}],
                    "prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(le, "RetryingClient", Solver)
    monkeypatch.setattr(le, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "arc-agi_evaluation_challenges.json").write_text(json.dumps({"t1": TASK, "t2": TASK}))
    (tmp_path / "data" / "arc-agi_evaluation_solutions.json").write_text(
        json.dumps({"t1": [[[5, 4]]], "t2": [[[0]]]}))
    monkeypatch.setattr(sys, "argv", ["local_eval.py", "--name", "x", "--tasks", "2", "--hours", "0.01", "--split", "evaluation"])
    le.main()
    s = json.loads((tmp_path / "work" / "local_eval" / "x.json").read_text())["summary"]
    assert (s["tasks"], s["verified"], s["correct"], s["false_verified"]) == (2, 2, 1, 1)


def test_direct_parse_grids_takes_last_blocks():
    from direct import parse_grids
    text = "think\n```grid\n12\n34\n```\nfinal:\n```grid\n5 6\n7 8\n```\n```\n9\n```"
    assert parse_grids(text, 2) == [[[5, 6], [7, 8]], [[9]]]
    assert parse_grids("no grid here", 1) is None


def test_merge_with_direct_answers():
    t = {"train": TASK["train"], "test": [{"input": [[1]]}]}
    nv = [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    d = lambda g: {"grids": [g]}
    # two direct samples agree -> they replace NVARC top2
    r = hybrid.merge_task(t, nv, None, [d([[7]]), d([[7]])], {0: 0.9})
    assert (r[0]["attempt_1"], r[0]["attempt_2"]) == ([[1]], [[7]])
    # one direct sample, NVARC confident -> keep NVARC top2
    r = hybrid.merge_task(t, nv, None, [d([[7]])], {0: 0.9})
    assert r[0]["attempt_2"] == [[2]]
    # one direct sample, NVARC unsure -> direct answer
    r = hybrid.merge_task(t, nv, None, [d([[7]])], {0: 0.1})
    assert r[0]["attempt_2"] == [[7]]
    # NVARC missing -> both direct samples
    empty = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    r = hybrid.merge_task(t, empty, None, [d([[7]]), d([[8]])], None)
    assert {repr(r[0]["attempt_1"]), repr(r[0]["attempt_2"])} == {"[[7]]", "[[8]]"}
    # REPL-verified still first
    r = hybrid.merge_task(t, nv, [([[[5]]], [])], [d([[7]]), d([[7]])], {0: 0.1})
    assert (r[0]["attempt_1"], r[0]["attempt_2"]) == ([[5]], [[1]])


def test_run_direct_sample_major_with_rescue():
    from direct import run_direct

    class Fake:
        max_tokens = 1000

        def __init__(self):
            self.calls = []

        def chat(self, messages, seed=None, max_tokens=None, timeout=None, template_kwargs=None, **kw):
            self.calls.append((seed, template_kwargs))
            if template_kwargs:                      # rescue call: answer
                return {"content": "```grid\n54\n```", "reasoning": "", "finish_reason": "stop",
                        "completion_tokens": 3, "prompt_tokens": 1}
            if seed == 7:                             # sample 0: cut off, no grid
                return {"content": "", "reasoning": "hmm " * 10, "finish_reason": "length",
                        "completion_tokens": 9, "prompt_tokens": 1}
            return {"content": "```grid\n54\n```", "reasoning": "", "finish_reason": "stop",
                    "completion_tokens": 5, "prompt_tokens": 1}

    c = Fake()
    res = run_direct(c, {"t1": TASK}, ["t1"], deadline=time.time() + 600, samples=2, concurrency=1,
                     log=lambda *a: None)
    st = sorted(r["status"] for r in res["t1"])
    assert st == ["answered", "rescued"]
    assert all(r["grids"] == [[[5, 4]]] for r in res["t1"])


def test_main_direct_mode_merges_direct_answers(tmp_path, monkeypatch):
    calls = {}

    def fake_direct(cfg, tasks, order, nv_sub, conf_of, deadline, direct):
        calls["order"] = order
        direct["t2"] = [{"grids": [[[9]]]}, {"grids": [[[9]]]}]
    real_setattr = monkeypatch.setattr
    real_setattr(hybrid, "run_direct_phase", fake_direct)

    def guarded(target, name, value=None, *rest):
        if target is hybrid and name == "run_direct_phase":
            return None
        return real_setattr(target, name, value, *rest)
    monkeypatch.setattr = guarded
    sub, seen = _fake_main_env(tmp_path, monkeypatch, False, agent_mode="direct")
    assert "order" not in seen and calls["order"] == ["t2", "t1"]   # REPL agent not used
    assert sub["t2"] == [{"attempt_1": [[3]], "attempt_2": [[9]]}]    # agreeing direct answer
    assert sub["t1"] == [{"attempt_1": [[1]], "attempt_2": [[2]]}]


def test_minimal_save_mode_skips_everything(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(hybrid, "run_nvarc", lambda *a, **k: called.setdefault("nvarc", 1))
    monkeypatch.setattr(hybrid, "prepare_runtime", lambda *a: called.setdefault("rt", 1))
    work = tmp_path / "w"
    work.mkdir()
    (work / "start_epoch.txt").write_text("0")
    monkeypatch.setattr(hybrid, "WORK", work)
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    hybrid.main({"save_mode": "minimal"})
    assert called == {}


def test_minimal_save_mode_does_not_skip_agent_eval(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "start_epoch.txt").write_text("0")
    monkeypatch.setattr(hybrid, "WORK", work)
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached

    monkeypatch.setattr(hybrid, "challenges_path", stop)
    import pytest
    with pytest.raises(Reached):                 # went past the minimal early return
        hybrid.main({"save_mode": "minimal", "agent_eval_h": 2.0})


def test_direct_plan_skips_confident_and_sizes_cap():
    conf = {"a": {0: 0.9}, "b": {0: 0.2}, "c": {0: 0.1, 1: 0.95}}
    cfg = {"direct_tok_per_s": 100, "direct_min_tokens": 1000, "direct_max_tokens": 50000}
    todo, cap = hybrid.direct_plan(cfg, ["a", "b", "c"], conf.get, 600, True)
    assert todo == ["b", "c"] and cap == 30000            # 100 tok/s * 600 s / 2 tasks
    todo, cap = hybrid.direct_plan(cfg, ["a", "b", "c"], conf.get, 600, False)
    assert todo == ["a", "b", "c"] and cap == 20000        # no NVARC: every task
    _, cap = hybrid.direct_plan(cfg, ["a", "b", "c"], conf.get, 1, True)
    assert cap == 1000                                     # clamped to the minimum
