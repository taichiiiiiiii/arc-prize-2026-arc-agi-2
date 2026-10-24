"""Local checks of the hybrid kernel's merge logic and the REPL agent loop (no GPU, fake LLM)."""
import json
import sys
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


def _fake_main_env(tmp_path, monkeypatch, split):
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
    cfg = {"mode": "split" if split else "sequential", "nvarc_hours": 5, "smoke_tasks": []}
    hybrid.main(cfg)
    return json.load(open(work / "submission.json")), seen


@pytest.mark.parametrize("split", [False, True])
def test_main_merges_nvarc_and_agent(tmp_path, monkeypatch, split):
    sub, seen = _fake_main_env(tmp_path, monkeypatch, split)
    assert sub["t1"] == [{"attempt_1": [[1]], "attempt_2": [[2]]}]
    assert sub["t2"] == [{"attempt_1": [[5, 4]], "attempt_2": [[3]]}]
    if not split:
        assert seen["order"] == ["t2", "t1"]       # uncertain NVARC task first


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
