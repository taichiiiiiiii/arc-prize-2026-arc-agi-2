"""Build and validate submission.json."""

from __future__ import annotations

import json
from pathlib import Path

from .data import Grid

ATTEMPTS = ("attempt_1", "attempt_2")


def is_valid_grid(g) -> bool:
    if not isinstance(g, list) or not 1 <= len(g) <= 30:
        return False
    width = len(g[0]) if isinstance(g[0], list) else -1
    if not 1 <= width <= 30:
        return False
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for v in row)
        for row in g
    )


def validate(submission: dict, challenges: dict) -> list[str]:
    """Return a list of problems; empty means the file matches the required format."""
    errors = []
    missing = set(challenges) - set(submission)
    extra = set(submission) - set(challenges)
    if missing:
        errors.append(f"{len(missing)} task ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown task ids, e.g. {sorted(extra)[:3]}")
    for task_id, task in challenges.items():
        preds = submission.get(task_id)
        if preds is None:
            continue
        if not isinstance(preds, list) or len(preds) != len(task["test"]):
            errors.append(f"{task_id}: expected {len(task['test'])} outputs")
            continue
        for i, p in enumerate(preds):
            for k in ATTEMPTS:
                if not isinstance(p, dict) or k not in p:
                    errors.append(f"{task_id}[{i}]: missing {k}")
                elif not is_valid_grid(p[k]):
                    errors.append(f"{task_id}[{i}].{k}: invalid grid")
    return errors


def make_submission(challenges: dict, predict) -> dict:
    """predict(task, test_index) -> (attempt_1, attempt_2). Falls back to [[0]] on error."""
    sub = {}
    for task_id, task in challenges.items():
        outs = []
        for i in range(len(task["test"])):
            try:
                a1, a2 = predict(task, i)
            except Exception:  # noqa: BLE001 - a crash must not invalidate the file
                a1 = a2 = None
            a1 = a1 if is_valid_grid(a1) else [[0]]
            a2 = a2 if is_valid_grid(a2) else a1
            outs.append({"attempt_1": a1, "attempt_2": a2})
        sub[task_id] = outs
    return sub


def write(submission: dict, path: Path | str = "submission.json") -> None:
    Path(path).write_text(json.dumps(submission))


def identity_predict(task, i) -> tuple[Grid, Grid]:
    """Trivial baseline: copy the test input."""
    g = task["test"][i]["input"]
    return g, g
