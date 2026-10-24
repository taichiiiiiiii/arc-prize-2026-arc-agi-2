"""Local re-implementation of the competition metric.

Each task test output scores 1 if attempt_1 or attempt_2 equals the ground
truth exactly (shape and every cell), else 0. The score is the mean over all
task test outputs.
"""

from __future__ import annotations

from .data import Grid


def score_submission(submission: dict, solutions: dict[str, list[Grid]]) -> float:
    total = 0
    correct = 0
    for task_id, truths in solutions.items():
        preds = submission.get(task_id, [])
        for i, truth in enumerate(truths):
            total += 1
            if i < len(preds):
                attempts = preds[i]
                if any(attempts.get(k) == truth for k in ("attempt_1", "attempt_2")):
                    correct += 1
    return correct / total if total else 0.0


def per_task_scores(submission: dict, solutions: dict[str, list[Grid]]) -> dict[str, float]:
    out = {}
    for task_id, truths in solutions.items():
        out[task_id] = score_submission({task_id: submission.get(task_id, [])}, {task_id: truths})
    return out
