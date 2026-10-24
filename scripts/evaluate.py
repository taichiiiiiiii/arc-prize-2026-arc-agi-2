"""Score a submission.json against a split with known solutions.

Usage:
  uv run python scripts/evaluate.py path/to/submission.json --split evaluation
  uv run python scripts/evaluate.py --baseline identity --split evaluation
"""

from __future__ import annotations

import argparse
import json

from arc2.data import load_challenges, load_solutions
from arc2.metric import score_submission
from arc2.submission import identity_predict, make_submission, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", nargs="?")
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--baseline", choices=["identity"])
    args = ap.parse_args()

    challenges = load_challenges(args.split)
    solutions = load_solutions(args.split)
    if args.baseline == "identity":
        sub = make_submission(challenges, identity_predict)
    elif args.submission:
        with open(args.submission) as f:
            sub = json.load(f)
    else:
        ap.error("give a submission path or --baseline")

    errors = validate(sub, challenges)
    for e in errors[:20]:
        print("INVALID:", e)
    n = sum(len(v) for v in solutions.values())
    print(f"split={args.split} tasks={len(challenges)} outputs={n} valid={not errors}")
    print(f"score={score_submission(sub, solutions):.4f}")


if __name__ == "__main__":
    main()
