"""NVARC launcher (from koushikrudra/failed-in-aimo, Apache 2.0), task list taken from a file.

    python starter.py --end-time <epoch> --keys keys.json --challenges <path>
"""
import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, challenges, gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    os.environ["HYBRID_CHALLENGES"] = challenges
    torch.set_default_device("cpu")
    # Fix Unsloth patching issue: workers import one after another.
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    from arc_solver import worker
    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    print(f"[Rank {rank}] start!", flush=True)
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    keys = json.load(open(args.keys))
    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    gpus = [int(g) for g in args.gpus.split(",")]
    for _ in gpus:
        queue.put(None)
    print(f"[starter] {len(keys)} tasks, end in {(args.end_time - time.time()) / 60:.1f} min",
          flush=True)
    mp.spawn(local_worker, args=(queue, args.end_time, args.challenges, gpus), nprocs=len(gpus))
