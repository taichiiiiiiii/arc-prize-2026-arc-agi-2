#!/usr/bin/env bash
# Run queued local agent evaluations one after another (Mac, mlx_lm.server, small Qwen).
# Queue: work/local_eval/queue.txt, one line per run:  <name> [--set key=json ...]
# A finished run is moved to work/local_eval/done.txt. Safe to append while running.
#   nohup scripts/local_eval_queue.sh > work/local_eval/queue.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
Q=$ROOT/work/local_eval/queue.txt
DONE=$ROOT/work/local_eval/done.txt
MLX=/Users/<user>/.hermes/profiles/arc2/cache/mlx
MODEL=${LOCAL_EVAL_MODEL:-$MLX/models/Qwen3.5-9B-MLX-4bit}
mkdir -p "$ROOT/work/local_eval"; touch "$Q" "$DONE"
if ! curl -s -o /dev/null http://127.0.0.1:8765/v1/models; then
  "$MLX/.venv/bin/mlx_lm.server" --model "$MODEL" --port 8765 --host 127.0.0.1 \
    --decode-concurrency 8 --prompt-concurrency 4 > "$ROOT/work/local_eval/server.log" 2>&1 &
  SERVER=$!
  sleep 30
fi
while true; do
  line=$(grep -v '^\s*$' "$Q" | head -1)
  [ -z "$line" ] && break
  # keep >= 30% memory free (the Mac is shared with the user's Docker and jobs)
  while [ "$(memory_pressure | awk -F': ' '/free percentage/{print $2+0}')" -lt 30 ]; do
    echo "$(date -u +%H:%M) memory low, waiting"; sleep 300
  done
  name=${line%% *}; args=${line#"$name"}
  echo "$(date -u '+%m-%d %H:%M') start $name $args"
  # shellcheck disable=SC2086
  (cd "$ROOT" && uv run python scripts/local_eval.py --name "$name" --tasks 40 --hours 2 --split training $args \
     > "work/local_eval/$name.log" 2>&1)
  tail -1 "$ROOT/work/local_eval/$name.log"
  echo "$line" >> "$DONE"
  grep -vxF "$line" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"
done
echo "$(date -u '+%m-%d %H:%M') queue empty"
[ -n "${SERVER:-}" ] && kill "$SERVER"
