#!/usr/bin/env bash
# Submit a hybrid kernel version once: its save run completed with the smoke markers and
# today's (UTC) slot is unused. Then watch the score.
#   watch_submit_hybrid.sh <kernel-ref> <version> "<message>"
set -u
K=$1; V=$2; MSG=$3
C=arc-prize-2026-arc-agi-2
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/Users/<user>/.hermes/profiles/arc2/cache/scratch/$(basename $K)-v$V.log
subs() { kaggle competitions submissions $C --page-size 30 --csv 2>&1 | grep -v Warning; }
for i in $(seq 1 60); do
  s=$(kaggle kernels status $K 2>&1 | grep -v Warning)
  echo "$(date -u +%H:%M) $s"
  echo "$s" | grep -qi authentication && { echo AUTH_FAIL; exit 2; }
  echo "$s" | grep -qi complete && break
  echo "$s" | grep -qiE 'error|cancel' && { echo KERNEL_FAILED; exit 3; }
  sleep 240
done
$PY $HERE/fetch_log.py $K $LOG 2>&1 | grep -v Warning | grep -E "server ready|SMOKE|failed|verified [0-9]" | tail -12
grep -q "server ready" <(cat $LOG) && grep -q "SMOKE score merged" <(cat $LOG) || { echo SMOKE_MARKERS_MISSING; exit 4; }
# wait for: nothing pending, and no submission yet today (UTC)
for i in $(seq 1 300); do
  r=$(subs)
  echo "$r" | grep -qi authentication && { echo AUTH_FAIL; exit 2; }
  if echo "$r" | grep -qF "$MSG"; then echo ALREADY_SUBMITTED; break; fi
  today=$(date -u +%Y-%m-%d)
  pending=$(echo "$r" | grep -c PENDING)
  used=$(echo "$r" | grep -c ",$today ")
  echo "$(date -u '+%m-%d %H:%M') pending=$pending today=$used"
  # pending submissions do not block a new one (observed 10/25); only the UTC daily slot matters
  if [ "$used" -eq 0 ]; then
    kaggle competitions submit $C -k $K -v $V -f submission.json -m "$MSG" 2>&1 | grep -v Warning
    break
  fi
  sleep 600
done
for i in $(seq 1 200); do
  line=$(subs | grep -F "$MSG" | head -1)
  echo "$(date -u '+%m-%d %H:%M') $line"
  if [ -n "$line" ] && ! echo "$line" | grep -qiE 'pending|running'; then echo SCORED; exit 0; fi
  sleep 600
done
