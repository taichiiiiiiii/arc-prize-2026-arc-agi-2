#!/usr/bin/env bash
# Watch kernel v1 of arc2-frontier-copy-3389, submit once when complete, then watch the score.
set -u
K=taichiiiii/arc2-frontier-copy-3389
C=arc-prize-2026-arc-agi-2
MSG="frontier copy koushikrudra/failed-in-aimo V1 (33.89) + model mount path fix, v2"
OUT=${ARC2_SCRATCH:-$(cd "$(dirname "$0")/.." && pwd)/work/scratch}/fc_out
for i in $(seq 1 120); do
  s=$(kaggle kernels status $K 2>&1 | grep -v Warning)
  echo "$(date -u +%H:%M) $s"
  if echo "$s" | grep -qi authentication; then echo AUTH_FAIL; exit 2; fi
  if echo "$s" | grep -qiE 'complete'; then break; fi
  if echo "$s" | grep -qiE 'error|cancel'; then echo KERNEL_FAILED; exit 3; fi
  sleep 180
done
rm -rf $OUT; mkdir -p $OUT
kaggle kernels output $K -p $OUT --file-pattern '(arc2-frontier-copy-3389\.log|submission\.json)$' 2>&1 | grep -v Warning | tail -2
ls -la $OUT
/usr/local/bin/python3 - "$OUT" <<'EOF'
import json,sys,os
o=sys.argv[1]
log=os.path.join(o,'arc2-frontier-copy-3389.log')
if os.path.exists(log):
    txt=''.join(e.get('data','') for e in json.load(open(log)))
    for line in txt.splitlines():
        if 'Reload score' in line or 'LegC' in line or 'Traceback' in line: print('LOG:',line[:200])
    if 'Reload score' not in txt:
        print('NO_RELOAD_SCORE'); sys.exit(5)
s=json.load(open(os.path.join(o,'submission.json'))); print('sub tasks',len(s))
EOF
[ $? -eq 0 ] || { echo LOG_CHECK_FAILED; exit 5; }
[ -f $OUT/submission.json ] || { echo NO_SUBMISSION_FILE; exit 4; }
if kaggle competitions submissions $C --page-size 20 --csv 2>/dev/null | grep -qF "$MSG"; then echo ALREADY_SUBMITTED; else
  kaggle competitions submit $C -k $K -v 2 -f submission.json -m "$MSG" 2>&1 | grep -v Warning
fi
for i in $(seq 1 200); do
  r=$(kaggle competitions submissions $C --page-size 20 --csv 2>&1 | grep -v Warning)
  if echo "$r" | grep -qi authentication; then echo AUTH_FAIL; exit 2; fi
  line=$(echo "$r" | grep -F "$MSG" | head -1)
  echo "$(date -u "+%m-%d %H:%M") $line"
  if [ -n "$line" ] && ! echo "$line" | grep -qiE 'pending|running'; then echo SCORED; exit 0; fi
  sleep 600
done
