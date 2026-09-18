#!/bin/bash
# Re-run Study 2 (2a listwise A/B/C/D + 2b IND/DEP) with the frozen rubric injected.
# claude-haiku-4-5-20251001, tag _rubric (preserves the prior no-rubric runs). Resumable.
set -u
cd "$(dirname "$0")/.."
M=claude-haiku-4-5-20251001
LWDIR="results/james_validation/listwise/${M}_rubric"
DEPBASE="results/james_validation/study2a_dependency_${M}_rubric"

echo "=== 2a listwise (seeds 42-51) ==="
for s in 42 43 44 45 46 47 48 49 50 51; do
  if [ "$s" = 42 ]; then out="$LWDIR/listwise_results.json"; else out="$LWDIR/seed_$s/listwise_results.json"; fi
  if [ -f "$out" ]; then echo "  seed $s exists, skip"; continue; fi
  echo "[listwise seed $s]"
  python3 scripts/james_match_listwise.py --model "$M" --tag _rubric --seed "$s" || echo "  seed $s FAILED"
done

echo "=== 2b IND/DEP (reps 1-10) ==="
for r in 1 2 3 4 5 6 7 8 9 10; do
  if [ -f "${DEPBASE}_rep${r}/pairs.json" ]; then echo "  rep $r exists, skip"; continue; fi
  echo "[dependency rep $r]"
  python3 scripts/james_dependency.py --model "$M" --tag _rubric --rep "$r" || echo "  rep $r FAILED"
done

echo "DONE study2 rubric reruns"
