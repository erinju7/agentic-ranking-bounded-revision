#!/bin/bash
cd "$(dirname "$0")"
PY=../../venv/bin/python
for dom in earth_science psychology sustainable_living; do
  if [ ! -f ../data/bright_hardpool/$dom/pools.json ]; then
    echo "### A build: $dom"; $PY -u bright_hardpool_baseline.py "$dom" || { echo "A FAILED $dom"; continue; }
  else echo "### A build: $dom (skip; pools exist)"; fi
  if [ ! -f ../results/bright_ch_anchor/$dom/per_query.json ]; then
    echo "### BCD: $dom"; $PY -u bright_headroom_bcd.py "$dom" || { echo "BCD FAILED $dom"; continue; }
  else echo "### BCD: $dom (skip; per_query exists)"; fi
done
echo "ALL DOMAINS DONE"
