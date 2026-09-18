import subprocess, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; SCR = ROOT/"scripts"; PY = sys.executable
for dom in ["earth_science", "psychology", "sustainable_living"]:
    if not (ROOT/"data"/"bright_hardpool"/dom/"pools.json").exists():
        print(f"### A build: {dom}", flush=True)
        subprocess.run([PY, "-u", "bright_hardpool_baseline.py", dom], cwd=SCR, check=True)
    else:
        print(f"### A build: {dom} (skip)", flush=True)
    if not (ROOT/"results"/"bright_ch_anchor"/dom/"per_query.json").exists():
        print(f"### BCD: {dom}", flush=True)
        subprocess.run([PY, "-u", "bright_headroom_bcd.py", dom], cwd=SCR, check=True)
    else:
        print(f"### BCD: {dom} (skip)", flush=True)
print("ALL DOMAINS DONE", flush=True)
