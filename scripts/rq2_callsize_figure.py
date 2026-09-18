"""Cross-model call-size figure for Experiment 1.

Replaces the single-model call-size-bias bar chart with the paper's actual
finding: the bias (large true calls ranking far worse than small ones) shrinks
and reverses as the reranker model gets stronger. Grouped bars of Recall@1 for
small (<=25 projects) vs large (>=150 projects) true calls, per model, ordered
weak->strong (held-out test set). Reads the per-model call-size strata computed
by rq2_error_breakdown.py; no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rq2_core import RESULT_ROOT

ORDER = [
    ("gemini-2.5-flash-lite", "Gemini\nflash-lite"),
    ("claude-haiku-4-5", "Claude\nHaiku 4.5"),
    ("claude-sonnet-5", "Claude\nSonnet 5"),
    ("claude-opus-4-8", "Claude\nOpus 4.8"),
    ("gemini-flash-latest", "Gemini\nflash-latest"),
]
SMALL_KEY, LARGE_KEY = "small (<=25)", "large (>=150)"
SMALL_COLOR, LARGE_COLOR = "#2a9d8f", "#e76f51"  # teal, terracotta (colour-blind safe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", default=str(RESULT_ROOT / "rq2_error_breakdown" / "rq2_error_breakdown.json"))
    ap.add_argument("--out", default="rq2_callsize_by_model")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rep = json.loads(Path(args.breakdown).read_text())
    small, large, labels = [], [], []
    for model, label in ORDER:
        r = next((x for x in rep if x["split"] == args.split and x["model"] == model
                  and x["condition"].startswith("baseline")), None)
        if not r:
            continue
        s = r["call_size_strata"]
        small.append(s.get(SMALL_KEY, {}).get("recall@1", 0.0))
        large.append(s.get(LARGE_KEY, {}).get("recall@1", 0.0))
        labels.append(label)

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], small, w, label="Small calls ($\\leq$25 projects)", color=SMALL_COLOR)
    b2 = ax.bar([i + w / 2 for i in x], large, w, label="Large calls ($\\geq$150 projects)", color=LARGE_COLOR)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 1), textcoords="offset points")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall@1  (higher is better)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("The call-size bias vanishes as the reranker model gets stronger\n"
                 "(held-out test set; weak $\\rightarrow$ strong model)", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
