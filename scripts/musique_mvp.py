"""MuSiQue MVP: (a) supporting-paragraph ranking with the single-pass generalist,
and (b) a closed-book memorization probe. The probe is the decisive check before
adopting MuSiQue: if the model answers without the paragraphs, "reasoning over
provided evidence" collapses (the Experiment-1 lesson). Fetches validation rows
from HuggingFace datasets-server (public). Separate from CORDIS and TREC CT.
"""
from __future__ import annotations
import json, re, urllib.request, random, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

DIR = Path(__file__).resolve().parents[1] / "data" / "musique_mvp"
DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
N = 30
PARA_CAP = 450
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=dgslibisey/MuSiQue&config=default&split=validation&offset=0&length=100"


def fetch():
    cache = DIR / "val_rows.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(ROWS_API, timeout=40) as r:
        data = json.load(r)
    rows = [x["row"] for x in data["rows"]]
    cache.write_text(json.dumps(rows))
    return rows


def norm(s): return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
def contains(pred, golds):
    p = norm(pred)
    return any(norm(g) and (norm(g) in p or p in norm(g)) for g in golds)


def rank_prompt(question, paras):
    body = [{"id": p["pid"], "text": p["paragraph_text"][:PARA_CAP]} for p in paras]
    return f"""You are given a multi-hop question and {len(paras)} candidate paragraphs
(opaque ids). Rank ALL paragraph ids from most to least likely to be a NEEDED
supporting fact for answering the question (a multi-hop question needs several
paragraphs chained together). Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Question: {question}

Paragraphs:
{json.dumps(body, ensure_ascii=False)}
"""


def closed_book_prompt(question):
    return f"""Answer this question with a short answer only (a name, entity, place,
or number). If you do not know, answer "UNKNOWN". Return JSON only.

Schema: {{"answer": "<short answer>"}}

Question: {question}
"""


def parse_ids(raw, valid):
    try:
        obj = extract_json_object(raw); ids = obj.get("ranked_ids", obj) if isinstance(obj, dict) else obj
    except Exception:
        return []
    out = []
    if isinstance(ids, list):
        for x in ids:
            v = str(x.get("id") if isinstance(x, dict) else x)
            if v in valid and v not in out:
                out.append(v)
    return out


def main():
    rows = fetch()[:N]
    client = make_client(MODEL)
    rng = random.Random(42)
    r1 = 0; mrrs = []; all_in5 = 0; supp_ranks = []
    cb_correct = 0; cb_unknown = 0
    per = []
    for r in rows:
        paras = list(r["paragraphs"])
        rng.shuffle(paras)
        for i, p in enumerate(paras):
            p["pid"] = f"P{i:02d}"
        lab = {p["pid"]: (str(p["is_supporting"]).lower() == "true") for p in paras}
        gold = {pid for pid, s in lab.items() if s}
        # (a) ranking
        raw = client.generate(rank_prompt(r["question"], paras))
        ranked = parse_ids(raw, set(lab))
        for p in paras:
            if p["pid"] not in ranked:
                ranked.append(p["pid"])
        first = next((i + 1 for i, pid in enumerate(ranked) if pid in gold), len(ranked) + 1)
        r1 += first <= 1; mrrs.append(1 / first)
        topk = ranked[:5]
        all_in5 += gold.issubset(set(topk))
        supp_ranks += [i + 1 for i, pid in enumerate(ranked) if pid in gold]
        # (b) closed-book
        try:
            cbraw = client.generate(closed_book_prompt(r["question"]))
            ans = extract_json_object(cbraw).get("answer", "") if cbraw.strip().startswith("{") else cbraw
        except Exception:
            ans = ""
        golds = [r["answer"]] + (r.get("answer_aliases") or [])
        is_unk = norm(ans) in ("unknown", "")
        ok = (not is_unk) and contains(ans, golds)
        cb_correct += ok; cb_unknown += is_unk
        per.append({"id": r["id"], "hops": len(r["question_decomposition"]),
                    "first_supp_rank": first, "closed_book_ans": ans, "gold": r["answer"], "cb_correct": ok})
    (DIR / "mvp_per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(rows)
    print(f"=== MuSiQue MVP (n={n} validation, {MODEL}) ===")
    print("\n(a) Supporting-paragraph RANKING (gold = is_supporting, 2-4 per q):")
    print(f"    R@1 (top paragraph is supporting): {r1}/{n} = {r1/n:.2f}")
    print(f"    MRR (first supporting):            {st.mean(mrrs):.3f}")
    print(f"    all supporting in top-5:           {all_in5}/{n} = {all_in5/n:.2f}")
    print(f"    mean rank of supporting paras:     {st.mean(supp_ranks):.2f} (of 20)")
    print("\n(b) CLOSED-BOOK memorization probe (NO paragraphs given):")
    print(f"    answered correctly from memory:    {cb_correct}/{n} = {cb_correct/n:.2f}")
    print(f"    said UNKNOWN:                      {cb_unknown}/{n} = {cb_unknown/n:.2f}")
    print(f"    -> memorization rate {cb_correct/n:.0%}; higher = worse for a 'reason over evidence' claim")


if __name__ == "__main__":
    main()
