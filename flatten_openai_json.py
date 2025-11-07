#!/usr/bin/env python3
import argparse, json, re, pandas as pd
from html import unescape

GENERIC = {
    "drug", "drugs", "therapy", "treatment", "bromodomain inhibitors",
    "fgfr inhibitors", "serotonergic gpcr agonists"
}

def html_strip(s: str) -> str:
    # remove simple <sub> / <sup> etc. while keeping text
    s = unescape(s)
    s = re.sub(r"<\s*sub\s*>", "_", s, flags=re.I)    # Aβ<sub>42</sub> -> Aβ_42
    s = re.sub(r"<\s*/\s*sub\s*>", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)                     # drop other tags
    return s

def split_multi_drugs(name: str):
    # split "5-HT4, 5-HT6, 5-HT2C agonists" -> ["5-HT4 agonist","5-HT6 agonist","5-HT2C agonist"]
    if "," in name and "agonist" in name.lower():
        base = name.strip()
        tail = "agonist" if "agonists" in base.lower() else "agonist"
        parts = [p.strip() for p in base.split(",")]
        # keep the last token (e.g., "5-HT2C agonists") to infer suffix if needed
        if not parts[-1].lower().endswith("agonist") and not parts[-1].lower().endswith("agonists"):
            return [base]
        return [re.sub(r"\s*agonists?$", "", p, flags=re.I).strip() + " " + tail for p in parts]
    return [name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV from openai_extract_candidates.py (raw_json column)")
    ap.add_argument("--out", required=True, help="Output CSV (flattened)")
    ap.add_argument("--summary", default="data/candidate_summary.csv", help="Output CSV (grouped summary)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    rows = []
    for i, s in enumerate(df["raw_json"].fillna("[]")):
        try:
            data = json.loads(s)
            if not isinstance(data, list):
                continue
        except json.JSONDecodeError:
            # try to salvage: find first [...] block
            m = re.search(r"\[[\s\S]*\]", s)
            if not m:
                continue
            try:
                data = json.loads(m.group(0))
            except Exception:
                continue

        for obj in data:
            drug = html_strip(str(obj.get("candidate_drug", "")).strip())
            ev   = html_strip(str(obj.get("evidence_sentence", "")).strip())
            mech = html_strip(str(obj.get("mechanism_or_pathway", "")).strip())
            stance = str(obj.get("stance", "")).strip().lower()

            # split multi-drug fields if present
            split_names = []
            for d in split_multi_drugs(drug):
                d = d.strip()
                if not d:
                    continue
                dnorm = d.lower()
                if dnorm in GENERIC:
                    continue
                split_names.append(d)

            if not split_names:
                # keep the original if nothing usable after split/filter
                if drug and drug.lower() not in GENERIC:
                    split_names = [drug]

            for name in split_names:
                rows.append({
                    "candidate_drug": name,
                    "stance": stance if stance in {"supportive","inconclusive","contradictory"} else "",
                    "mechanism_or_pathway": mech,
                    "evidence_sentence": ev
                })

    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(args.out, index=False)

    # Make a quick grouped summary
    if not out.empty:
        summ = (out.assign(stance=out["stance"].replace({"": "unspecified"}))
                  .groupby(["candidate_drug","stance"])
                  .size()
                  .reset_index(name="hits")
                  .sort_values(["hits"], ascending=False))
        summ.to_csv(args.summary, index=False)

    print(f"✅ Flattened rows: {len(out)} -> {args.out}")
    if not out.empty:
        print(f"📊 Summary written to: {args.summary}")

if __name__ == "__main__":
    main()
