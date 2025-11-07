#!/usr/bin/env python3
import os, time, math, json, argparse, re, pandas as pd
from openai import OpenAI

SYSTEM_PROMPT = """You are a biomedical literature analysis assistant.
From the abstract, extract drug repurposing candidates for Alzheimer's disease.

Return ONLY a JSON array (no prose, no backticks). Each item:
- candidate_drug: specific drug/compound name
- evidence_sentence: a single sentence copied verbatim from the abstract
- stance: supportive | inconclusive | contradictory
- mechanism_or_pathway: short phrase if present, else ""

If nothing clear, return [] (an empty JSON array).
"""

def strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

def parse_json_lenient(text: str):
    # 1) strip code fences
    t = strip_code_fences(text)
    # 2) direct attempt
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # 3) extract first [...]-block
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            pass
    return []

def call_model(client, model: str, abstract: str, retries: int = 2) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"ABSTRACT:\n{abstract}\n\nJSON:"}
    ]
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=500, messages=messages
            )
            content = (resp.choices[0].message.content or "").strip()
            data = parse_json_lenient(content)
            return data
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    print(f"[warn] API error after retries: {last_err}")
    return []

def main():
    ap = argparse.ArgumentParser(description="Extract AD repurposing candidates from abstracts with OpenAI.")
    ap.add_argument("--input", required=True, help="Input CSV (must have 'abstract' column)")
    ap.add_argument("--output", required=True, help="Output CSV (column: raw_json)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--debug-out", default="data/openai_debug.txt", help="Write raw model replies here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first: export OPENAI_API_KEY='sk-...'")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.input).fillna("")
    if "abstract" not in df.columns:
        raise SystemExit("Input CSV must contain an 'abstract' column.")
    abstracts = [a for a in df["abstract"].tolist() if str(a).strip()]
    abstracts = abstracts[:args.limit]

    # conservative spend guard
    est_cost_per = 0.002
    max_items = min(len(abstracts), math.floor(args.max_cost / est_cost_per))
    abstracts = abstracts[:max_items]

    if args.verbose:
        print(f"[info] processing {len(abstracts)} abstracts with model={args.model} (cap≈${args.max_cost})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.debug_out) or ".", exist_ok=True)

    rows, debug_lines = [], []
    for i, abs_text in enumerate(abstracts, 1):
        print(f"Processing {i}/{len(abstracts)}...")
        data = call_model(client, args.model, abs_text)
        rows.append({"raw_json": json.dumps(data, ensure_ascii=False)})
        # save the raw for the first few to inspect
        if i <= 5:
            debug_lines.append(f"--- ITEM {i} RAW ---\n{json.dumps(data, ensure_ascii=False)}\n")
        time.sleep(0.3)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    with open(args.debug_out, "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines))

    print(f"✅ Saved results to {args.output}")
    print(f"🪵 Wrote debug samples to {args.debug_out}")

if __name__ == "__main__":
    main()
