from __future__ import annotations

import os
import re
import csv
import json
from collections import Counter
from typing import Literal, List

import pandas as pd
import asyncio
import pydantic
from semlib import Session, OnDiskCache
from tqdm import tqdm


session = Session(
    cache=OnDiskCache("cache.db"),
    model="openai/gpt-4.1-mini", 
    max_concurrency=3            
)
NUMERIC_START = re.compile(
    r"""^\s*(
        \d{1,3}([-/]\d{1,3})?(\s*(day|days|week|weeks|month|months|year|years|y/o|yo))?  # 2 days, 44 y/o, 3-4 weeks
        |\d{1,2}/\d{1,2}/\d{2,4}                                                       # 7/31/2008
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

CSV_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv"

#load & group visits
START_SECTIONS = {"CC","GENHX","HPI","PASTMEDICALHX","MEDICATIONS","FAM/SOCHX","ALLERGY","ROS","PHYSICAL","ASSESSMENT","PLAN"}

'''
CC           → reason for visit
GENHX        → general history
MEDICATIONS  → meds list
PASTMEDICALHX → past conditions
'''

def load_grouped_visits(limit_visits: int = 1200, max_chars_per_visit: int = 6000) -> list[dict]:
    df = pd.read_csv(CSV_URL)

    # column mapping (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    id_col  = cols.get("id")
    sh_col  = cols.get("section_header") or "section_header"
    st_col  = cols.get("section_text") or "section_text"
    dlg_col = cols.get("dialogue") or "dialogue"

    visits: list[dict] = []
    for idx, r in df.iterrows():
        section_text = str(r.get(st_col, "") or "")
        dialogue     = str(r.get(dlg_col, "") or "")

        # keep prompts reasonable for cost
        combined_len = len(section_text) + len(dialogue)
        if combined_len > max_chars_per_visit:
            take_sec = max_chars_per_visit // 2
            take_dlg = max_chars_per_visit - take_sec
            section_text = section_text[:take_sec]
            dialogue     = dialogue[:take_dlg]

        visit_id = int(r[id_col]) if id_col in r and pd.notna(r[id_col]) else idx

        visits.append({
            "visit_id": visit_id,
            "section_text": section_text.strip(),
            "dialogue": dialogue.strip(),
        })

        if len(visits) >= limit_visits:
            break

    # If limit_visits > number of rows, we just return all rows
    return visits


#Typed output model, set one DICTIONARY per visit
class VisitSummary(pydantic.BaseModel):
    visit_id: int
    chief_complaint: str | None
    family_illnesses: List[str]
    medications: List[str]
    small_talk: bool                         # useful extra: did small talk occur?
    mental_health_symptoms: List[str]        # useful extra: stress/anxiety/etc.
    past_medical_conditions: list[str]
    past_surgeries: list[str]
    symptoms: list[str]
    disease_category: str

    #3 Prompt template
SUMMARY_INSTRUCTIONS = """
You are a clinical scribe. Extract a compact JSON object with EXACTLY these fields:
- chief_complaint (string or null): patient's main reason for the visit. Prefer CC/GENHX wording; otherwise infer from dialogue. Keep it short.
- family_illnesses (list[str]): illnesses explicitly mentioned in family history (e.g., "mother had heart disease", "maternal uncles had polio").
- medications (list[str]): medications the patient STATES they currently take at home. EXCLUDE meds administered during this visit in the ER/clinic (e.g., "morphine given").
- small_talk (bool): true only if there is at least one short exchange (≥2 lines)
  on a non-medical topic (greetings alone do NOT count).
- mental_health_symptoms (list[str]): short phrases like "stress", "anxiety", "depression", "insomnia", "panic".
- past_medical_conditions (list[str]): prior or chronic conditions (from PASTMEDICALHX/GENHX/ROS when used as history), e.g., "hypertension", "asthma", "osteoporosis".
- past_surgeries (list[str]): prior surgeries/procedures, e.g., "appendectomy", "C-section", "pacemaker placement".
- symptoms (list[str]): CURRENT visit symptoms/signs mentioned by the patient (e.g., "headache", "itching", "nausea", "weakness", "rash", "pain"). DO NOT include chronic conditions here.
- disease_category (string): choose ONE best high-level label for this visit from:
  ["cardiovascular","respiratory","neurological","dermatological","gastrointestinal",
   "musculoskeletal","endocrine","psychiatric","genitourinary","infectious",
   "injury/trauma","oncology","obstetrics/gynecology","toxicology","unsure"].

Rules:
- Use BOTH section_text and dialogue. Prefer explicit mentions over inference.
- Be conservative. If a field is absent, use null for the string, [] for lists, and false for small_talk when unclear.
- For medications: include only active home meds the patient reports (words like "take", "on", "uses", "daily", "prn"). EXCLUDE meds "given", "ordered", "administered" in the ER/clinic unless the patient says they take them at home.
- Keep items concise, normalized, and de-duplicated (e.g., "tylenol" not "Tylenol p.r.n. 500 mg").
- Output JSON ONLY (no commentary).
""".strip()



def extraction_template(v: dict) -> str:
    return f"""
{SUMMARY_INSTRUCTIONS}

<visit_id>{v["visit_id"]}</visit_id>

<section_text>
{v["section_text"]}
</section_text>

<dialogue>
{v["dialogue"]}
</dialogue>
""".strip()


#Run one LLM call per visit

import traceback
import json

import asyncio, random
from typing import Any

async def _extract_one_with_retry(visit: dict, attempts: int = 6, base_delay: float = 0.6) -> Any:
    """
    Calls the model for a single visit with exponential backoff on rate limits.
    """
    for i in range(attempts):
        try:
            return await session.prompt(
                extraction_template(visit),
                return_type=VisitSummary,
            )
        except Exception as e:
            msg = str(e).lower()
            # backoff on rate limits / transient network errors
            if "rate limit" in msg or "429" in msg or "timeout" in msg:
                sleep_s = base_delay * (2 ** i) + random.uniform(0, 0.2)
                await asyncio.sleep(sleep_s)
                continue
            # if it's some other error, re-raise immediately
            raise
    # ran out of retries
    raise RuntimeError(f"Failed after retries for visit_id={visit.get('visit_id')}")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

async def run_extraction(visits: list[dict]) -> list[VisitSummary]:
    """
    Process in batches to respect TPM:
    - small batch size
    - low concurrency inside each batch
    - short pause between batches
    """
    results: list[VisitSummary] = []
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))     # try 50; tune as needed
    PAUSE_SECS = float(os.environ.get("BATCH_PAUSE", "3.0")) # short pause between batches

    for bidx, batch in enumerate(chunked(visits, BATCH_SIZE), start=1):
        # schedule tasks for this batch
        tasks = [asyncio.create_task(_extract_one_with_retry(v)) for v in batch]
        # gather (concurrency is limited by session.max_concurrency)
        batch_out = await asyncio.gather(*tasks)
        results.extend(batch_out)
        print(f"Batch {bidx}: processed {len(batch)} visits (total {len(results)}/{len(visits)})")
        # small pause to keep well below TPM
        await asyncio.sleep(PAUSE_SECS)

    return results

#Non-semantic post-processing & reporting
CANON = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "benadryl": "diphenhydramine",
    "lexapro": "escitalopram",
    "mavik": "trandolapril",
    "advair": "fluticasone/salmeterol",
}

DOSE_TERMS = r"\b(\d+(\.\d+)?\s*(mg|mcg|g|ml|units?)|prn|p\.r\.n\.|tid|bid|qid|qhs|q\d+h)\b"

def normalize_meds(meds: List[str]) -> List[str]:
    out = []
    for m in meds:
        s = m.lower()
        s = re.sub(DOSE_TERMS, "", s)
        s = re.sub(r"[^\w\s/\'-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip(" -_/")
        if s:
            out.append(CANON.get(s, s))
    return out



def main():
    import os,sys

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: Set OPENAI_API_KEY first (export OPENAI_API_KEY=\"sk-...\")")

    print("Loading & grouping rows into visits ...")
    visits = load_grouped_visits(limit_visits=1200, max_chars_per_visit=6000)
    print(f"Prepared {len(visits)} visits")
    
    env_limit = os.environ.get("LIMIT_VISITS")
    if env_limit:
        try:
            limit = int(env_limit)
            visits = visits[:limit]
            print(f"Using LIMIT_VISITS={limit}")
        except ValueError:
            pass
            
    # --- optional dry-run flag ---
    DRY_RUN = bool(int(os.environ.get("DRY_RUN", "0")))  # default off
    if DRY_RUN:
        print("=== DRY RUN PREVIEW ===")
        print(f"Total grouped visits: {len(visits)}\n")
        for i, v in enumerate(visits[:2], 1):  # show first 2 visits
            print(f"--- Visit {i} ---")
            print("Section_text snippet:\n", v["section_text"][:800], "...\n")
            print("Dialogue snippet:\n", v["dialogue"][:800], "...\n")
            print("=" * 80)
        sys.exit(0)


    print("Extracting structured summaries (this costs a little) ...")
    visit_summaries = asyncio.run(run_extraction(visits))

    # Convert to DataFrame (one DICT per visit)
    df = pd.DataFrame([vs.model_dump() for vs in visit_summaries]).sort_values("visit_id")
    df.to_csv("visit_summaries.csv", index=False)

    from ast import literal_eval

    # Helper to safely parse list columns whether stored as list or string
    def as_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.startswith("["):
            try:
                return literal_eval(x)
            except Exception:
                return []
        return []


    # --- Q1: Proportion with mental health mentions
    df["has_mh"] = df["mental_health_symptoms"].apply(as_list).apply(lambda xs: len(xs) > 0)
    prop_mh = df["has_mh"].mean() * 100
    print(f"\nQ1: Proportion of visits with mental health mentions: {prop_mh:.1f}%")

    # --- Q4: Correlation between number of chronic conditions and number of meds
    df["num_conditions"] = df["past_medical_conditions"].apply(as_list).apply(len)
    df["num_meds"] = df["medications"].apply(as_list).apply(len)
    corr = df[["num_conditions", "num_meds"]].corr().iloc[0,1]
    print(f"Q4: Correlation (conditions vs. medications): {corr:.2f}")

    # --- Q8: Small talk in chronic vs non-chronic (proxy for follow-ups)
    df["chronic_visit"] = df["past_medical_conditions"].apply(as_list).apply(lambda xs: len(xs) > 0)
    st_by_chronic = df.groupby("chronic_visit")["small_talk"].mean().to_dict()
    print(f"Q8: Small talk rate by chronic_visit={st_by_chronic}")

    # --- Q11: Family heart history vs. disease category distribution
    def has_family_heart(xs):
        xs = as_list(xs)
        return any("heart" in s.lower() for s in xs)

    df["family_heart"] = df["family_illnesses"].apply(has_family_heart)
    cross = pd.crosstab(df["family_heart"], df["disease_category"], normalize="index") * 100
    print("Q11: Family heart history → disease category (% within group):")
    print(cross.round(1))

    print("Saved visit_summaries.csv")
    # --- Top-5 medications (normalized) ---
    from collections import Counter

    def as_list(x):
        # reuse your helper if it's already defined above;
        # otherwise keep this small version here
        if isinstance(x, list): return x
        from ast import literal_eval
        if isinstance(x, str) and x.startswith("["):
            try: return literal_eval(x)
            except Exception: return []
        return []

    # normalize + flatten
    all_meds = []
    for meds in df["medications"].apply(as_list):
        all_meds.extend(normalize_meds(meds))

    # (optional) filter generic placeholders
    GENERIC_SKIP = {"inhaler"}
    all_meds = [m for m in all_meds if m not in GENERIC_SKIP and m != ""]

    top5 = Counter(all_meds).most_common(5)
    pd.DataFrame(top5, columns=["medication", "count"]).to_csv("top5_medications.csv", index=False)
    print("Saved top5_medications.csv")

    # also include in the Q/A rows as Q0
    if top5:
        top5_str = "; ".join(f"{name} ({cnt})" for name, cnt in top5)
    else:
        top5_str = "No medications found."

    # --- Build clear question/answer style output ---
    qa_rows = []
    qa_rows.append({
        "question": "Q0: What are the top five most-mentioned medications?",
        "answer": top5_str
    })
    # Q1
    qa_rows.append({
        "question": "Q1: What proportion of visits include mental health mentions?",
        "answer": f"{prop_mh:.1f}% of analyzed visits."
    })

    # Q4
    qa_rows.append({
        "question": "Q4: How strongly are chronic conditions related to number of medications?",
        "answer": f"Correlation coefficient = {corr:.2f} (positive means more conditions → more meds)."
    })

    # Q8
    st_nonchronic = round(float(st_by_chronic.get(False, 0)), 2)
    st_chronic = round(float(st_by_chronic.get(True, 0)), 2)
    qa_rows.append({
        "question": "Q8: Do chronic patients engage in more small talk?",
        "answer": f"Chronic visits: {st_chronic*100:.1f}%, Non-chronic visits: {st_nonchronic*100:.1f}%"
    })

    # Q11
    qa_rows.append({
        "question": "Q11: How does family heart history relate to disease category?",
        "answer": "Distribution by category (percent of each group):"
    })
    for heart_status, row in cross.round(1).iterrows():
        dist = ", ".join(f"{cat}: {pct:.1f}%" for cat, pct in row.items())
        qa_rows.append({
            "question": f"   • Family heart={heart_status}",
            "answer": dist
        })

    # Save pretty CSV
    pd.DataFrame(qa_rows).to_csv("analysis_summary.csv", index=False)
    print("\nSaved analysis_summary.csv")

    # --- Also print nicely to console ---
    print("\n=== Analysis Summary ===")
    for row in qa_rows:
        print(f"{row['question']}\nA: {row['answer']}\n")

if __name__ == "__main__":
    main()
