#!/usr/bin/env python3
"""
fetch_pubmed.py
- Reads a PubMed query from --query or --query-file
- Uses NCBI Entrez (Biopython) to find PMIDs and fetch titles/abstracts
- Writes CSV to --out (default: data/abstracts.csv)

Usage:
  export NCBI_EMAIL="you@ucsd.edu"   # required by NCBI
  export NCBI_API_KEY="optional_key" # optional but helpful
  python fetch_pubmed.py --query-file query_examples.txt --retmax 500 --out data/abstracts.csv
"""

import os
import time
import argparse
from typing import List, Dict

import pandas as pd
from Bio import Entrez
from tqdm import tqdm


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def esearch_ids(query: str, retmax: int = 500) -> List[str]:
    """Search PubMed and return up to retmax PMIDs."""
    h = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    rec = Entrez.read(h)
    h.close()
    return rec.get("IdList", [])


def efetch_abstracts(pmids: List[str]) -> List[Dict]:
    """Fetch titles + abstracts for a list of PMIDs."""
    out = []
    for chunk in tqdm(list(chunked(pmids, 100)), desc="Fetching abstracts"):
        h = Entrez.efetch(db="pubmed", id=",".join(chunk), rettype="abstract", retmode="xml")
        rec = Entrez.read(h)
        h.close()
        for art in rec.get("PubmedArticle", []):
            pmid = str(art["MedlineCitation"]["PMID"])
            article = art["MedlineCitation"]["Article"]
            title = str(article.get("ArticleTitle", ""))

            abstract = ""
            ab = article.get("Abstract")
            if ab and "AbstractText" in ab:
                # AbstractText can be list-like with labeled sections
                parts = [str(x) for x in ab["AbstractText"]]
                abstract = "\n".join(parts)

            out.append({"pmid": pmid, "title": title, "abstract": abstract})

        # gentle throttle to be kind to NCBI
        time.sleep(0.34)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="PubMed query string (overrides --query-file)")
    ap.add_argument("--query-file", help="File containing PubMed query")
    ap.add_argument("--retmax", type=int, default=500, help="Max PMIDs to retrieve")
    ap.add_argument("--out", default="data/abstracts.csv", help="Output CSV path")
    args = ap.parse_args()

    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise SystemExit("ERROR: Please set NCBI_EMAIL environment variable (required by NCBI).")
    Entrez.email = email

    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    if args.query:
        query = args.query
    elif args.query_file:
        with open(args.query_file, "r") as f:
            query = f.read().strip()
    else:
        raise SystemExit("Provide --query or --query-file")

    print(f"Running PubMed search (retmax={args.retmax})...")
    pmids = esearch_ids(query, retmax=args.retmax)
    if not pmids:
        raise SystemExit("No PMIDs returned. Try loosening the query or increasing --retmax.")

    print(f"Found {len(pmids)} PMIDs. Fetching abstracts...")
    rows = efetch_abstracts(pmids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"✅ Wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
