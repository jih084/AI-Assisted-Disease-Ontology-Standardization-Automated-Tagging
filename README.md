# AI-Assisted-Disease-Ontology-Standardization-Automated-Tagging
DSC 180A B23 Capstone project
This repository contains two integrated components developed for the DSC 180A B23 Capstone:

1. **Medical Conversation Structured Extraction (MTS-Dialog Component)**  
   Tools for parsing raw clinical dialogues and generating structured visit summaries.

2. **Alzheimer’s Drug Repurposing Literature Mining (Current Component)**  
   A pipeline that retrieves PubMed abstracts related to Alzheimer’s disease and automatically extracts candidate repurposed drugs, their mechanisms, stance of evidence (supportive vs inconclusive), and key supporting sentences.

---

## 1. Data Access

### A) PubMed Abstracts (Current Component)
Retrieved automatically via **NCBI E-utilities API** (Biopython) using the query:
("Alzheimer Disease"[MeSH Major Topic] OR Alzheimer*[Title/Abstract])
AND ("Drug Repositioning"[MeSH Terms] OR repurpos* OR reposition* OR "drug rediscovery")
AND hasabstract[text]
AND english[Language]
AND (1980:3000[pdat])


Output stored as:
data/abstracts.csv

### B) MTS-Dialog Dataset (Previous Component)
Source:  
https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv

Loaded directly via `pandas.read_csv`.

---

## 2. Software Requirements

### Python Version
Python 3.12

### Recommended Environment Setup
```bash
conda create -n dsc180 python=3.12 -y
conda activate dsc180
Install Dependencies
pip install pandas tqdm biopython openai
(Optional NER model):
pip install scispacy spacy
Required Environment Variables
export NCBI_EMAIL="your_email@ucsd.edu"
export OPENAI_API_KEY="your_api_key_here"
```
### 3. Running the Code

#### Step 1 — Fetch PubMed Abstracts
python fetch_pubmed.py --query-file query_examples.txt --retmax 500 --out data/abstracts.csv
#### Step 2 — Extract Drug Candidates (Cost-Controlled)
python openai_extract_candidates.py \
  --input data/abstracts.csv \
  --output data/openai_candidates_raw.csv \
  --limit 500 \
  --max-cost 2.00 \
  --model gpt-4o-mini \
  --verbose
#### Step 3 — Flatten & Summarize Results
python flatten_openai_json.py \
  --in data/openai_candidates_raw.csv \
  --out data/openai_candidates_table.csv \
  --summary data/candidate_summary.csv

### 4.Output Files
| File                                                                                         | Description                                        |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `data/abstracts.csv`                                                                         | PubMed PMIDs, titles, and abstracts                |
| `openai_candidates_raw.csv`                                                                  | One extracted JSON result per abstract             |
| `openai_candidates_table.csv`                                                                | Tidy table with drug, mechanism, stance, evidence  |
| `candidate_summary.csv`                                                                      | Aggregated candidate hit counts by stance          |
| (MTS-Dialog Component) `visit_summaries.csv`, `analysis_summary.csv`, `top5_medications.csv` | Structured dialogue summaries and analysis results |

#### Citation Guidance
If using the MTS-Dialog dataset:
A. Bachaa et al. "MTS-Dialog: Medical Transcripts Structured Dataset", 2021.
If using PubMed:
National Center for Biotechnology Information (NCBI). PubMed database.
If reporting drug extraction results:
OpenAI API, 2025.
