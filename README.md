# AI-Assisted-Disease-Ontology-Standardization-Automated-Tagging
DSC 180A B23 Capstone project

# Medical Conversation Structured Extraction & Analysis

This project parses a clinical dialogue dataset (MTS-Dialog) and uses a Large Language Model (LLM) to extract structured visit summaries. It then performs exploratory analyses such as medication frequency, small talk prevalence, and relationships between family history and disease categories.

The final output includes:
- `visit_summaries.csv`: One structured summary per visit.
- `analysis_summary.csv`: Results for several higher-level research questions.
- `top5_medications.csv`: Frequency counts of normalized medication names.

---

## 1. Dataset Access

This project uses the **MTS-Dialog Training Set dataset**, hosted at:
https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv


No manual download is required. The code pulls the dataset directly via `pandas.read_csv`.

If needed, you can download it manually and replace the URL in the script.

---

## 2. Software Requirements

### Python Version
Python 3.12

### Recommended Environment Setup (conda)
conda create -n dsc180 python=3.12
conda activate dsc180


### Install Dependencies
pip install pandas pydantic tqdm semlib litellm

You must also have an OpenAI-compatible API key set in your environment:
export OPENAI_API_KEY="your_api_key_here"


---

## 3. Running the Code

### File Overview
| File | Purpose |
|------|---------|
| `mts_extract.py` | Extracts structured summaries and runs analysis. |
| `visit_summaries.csv` | Output structured dataset of visits. |
| `analysis_summary.csv` | Output results of key research questions. |
| `top5_medications.csv` | Medication frequency counts. |

---

### **A) Dry Run Preview (No API Calls, Safe Test)**
DRY_RUN=1 python mts_extract.py
This prints a preview of two grouped visits so you can verify the data grouping is correct.

---

### **B) Run on a Subset (Recommended for Cost Control)**
LIMIT_VISITS=50 python mts_extract.py


---

### **C) Run on Full Dataset**
*(Expect higher token usage & longer runtime)*

LIMIT_VISITS=1200 python mts_extract.py

If you encounter **rate limits**, reduce concurrency:
Inside the script:
```python
session = Session(..., max_concurrency=3)

### 7. Citation
If using the dataset, cite the MTS-Dialog authors:

A. Bachaa et al. "MTS-Dialog: Medical Transcripts Structured Dataset", 2021.
