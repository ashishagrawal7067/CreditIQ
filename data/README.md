# Dataset Instructions

The full Lending Club loan dataset contains records from 2007 to 2018Q4 with over 1.3 million historical loans.

### Source
Download the dataset from Kaggle:
- **Direct Link:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Primary file needed:** `accepted_2007_to_2018Q4.csv.gz` (or `accepted_2007_to_2018Q4.csv`)

### Directory Placement
Place the downloaded file into this `data/` directory or `data/raw/`:
```text
CreditIQ/
└── data/
    └── accepted_2007_to_2018Q4.csv.gz   (or .csv)
```
The notebooks and preprocessing pipelines automatically scan for either compressed (`.gz`) or extracted (`.csv`) versions across `data/` and `data/raw/`.

