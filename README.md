# UmichDatathon2026
This is our repository for University of Michigan Ross School of Business's Datathon 2026.

## Team Members
- Jeff Allen
- Kolbe Sussman
- Lina Al Rawahi
- Sachin Murthy
- Ushasree Jakilinki

## Tax Systems as Implicit Incentive Environments

### Problem Statement
Do long-run state tax structures act as implicit incentive systems that “train” the economic composition and demographic evolution of states over time?

### Goal
Build a fast, CSV-based pipeline that constructs:
- a canonical state-quarter panel
- tax “signal” features (incentive exposure vectors)
- adaptation outcomes (economic + demographic drift)
- a basic quasi-causal design table
- a thin-slice fixed-effects regression + one figure

### Running Scripts
All bash commands should be run from the repo / project directory root.

### Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data
Place raw CSVs in data/raw/..  
(file names are all text within quotations)
- "1. DatathonMasterStateTaxData_2004_2025Q2.csv"
- "2. DatathonMasterEconomicDataset_2004_2025Q2.csv"
- "3. (optional) DatathonAnnualDemographicsDataset.csv"

### To Run End-to-End Pipeline
```
PYTHONPATH=src python scripts/run_pipeline.py
```

### Output Files
- data/processed/panel_state_quarter.csv
- data/processed/tax_signals.csv
- data/processed/outcomes.csv
- data/processed/analysis_design.csv
- outputs/tables/panel_model_summary.csv
- outputs/tables/panel_model_summary_all.csv
- outputs/tables/directionality_significance.csv
- outputs/figures/gdp_per_capita_sample.png

### Findings & Analysis
For a summary of findings and analysis, please see the PPT, PDF, and .MD files located in the /docs folder.
