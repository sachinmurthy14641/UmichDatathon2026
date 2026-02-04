## Pipeline Overview

### Objective
This project tests whether long-run **state tax revenue structures** act as *implicit incentive systems* that shape the economic composition and demographic evolution of U.S. states over time.

The goal is not to estimate causal effects, but to assess **directionality, persistence, and plausibility** of long-horizon adaptation.

---

### Step 1: Ingest & Canonical Panel
Raw tax, economic, and demographic datasets are ingested and standardized into a quarterly `(state, period)` panel.

- Aligns all data sources to a common time index
- Uses quarterly economic data as the canonical backbone
- Annual demographic data is expanded to quarterly for alignment

**Output:**  
`data/processed/panel_state_quarter.csv`

---

### Step 2: Incentive Signal Construction
Tax revenue composition is transformed into **revealed incentive signals** using revenue shares rather than statutory rates.

- Tax codes are grouped into conceptual buckets (e.g., labor, consumption, capital, resource)
- Signals are smoothed using rolling windows and lagged to reflect persistence
- Concentration and diversity metrics (e.g., HHI, entropy) are computed

**Output:**  
`data/processed/tax_signals.csv`

---

### Step 3: Structural Outcome Construction
Economic and demographic **adaptation outcomes** are constructed to capture structural change rather than levels.

- Economic structure: sector shares, concentration, diversification
- Demographics: aging, dependency ratios, drift metrics
- Outcomes emphasize directional change over multi-quarter horizons

**Output:**  
`data/processed/outcomes.csv`

---

### Step 4: Quasi-Causal Design
A quasi-causal layer is added to test whether incentive exposure *precedes* adaptation.

- National shock windows (e.g., GFC, COVID) are defined
- States are grouped into exposure-based cohorts
- Event-style and divergence analyses are run
- Falsification and placebo checks are included

**Output:**  
`data/processed/analysis_design.csv`

---

### Step 5: Validation & Synthesis
Evidence is synthesized across multiple lenses:

- Fixed-effects panel regressions
- Shock-based divergence analysis
- Optional predictive validation (signals vs macro baselines)

Results are framed as **directional and probabilistic**, not causal.

**Outputs:**  
- Summary tables (`outputs/tables/`)
- Judge-ready figures (`outputs/figures/`)

---

### Interpretation
Findings are interpreted through a neutral framework that assesses whether the hypothesis is **probably true or probably false**, with explicit discussion of limitations and failure cases.

---

### Design Principles
- Revealed incentives over statutory intent
- Long-run persistence over short-run noise
- Falsifiability over overclaiming
- Modularity to support parallel development
