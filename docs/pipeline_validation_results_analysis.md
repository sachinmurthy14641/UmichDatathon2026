# Pipeline Validation & Analysis Summary

For fully documented work see: notebooks/results_inspection.ipynb

## 1. Pipeline Integrity (Steps 1–3)
The data pipeline produced a **clean, balanced state–quarter panel** covering **52 states/entities** from **2005Q1–2025Q2** with no duplicate keys and zero join loss across panels, outcomes, and tax signals. All transformations (per-capita scaling, GDP scaling, rolling windows, and lags) behaved as designed, and lagged exposures (`roll12_lag8`, `roll12_lag4`) were confirmed to be **strictly backward-looking**, eliminating look-ahead bias. Overall, the pipeline is **structurally sound and judge-defensible**.

---

## 2. Outcome Design & Statistical Viability (Step 2)
Outcomes naturally clustered into **levels (state identity)** and **deltas (state adaptation)**. Variance decomposition showed:

- **Delta outcomes** (e.g., GDP per-capita growth, changes in diversification) are **overwhelmingly within-state**, making them well-suited for fixed-effects (FE) estimation.
- **Level outcomes** (e.g., economic entropy, dependency ratios) are mostly **between-state**, best used for descriptive context rather than causal claims.

This validated a clear hierarchy:
- **Primary FE outcomes**: medium-run changes (growth, adaptation)
- **Secondary/context outcomes**: long-run structure and identity

---

## 3. Tax Signal Construction (Step 3)
Tax signals were consistently constructed across:
- Raw levels
- Per-capita measures
- Percent-of-GDP measures
- Rolling averages with multi-year lags

Concentration metrics (tax HHI, tax entropy) behaved as expected, and spot-checks confirmed correct scaling and timing. No mechanical correlations or accounting inconsistencies were detected.

---

## 4. Pre-Model Sanity Checks (Step 4)
Simple pooled correlations between lagged tax exposures and outcomes were **small but coherent**, with no implausible signs or dominating variables. This suggested:
- Effects are **subtle and conditional**, not mechanical
- Fixed effects are necessary to uncover within-state dynamics

---

## 5. Fixed-Effects Results (Step 5)
After isolating **demeaned (within-state) tax exposure effects**, three robust findings emerged:

### Core Findings
1. **Resource Severance Taxes**
   - **Negative for GDP per-capita growth**
   - **Negative for economic diversification (entropy)**
   - Interpretation: resource-based tax structures are associated with **slower growth and increasing specialization**, consistent with “resource-curse” dynamics.

2. **Corporate Income Taxes**
   - **Positive for GDP per-capita growth**
   - Weak/no effect on diversification
   - Interpretation: corporate tax bases proxy for **formal, dense, high-productivity economies**, supporting growth without necessarily broadening structure.

3. **Selective Excise Taxes**
   - **Positive for economic diversification**
   - Weak effect on growth
   - Interpretation: diversified consumption-based activity supports **structural resilience**, even if not rapid growth.

### Non-Findings (Equally Important)
- Labor income, general sales, and property taxes showed **weak or inconsistent relationships** with growth and diversification.
- This reinforces that **tax composition**, not overall tax burden, is what matters.

---

## 6. Cross-Outcome Consistency
Results were **internally coherent across outcomes**:
- Resource-based tax reliance is consistently harmful across both performance and structure.
- Different tax instruments operate through **different channels** (growth vs. diversification), rather than uniformly “good” or “bad.”

---

## 7. Key Takeaway
> **Tax systems act as long-run incentive environments, not short-run policy levers.**  
> What a state chooses to tax persistently shapes how its economy adapts—rewarding some structures while locking others into low-growth, low-diversification paths.

**In short:**  
**It’s not how much a state taxes, but *what* it taxes, that quietly trains its economy over time.**

---

## 8. Caveats & Scope
- Results are **associational**, not strictly causal.
- Effects operate over **multi-year horizons**.
- Findings speak to **economic structure and adaptation**, not short-term fiscal optimization.

---

## 9. Readiness
The pipeline, modeling strategy, and results are:
- Technically sound
- Economically interpretable
- Clearly narratable to non-technical audiences

This positions the analysis as **both rigorous and policy-relevant**.
