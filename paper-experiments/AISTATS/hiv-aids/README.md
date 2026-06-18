# ACTG 315 HIV / ART case study (pooled BNODE)

BNODE+MAP+HMC applied to a real clinical dataset: 46 HIV-positive adults followed
for 28 weeks after starting combination antiretroviral therapy (ART). The sibling
of the [Hudson Bay lynx-hare](../hudson-bay/) case study — same neural-ODE harness,
same MAP→NUTS machinery — but the modelling question moves from ecology to medicine,
and the data is *pooled across patients* rather than a single trajectory.

The purpose, as with Hudson Bay, is to show that the calibrated-band framework
characterised on synthetic Lotka–Volterra data in §§2–3 transfers to a real dataset
whose noise structure is unknown — here, additionally, with **sparse per-unit data
shared across a population**.

## The clinical framing

Not *"does this patient have AIDS?"* (diagnosis) but the **treatment-monitoring**
question:

> Given a patient's early ART response, will their viral load suppress and CD4
> recover — or are they heading toward treatment failure and AIDS progression?

The two state variables:
- **CD4 count** (cells/µL) — the "commander" helper T-cells HIV destroys. Recovery
  is good; CD4 < 200 is the AIDS-defining threshold.
- **Viral load** (HIV-1 RNA copies/mL, modelled in log10) — virus in the blood.
  Collapse toward the ~200-copy suppression target is the goal of ART.

The decision-relevant posterior outputs are the **5th-percentile CD4 trajectory**
(precautionary AIDS-risk lower bound) and the **95th-percentile viral-load
trajectory** (precautionary treatment-failure upper bound).

## Dataset

`data/actg315.csv` — 361 observations from 46 patients (4–10 draws each over days
0–196), columns:
- `id` — patient identifier (1–46)
- `day` — days since ART initiation
- `log10_rna` — HIV-1 viral load, log10 copies/mL (below-assay values imputed at
  50 copies → 1.699)
- `cd4` — CD4+ T-cell count, cells/µL

Source: AIDS Clinical Trials Group protocol **ACTG 315**, distributed by Dr. Hulin
Wu (UTHealth Houston Biostatistics). Regimen: zidovudine + lamivudine + ritonavir.
Re-fetch and re-tidy with `julia --project=../../.. fetch_data.jl` (downloads the
original whitespace file and rewrites the CSV; the CSV is committed for offline runs).

This is *traded-record-grade* noise like Hudson Bay: assay error, below-detection
censoring, irregular visit times, and real biological variation the pooled model
does not resolve — exactly the unknown-noise regime the framework is meant to stress.

## Why pooled (not 46 separate BNODEs)

8 observations cannot constrain a neural network with hundreds of weights — the
posterior would be prior-dominated and meaningless (the same reason you can't fit a
BNODE to 8 lynx-hare points). Instead **one shared network** learns the population
HIV-ART dynamics, and each patient contributes only their own initial condition:

```
Shared:     dz/dt = f_θ(z),   z = [CD4, log10 VL]      ← one neural net, all 46
Patient i:  z_i(0) = [CD4₀ⁱ, VL₀ⁱ]  → integrate → trajectory i
            all 46 noisy trajectories  →  one posterior p(θ | data)
```

~360 observations now constrain one network — tractable.

## Train / test protocols (`SPLIT` env var)

- **B (default — clinically realistic):** hold out **late time points**. Train on
  weeks 0–8 (`day ≤ SPLIT_DAY`, default 56) for every patient; forecast weeks 12–28.
  This *is* the clinical question: "given 8 weeks of response, predict the week-24
  outcome." Three patients with no post-week-8 draws inform θ but aren't scored.
- **A (strongest generalisation test):** hold out **patients entirely** (`SPLIT=A`,
  train patients 1..`N_TRAIN_PAT`=36, forecast the unseen 10 from their baseline).
  Tests whether the posterior predictive is calibrated for a brand-new patient.

## Pipeline

Re-uses `../map-tests/lv_bnode_common.jl` for the network architecture, the
two-phase MAP scheduler (`run_map`), the NUTS driver (`run_nuts`), the diagnostics
(`nuts_diagnostics`), and result aggregation. The only new machinery, defined inline
in `hiv_aids.jl`, is:

1. Per-patient data bundling — each patient integrates from its **own first
   observation** (3 patients start at day 2, not day 0) on a rescaled time axis
   (`day / DAY_SCALE`, default 28 → ODE time ∈ [0, 7], matching the synthetic LV scale).
2. A **pooled Gaussian log-posterior** `l(θ)` summing the likelihood over all
   training patients + an N(0,1) weight prior.
3. Forecast-window posterior-predictive coverage aggregated over held-out points.
4. The clinical **decision-relevance** plot (precautionary CD4/VL bounds vs thresholds).

## Run

**Dev (minutes):**
```bash
cd paper-experiments/AISTATS/hiv-aids
julia --project=../../.. fetch_data.jl      # once; CSV is already committed
julia --project=../../.. hiv_aids.jl
```

**Paper (long — the pooled solve costs ~N_patients× a single-trajectory run):**
```bash
NSAMP=250 NADPT=250 MAXDEPTH=8 DEV_TOL=1e-7 \
MAP_PHASEA=4000 MAP_PHASEB=600 \
  julia --project=../../.. hiv_aids.jl
```
Run the patient-holdout protocol with `SPLIT=A`. Use `caffeinate -i bash -c '…'`
on macOS to prevent sleep during long runs.

## Outputs

- `hiv_aids_results.csv` — one row per run (config + all metrics; appended).
- `outputs/hiv_aids/`:
  - `map_fit.png` — the **MAP pre-training** point-estimate trajectories vs data
    for representative patients (the deterministic fit, before NUTS).
  - `posterior_predictive.png` — 90% PP CD4 band with held-out points + AIDS line.
  - `phase_space.png` — posterior cloud in (CD4, log10 VL) space; suppression =
    drift toward high-CD4 / low-VL corner.
  - `decision_relevance.png` — 5th-pct CD4 vs AIDS threshold, 95th-pct VL vs
    suppression target. **The clinical figure.**
  - `coverage_breakdown.png` — **the misspecification diagnostic.** Per-patient
    CD4 forecast coverage and signed bias vs each patient's early CD4 response
    rate (slope over weeks 0–8). If slow responders cluster at low coverage /
    positive bias, the pooled model is over-predicting recovery for them —
    structural pooling error, motivating the hierarchical extension.
  - `per_patient_coverage.csv` — the per-patient table behind that figure
    (id, baseline CD4/VL, early slopes, #held-out, CD4/VL coverage, CD4 bias).
  - `chain_trace.png`, `autocor.png` — NUTS diagnostics.

## What to report in the paper

1. **Forecast coverage** (CD4, VL) on held-out points — target ≈ 90%.
2. **Posterior σ̂** — inferred (normalised) noise scale; no `σ_true` for real data.
3. **NUTS diagnostics** — EBFMI, acceptance, ESS_min, R̂_max, divergences.
4. **MAP forecast rel_err** — pooled point-forecast accuracy on the held-out window.
5. **Misspecification correlations** — `corr(early CD4 slope, per-patient coverage)`
   (expect > 0) and `corr(early CD4 slope, CD4 forecast bias)` (expect < 0). If the
   under-coverage concentrates on slow responders, that's the quantitative evidence
   that naive pooling is the limiting factor (the hierarchical-BNODE motivation).

Plus the decision-relevance and coverage-breakdown figures.

## Honest caveats to surface in the paper

- **Pooled, not hierarchical.** The model shares one dynamics network with
  patient-specific *initial conditions only* — it does not represent inter-patient
  variation in *response rate* (genetics, adherence, immune function). The principled
  fix is a hierarchical BNODE (population network + patient random effects); we note
  this as future work — overkill for a 4-page workshop paper.
- **Baseline-as-IC.** Each patient's (noisy) first observation is used as a fixed
  initial condition, injecting observation noise into the IC (same choice as the
  Hudson Bay harness).
- **No ground truth.** Calibration is checked against noisy held-out observations
  only; real assay noise is censored (below-detection imputation), heteroscedastic,
  and non-stationary — the union of regimes §§2.4–2.5 stress-tested synthetically.
- **Viral load in log10.** Modelling raw copies/mL spans 5 orders of magnitude;
  log10 is the clinical convention and keeps the network in range.
