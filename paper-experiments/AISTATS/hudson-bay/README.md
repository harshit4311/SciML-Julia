# Hudson Bay lynx-hare case study

BNODE+MAP+HMC applied to the canonical Hudson Bay Company pelt-trading record
(1900–1920), the dataset Lotka–Volterra was originally invented to model. The
purpose is to demonstrate that the calibrated-band framework characterised on
synthetic data in §§2–3 of the paper transfers to a real ecological dataset
whose noise structure is unknown.

## Dataset

`data/lynx_hare.csv` — 21 annual observations:
- `year` 1900–1920
- `hare` snowshoe hare pelts (prey, thousands traded)
- `lynx` Canadian lynx pelts (predator, thousands traded)

Source: Hudson Bay Company trade records (Elton & Nicholson 1942, MacLulich 1937);
21-year subset as used in the Stan case study on Lotka–Volterra inference
(Carpenter et al.).

The data is what was **traded**, not what was alive — so it is noisy in a way the
synthetic experiments don't capture (trapping effort, climate, market, etc.).
That's intentional: it stresses the BNODE framework against unknown noise structure.

## Pipeline

Re-uses `../map-tests/lv_bnode_common.jl` for the model, MAP scheduler, NUTS
driver, posterior analysis, and plotting. Only adds:

1. CSV loading + normalisation (training-window mean) + time rescaling (years → [0, 7]).
2. Forecast-window-only coverage (the real-data calibration check).
3. A decision-relevance plot overlaying 5th-percentile forecasts on held-out data.

## Run

**Dev (~10 min after precompile):**
```bash
cd paper-experiments/AISTATS/hudson-bay
julia --project=../../.. hudson_bay.jl
```

**Paper (~24 hours):**
```bash
NSAMP=250 NADPT=250 MAXDEPTH=10 DEV_TOL=1e-8 \
MAP_PHASEA=6000 MAP_PHASEB=800 \
  julia --project=../../.. hudson_bay.jl
```

Use `caffeinate -i bash -c '…'` on macOS to prevent sleep during the long run.

## Outputs

- `hudson_bay_results.csv` — one row per run, capturing config + all metrics.
- `outputs/hudson_bay/`:
  - `point_fit.png` — MAP trajectory over the data.
  - `posterior_predictive.png` — full-trajectory 90% PP CI band.
  - `phase_space.png` — posterior cloud in (hare, lynx) space.
  - `chain_trace.png`, `autocor.png` — NUTS diagnostics.
  - `decision_relevance.png` — forecast quantiles framed for management decisions.

## What to report in the paper

After the paper-budget run, four numbers go into the case-study section:

1. **Forecast coverage** (hare, lynx) — target ~90% on held-out years.
2. **Posterior σ̂** — inferred noise scale (no `σ_true` exists for real data).
3. **NUTS diagnostics** — EBFMI, acceptance, ESS_min, R̂_max.
4. **MAP rel_err** — point forecast accuracy on held-out window.

Plus the decision-relevance plot as a paper figure.

## Honest caveats to surface in the paper

- No "true trajectory" exists on real data; calibration is checked against
  noisy observations only.
- The 21-year subset is small; the BNODE is over-parameterised relative to data
  size, so posterior diagnostics carry more weight than usual.
- Real noise may be heavy-tailed, heteroscedastic, or non-stationary —
  precisely the regimes §§2.4–2.5 stress-tested on synthetic data. Whatever
  calibration shows up here is implicitly testing the framework against the
  union of those regimes simultaneously.
