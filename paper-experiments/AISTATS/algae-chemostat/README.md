# Rotifer–algae chemostat case study (Blasius et al. 2020)

BNODE+MAP+HMC applied to the **Blasius et al. (2020)** long-term rotifer–algae
chemostat predator–prey time series — the experimental realisation of the
Lotka–Volterra system. The third real-data case study alongside the
[Hudson Bay lynx–hare](../hudson-bay/) pelt record and the
[ACTG 315 HIV](../hiv-aids/) clinical study: same neural-ODE harness, same
MAP→NUTS machinery, a new noise regime.

Where Hudson Bay is a *single noisy trajectory* and HIV is *sparse data pooled
across patients*, this dataset is the opposite extreme: **dense, high-quality,
long-horizon predator–prey oscillations** measured under controlled lab
conditions across 50+ cycles and >300 predator generations. It is the cleanest
real test of whether the calibrated-band framework (characterised on synthetic
LV data in §§2–3) holds up where the *true* dynamics are genuinely
Lotka–Volterra-like but the observation noise is still unknown.

## The system

A planktonic predator–prey microcosm in a nitrogen-limited chemostat:

- **Algae** (prey) — the green alga *Monoraphidium minutum*, in 10⁶ cells/ml.
- **Rotifers** (predator) — *Brachionus calyciflorus*, in animals/ml.

These are the two Lotka–Volterra state variables for the BNODE:

```
dz/dt = f_θ(z),   z = [algae, rotifers]      ← one neural net
```

Each of the ten experiments differs in chemostat dilution rate / running length
(Extended Data Table 1 of the paper), giving a range of cycle regimes — some
showing the classic quarter-period predator–prey phase lag, others irregular or
damped oscillations.

## Dataset

`data/blasius_rotifer_algae.csv` — **1953 measurement-days** across **10
chemostat experiments** (`experiment` 1–10), columns:

- `experiment` — experiment id (1–10), corresponding to files C1–C10 in the source
- `day` — measurement time (days since start)
- `algae` — prey density, 10⁶ cells/ml  *(LV state 1)*
- `rotifers` — predator density, animals/ml  *(LV state 2)*
- `egg_ratio` — rotifer eggs per female (predator reproductive state)
- `eggs` — egg density, per ml
- `dead` — dead rotifers, per ml
- `medium_N` — external medium nitrogen, µmol N / l (chemostat inflow)

Missing measurements (coded `NaN` in the source) are left **empty** in the CSV.
Experiment lengths vary from ~85 days (C5) to ~374 days (C1, the flagship long
run). The `algae`/`rotifers` columns are the modelling targets; the remaining
columns are retained for reference and possible extensions (e.g. egg-ratio as a
trait-dynamics covariate).

Source: Blasius B, Rudolf L, Weithoff G, Gaedke U, Fussmann G.F. *Long-term
cyclic persistence in an experimental predator–prey system.* Nature **577**,
226–230 (2020). Data on figshare (CC BY 4.0), DOI
[10.6084/m9.figshare.10045976](https://doi.org/10.6084/m9.figshare.10045976).
Re-fetch and re-tidy with `julia --project=../../.. fetch_data.jl` (downloads
the ten original files and rewrites the CSV; the CSV is committed for offline
runs).

This is *experiment-grade* noise rather than traded-record noise: counting
error, sub-sampling variance, and irregular sampling — much lower than Hudson
Bay or HIV, but still an unknown structure the BNODE must calibrate against. Its
value is the long horizon: enough cycles to actually constrain a neural ODE and
to forecast many periods ahead, which the 21-point lynx–hare record cannot.

## Why this dataset (vs the two siblings)

- **Hudson Bay:** real but short (21 pts), one trajectory, heavy trade noise.
- **HIV ACTG-315:** real, sparse per-unit, pooled across a population, censored.
- **This:** real, *dense and long*, near-textbook LV dynamics, low-but-unknown
  noise — the case where the framework should perform *best*, completing the
  spectrum from "framework stress-tested" to "framework validated".

A single long experiment (e.g. C1) is the natural analogue of the Hudson Bay
single-trajectory fit; the ten experiments together also support a pooled fit in
the HIV style (one shared dynamics network, per-experiment initial conditions).

## Status / next steps

- [x] Folder + data fetch (`fetch_data.jl`, `data/blasius_rotifer_algae.csv`).
- [ ] `algae_chemostat.jl` fit script — to re-use `../map-tests/lv_bnode_common.jl`
      (model, two-phase MAP scheduler, NUTS driver, diagnostics, plotting),
      mirroring `hudson_bay.jl` / `hiv_aids.jl`, with:
  1. CSV loading + per-experiment selection, normalisation (training-window
     mean), and time rescaling (days → ODE time on the synthetic-LV scale).
  2. Train-on-early / forecast-late split (and optionally a pooled multi-experiment
     mode in the HIV style).
  3. Forecast-window posterior-predictive coverage + decision-relevance plot.

(The fit script is intentionally not written yet — this commit just establishes
the folder and the reproducible data fetch, matching how the sibling folders were
seeded.)
