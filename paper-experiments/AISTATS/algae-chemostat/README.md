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

## Pipeline

`algae_chemostat.jl` re-uses `../map-tests/lv_bnode_common.jl` for the model,
two-phase MAP scheduler (`run_map`), NUTS driver (`run_nuts`), diagnostics
(`nuts_diagnostics`), posterior analysis (`analyze_posterior`), and result
aggregation — mirroring `hudson_bay.jl`. It adds only:

1. **Experiment selection** — one chemostat is pulled from the pooled 10-experiment
   CSV via the `EXPT` env var (default C1, the ~374-day flagship run; the
   single-trajectory analogue of the Hudson Bay fit).
2. **Missing-day handling** — measurement-days missing either channel (`NaN` in
   the source, empty in the CSV) are dropped so the Gaussian likelihood stays
   finite; the irregular surviving sample times are used directly as `saveat`.
3. CSV loading + per-channel normalisation (training-window mean) + time
   rescaling (days → ODE time `[0, TMAX]`).
4. Forecast-window posterior-predictive coverage + a decision-relevance plot.

**Architecture.** The synthetic-LV 2-32-32-32-2 tanh network (2274 params) — C1
has ~360 usable days, comparable to the synthetic 200-point fit, so unlike the
sparse HIV case no capacity reduction is needed.

**Time rescaling.** Unlike the 21-point lynx–hare record (≈2 cycles → `[0,7]`),
C1 contains many predator–prey cycles. `TMAX` (default 30) rescales the record so
per-cycle resolution stays in the range the network was sized for on synthetic LV;
it is the main knob to tune (lower = compress more cycles = harder for the net;
higher = longer integration = slower/harder NUTS).

## Run

**Dev (~20–40 min after precompile):**
```bash
cd paper-experiments/AISTATS/algae-chemostat
julia --project=../../.. algae_chemostat.jl       # C1 by default
EXPT=6 julia --project=../../.. algae_chemostat.jl # a different chemostat
```

**Paper (long — relax tol, deepen tree, more MAP):**
```bash
NSAMP=250 NADPT=250 MAXDEPTH=10 DEV_TOL=1e-8 \
MAP_PHASEA=6000 MAP_PHASEB=800 \
  julia --project=../../.. algae_chemostat.jl
```
Use `caffeinate -i bash -c '…'` on macOS to prevent sleep during long runs.

## Outputs

- `algae_chemostat_results.csv` — one row per run (config + all metrics; appended).
- `outputs/algae_chemostat/`: **`posterior_faceted.png`** (the legible PP figure
  — data points drawn on top of the 90% band, split by channel × time window,
  `NWIN` rows; the right one to read for a dense, many-cycle run),
  `posterior_predictive.png` (the shared single-axis PP band),
  `point_fit.png` (MAP), `phase_space.png` (posterior cloud in algae–rotifer
  space), `decision_relevance.png` (forecast quantiles on held-out data), and
  `chain_trace.png` / `autocor.png` (NUTS diagnostics).

`plot_data.jl` writes the raw-data views to `outputs/data_explore/`:
`C1_faceted.png` (the same time-window faceting, no posterior — the clear look at
the raw cycles), plus `timeseries_all.png` / `phase_all.png` overviews of all 10
experiments. `NWIN` (default 5) sets the number of time-window rows in both the
raw and posterior faceted plots.

## Status / next steps

- [x] Folder + data fetch (`fetch_data.jl`, `data/blasius_rotifer_algae.csv`).
- [x] Exploratory plots (`plot_data.jl` → `outputs/data_explore/`).
- [x] `algae_chemostat.jl` single-experiment fit (verified end-to-end on a smoke
      budget; needs a full MAP+NUTS budget for paper numbers).
- [ ] Decide noise handling after inspecting a full-budget run (the raw series is
      heavy-tailed/spiky; if the posterior visibly noise-chases, revisit light
      smoothing or a larger `TMAX`).
- [ ] Optional: pooled multi-experiment mode in the HIV style.
