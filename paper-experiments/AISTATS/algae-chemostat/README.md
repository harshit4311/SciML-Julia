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

A single experiment is the natural analogue of the Hudson Bay single-trajectory
fit; the ten experiments together also support a pooled fit in the HIV style (one
shared dynamics network, per-experiment initial conditions).

## Which experiment to fit (important)

Experiment choice dominates everything. `periodicity.jl` ranks the 10 by
autocorrelation cycle-cleanliness; the winners are **C9** (140d, ~16-day period,
algae ACF 0.73) and **C8** (183d, ~16-day, 0.67). The long **C1** (374d) — the
obvious "flagship" pick — is actually **noise-dominated**: its day-to-day swings
have no coherent slow cycle (algae ACF 0.26; even a 3-week moving average reveals
nothing), so a smooth 2-D BNODE's MAP fit collapses to a *flat line at the mean* —
that genuinely is the pointwise-MSE optimum there, and no amount of MAP/NUTS budget
fixes it. On C9 the identical harness recovers the predator–prey cycle cleanly.

So the **default is `EXPT=9`**, fit on a short window (`DAY_MAX=50`, ≈3 cycles) to
keep the single-shooting solve to a few periods. Diagnostics that establish this —
`periodicity.jl` (ACF rank → `outputs/data_explore/acf_all.png`) and
`check_smoothing.jl` (is a slow cycle visible under the noise?) — are committed
alongside the fit script.

## Pipeline

`algae_chemostat.jl` re-uses `../map-tests/lv_bnode_common.jl` for the model,
two-phase MAP scheduler (`run_map`), NUTS driver (`run_nuts`), diagnostics
(`nuts_diagnostics`), posterior analysis (`analyze_posterior`), and result
aggregation — mirroring `hudson_bay.jl`. It adds only:

1. **Experiment selection** — one chemostat is pulled from the pooled 10-experiment
   CSV via the `EXPT` env var (default **C9**; see "Which experiment to fit").
2. **Window truncation** — `DAY_MAX` (default 50) keeps only the first N days, so
   the single-shooting solve spans a few cycles rather than many (pointwise MSE
   suffers phase-error collapse over many cycles).
3. **Missing-day handling** — measurement-days missing either channel (`NaN` in
   the source, empty in the CSV) are dropped so the Gaussian likelihood stays
   finite; the irregular surviving sample times are used directly as `saveat`.
4. CSV loading + per-channel normalisation (training-window mean) + time
   rescaling (days → ODE time `[0, TMAX]`).
5. Forecast-window posterior-predictive coverage + a decision-relevance plot.

**Architecture.** The synthetic-LV 2-32-32-32-2 tanh network (2274 params); no
capacity reduction (unlike the sparse HIV case).

**Time rescaling.** `TMAX` (default 7) maps the kept window to ODE time `[0,TMAX]`,
putting ~3 cycles of the 50-day C9 window on the synthetic-LV per-cycle scale.
Extending `DAY_MAX` to many cycles needs a proportionally larger `TMAX` and
re-introduces single-shooting phase strain (→ multiple-shooting / smoothing).

## Run

**Dev (default = C9 first ~50 days, ≈3 cycles):**
```bash
cd paper-experiments/AISTATS/algae-chemostat
julia --project=../../.. algae_chemostat.jl            # C9 short window
MAP_ONLY=1 julia --project=../../.. algae_chemostat.jl # just the MAP point fit
EXPT=8 julia --project=../../.. algae_chemostat.jl     # the other clean experiment
DAY_MAX=Inf TMAX=30 julia --project=../../.. algae_chemostat.jl  # full C9 (~9 cycles)
```

**Paper (long — many samples for ESS; keep the solver accurate):**
```bash
NSAMP=2500 NADPT=400 MAXDEPTH=8 DEV_TOL=1e-6 \
MAP_PHASEA=6000 MAP_PHASEB=800 \
  julia --project=../../.. algae_chemostat.jl
```
Use `caffeinate -i bash -c '…'` on macOS to prevent sleep during long runs.

## Performance & convergence (dev findings)

A dev-budget C9 fit works — the MAP/posterior-mean track the predator–prey cycle
and the 90% bands cover most held-out points (σ̂ ≈ 0.4, forecast coverage ≈ 0.73–0.80)
— **but NUTS mixes very poorly**: ESS_min ≈ 1.7, R̂_max ≈ 2.1, and these do **not**
improve with more warmup. The BNODE posterior is intrinsically curved/correlated
(step size collapses to ~3e-4, tree depth pins at the cap), so each draw is heavily
autocorrelated. This is a *sampling-geometry* problem, not a budget or warmup problem.

What was tried (per-iteration cost, same C9 short window):

| change | speed | inference | verdict |
|---|---|---|---|
| 2-32-32-32-2 → **2-16-16-2** (`HIDDEN`/`N_HIDDEN`) | 3.3× | R̂ 2.13→1.57, σ̂/coverage preserved | **adopted as default** |
| `MAXDEPTH=7` (vs 8) | +2.8× | σ̂/coverage ok, EBFMI 0.42→0.20 (borderline) | speed/quality knob |
| `DEV_TOL=1e-5` + `MAXDEPTH=6` | +6× | **broken** (EBFMI 0.11, coverage collapse) | rejected |
| more warmup (`NADPT` 50→160) | — | R̂/ESS unchanged | doesn't help |

Net: per-iteration cost is down ~9.5× from the original (2-32-32-32-2, depth-8)
baseline, so a long run is ~1.5–2 h, not ~17 h. **The remaining blocker is ESS, not
speed.** To get paper-trustworthy numbers, two non-speed fixes are still open:
1. **Many more samples** (`NSAMP`≈2500) to push ESS_min toward ~100.
2. **Multiple chains** (different `INIT_SEED`) for a meaningful R̂ — needs a small
   wrapper around the single-chain `run_nuts`.

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
- [x] Experiment selection: C1 noise-dominated → **default C9** (`periodicity.jl`).
- [x] `algae_chemostat.jl` single-experiment fit — validated on C9 short window:
      bands track the cycle, σ̂ ≈ 0.4, coverage ≈ 0.73–0.80.
- [x] Sampling speedup: smaller net (default) cuts per-iter cost 3.3× and improves
      mixing; full lever sweep documented above.
- [ ] **Convergence**: ESS_min ≈ 1.7, R̂ ≈ 2.1 — need a long run (`NSAMP`≈2500)
      and/or multi-chain (`run_nuts` is single-chain). This is the open blocker
      for paper numbers.
- [ ] Optional: extend to the full C9 record (multiple-shooting / light smoothing
      to beat single-shooting phase error over ~9 cycles).
- [ ] Optional: pooled multi-experiment mode in the HIV style.
