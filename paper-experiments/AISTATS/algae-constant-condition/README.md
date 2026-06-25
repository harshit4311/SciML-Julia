# Constant-condition chemostat fits + residual diagnostics

A follow-up to [`../algae-chemostat/`](../algae-chemostat/) prompted by a structural
discovery: **C9 (our previous showcase) is an externally *forced* experiment**, and an
autonomous neural ODE cannot fit a forced system. This folder moves the headline
real-data fit to a **constant-condition** experiment (where the autonomous BNODE is
well-specified) and adds a **residual-variance / stationarity diagnostic** to make the
misspecification visible.

## The discovery: C8/C9 are forced, C1–C7/C10 are not

The pooled CSV's `medium_N` column (external nutrient inflow Nᵢₙ, µmol N/l) is the
chemostat's control knob. Checking it over time per experiment:

| experiment | Nᵢₙ | regime |
|---|---|---|
| **C1–C5, C10** | constant **80** | constant-condition — autonomous BNODE **well-specified** |
| C6, C7 | constant **160** | constant-condition |
| **C8, C9** | **0 ↔ 160 square wave (~8-day period)** | **externally forced** — autonomous BNODE **misspecified** |

`plot_forcing.jl` renders `outputs/forcing.png` showing the square wave on C8/C9 vs the
flat line elsewhere.

**Why this matters.** Our model is autonomous, `ẋ = f_θ(x)`. A time-dependent driver
`Nᵢₙ(t)` cannot be represented by an autonomous field, so on C9 there is a hard accuracy
ceiling (~52% relErr) that is *insensitive to network width, MAP iters, segment length,
and prior strength* — all of which we swept and none of which moved it. The "cleanest
ACF" that made us pick C9 was the **forcing signal**, not intrinsic dynamics.

## Expectation-setting for C1–C4

Constant-condition ≠ clean cycle. Removing the forcing, the *intrinsic* predator–prey
cycle in C1–C4 is faint (ACF ≈ 0.23–0.35 vs C9's forced 0.73). So expect the autonomous
BNODE to be **well-specified but lower-amplitude / noisier** here — that's honest. (C6/C7/C10
have stronger intrinsic cycles, ~0.4, if a cleaner constant-condition showcase is wanted.)

## Two questions this folder answers

1. **Does the autonomous BNODE fit a *constant-condition* experiment better than the
   forced C9?** (well-specification check) — start with **C1**.
2. **Is the residual variance stationary?** For a well-specified model with stationary
   noise, residuals `actual − point_estimate` should have roughly constant variance over
   time. On forced/misspecified C9 we expect the opposite: residual variance that grows in
   the forecast window and clusters around the Nᵢₙ switch times. `residual_diag.jl`
   quantifies and plots this.

## How it reuses the existing harness (no duplication)

Fitting is done by the existing `../algae-chemostat/algae_chemostat.jl` (select the
experiment with `EXPT`). Each MAP run now also writes
`../algae-chemostat/outputs/algae_chemostat/map_prediction.csv` (day, actual & predicted
per channel, train/forecast split). This folder's `residual_diag.jl` reads that CSV and
produces the variance/stationarity diagnostics — no model code is duplicated.

## Run

```bash
cd paper-experiments/AISTATS/algae-constant-condition
julia --project=../../.. plot_forcing.jl                 # forcing evidence → outputs/forcing.png

# Fit a constant-condition experiment (well-specified), then analyse residuals:
MAP_ONLY=1 EXPT=1 julia --project=../../.. ../algae-chemostat/algae_chemostat.jl
julia --project=../../.. residual_diag.jl 1              # reads C1's map_prediction.csv

# Compare against the forced C9:
MAP_ONLY=1 EXPT=9 julia --project=../../.. ../algae-chemostat/algae_chemostat.jl
julia --project=../../.. residual_diag.jl 9
```

`residual_diag.jl <EXPT>` writes `outputs/residuals_C<EXPT>.png` and appends a row to
`outputs/residual_summary.csv` (train vs forecast residual std, ratio, rolling-std trend).

## What to look for

- **Well-specified (C1):** residual std roughly **flat** across train→forecast (ratio ≈ 1),
  no obvious structure at any particular days.
- **Forced/misspecified (C9):** residual std **larger in the forecast window** (ratio ≫ 1)
  and structured (peaks near Nᵢₙ switches) — the visual signature that an autonomous model
  is missing a driver. That figure is the honest "here's where/why it fails" panel.
