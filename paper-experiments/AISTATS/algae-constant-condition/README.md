# Constant-condition chemostat fits + residual diagnostics

A follow-up to [`../algae-chemostat/`](../algae-chemostat/) prompted by a structural
discovery: **C9 (our previous showcase) is an externally *forced* experiment**, and an
autonomous neural ODE cannot fit a forced system. This folder moves the headline
real-data fit to a **constant-condition** experiment (where the autonomous BNODE is
well-specified) and adds a **residual-variance / stationarity diagnostic** to make the
misspecification visible.

## The discovery: three regimes (verified against the paper, not just our CSV)

Our model is autonomous, `ẋ = f_θ(x)`, so it can only fit **constant-condition** runs.
**Caution:** the pooled CSV only has `medium_N` (nutrient inflow) — it does **not** have
the **dilution rate δ**, which is the *other* control knob and the one some perturbations
act on. So "constant `medium_N`" does **not** prove "constant conditions." The
authoritative grouping is from Blasius 2020 Methods / Extended Data Fig. 8 (standard
params: alga *M. minutum*, δ=0.55/day, Nᵢₙ=80):

| group | experiments | what varies | autonomous BNODE |
|---|---|---|---|
| **Constant conditions** | **C1–C4** (standard); **C5** (δ=0.66, 85d); **C6, C7** (*C. vulgaris*, Nᵢₙ=160) | nothing within a run | ✅ well-specified |
| **External forcing** | C8, C9 | Nᵢₙ square wave 160↔0, 8-day period | ❌ misspecified |
| **Press perturbation** | **C10** | δ stepped 0.55→1.2 (day 84)→1.35 (day 123); oscillations then die | ❌ misspecified |

`plot_forcing.jl` (`outputs/forcing.png`) shows the Nᵢₙ square wave on C8/C9 — but note it
**cannot** show C10's press (δ isn't in the data); C10 looks flat there yet is *not*
constant-condition.

**Why this matters.** A time-dependent driver `Nᵢₙ(t)` (C8/C9) or a mid-run parameter step
(C10) cannot be represented by an autonomous field, so on C9 there is a hard accuracy
ceiling (~52% relErr) *insensitive to network width, MAP iters, segment length, and prior*
— all swept, none moved it. The "cleanest ACF" that made us pick C9 was the **forcing
signal**, not intrinsic dynamics.

## Candidate experiments

Need BOTH boxes: **constant-condition** (well-specified) AND **strong intrinsic cycle**
(real signal to fit). Only **C6 and C7** pass both (constant, ACF ~0.4). C1–C4 are
constant but noise-dominated (ACF ~0.23–0.35 → faint/poor fit, as C1 confirmed); C5 is
constant but weak *and* shortest; C10 is press-perturbed (out). So the headline candidates
are **C6, C7**; keep **C9** (forcing) and **C10** (press) as honest stress cases.

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
