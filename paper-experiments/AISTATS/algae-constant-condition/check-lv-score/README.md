# LV-ness score

How well does the **classical 4-parameter Lotka–Volterra** model fit a chemostat
experiment? This quantifies the "is the data LV-enough?" question that motivates
the LV inductive bias of the BNODE work.

## What it does

`check_lv_score.jl` fits the textbook LV system
```
dx/dt =  α·x − β·x·y      (prey  = algae)
dy/dt = −δ·y + γ·x·y      (predator = rotifer)
```
to an experiment's **full record**, using the *same multiple-shooting protocol* as
the BNODE (so it's a fair comparison), with random restarts (the LV loss is
non-convex). It reports:

- **best-fit (α, β, δ, γ)** — per-day rates on mean-normalised states. These are
  the parameters of the *closest* LV model; the data is real biology, not
  LV-generated, so they're estimates, not ground truth.
- **LV-ness %** = Nash–Sutcliffe efficiency (variance explained), clamped at 0:
  - **LOCAL** (~2-cycle multiple-shooting horizon): does the data's *flow* look LV?
  - **GLOBAL** (one free-running orbit from the first point): does a single LV
    orbit reproduce the whole series? Strict — penalises phase drift over cycles.

100% = LV explains everything; 0% = no better than a flat mean line.

## Run

```bash
cd paper-experiments/AISTATS/algae-constant-condition/check-lv-score
EXPT=6 julia --project=../../../.. check_lv_score.jl
```
Knobs: `EXPT`, `SEG_LEN`/`SEG_STRIDE` (match the BNODE: 36/18), `RESTARTS`, `ITERS`.

## Outputs

- `outputs/lv_fit_C<EXPT>.png` — the best-fit LV orbit (line) over the data (points), faceted by time window.
- `outputs/lv_scores.csv` — one row per experiment: params + LOCAL/GLOBAL LV-ness %.

## Reading it

- High LOCAL but low GLOBAL ⇒ the data is *locally* LV-shaped but its period/phase
  isn't perfectly constant, so one fixed orbit drifts out of sync over many cycles.
- Both low ⇒ the dynamics aren't well described by classical LV (noise-dominated,
  forced, or a different functional form).
- Compare LV-ness against the BNODE's fit on the same data: how much of the
  learnable signal is already captured by the 4-param mechanistic model vs. needing
  the flexible 354-param net.
