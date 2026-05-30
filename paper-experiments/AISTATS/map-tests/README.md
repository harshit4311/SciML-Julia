# MAP / HMC ablation harness (AISTATS)

Four arms sharing **identical** data, architecture, likelihood, and budget, so the
"why MAP / why HMC / why Bayesian" comparison is apples-to-apples. All shared code
lives in `lv_bnode_common.jl`; each arm script only sets a config and calls drivers.

| Script | Arm | Init | Inference | Answers reviewer question |
|--------|-----|------|-----------|---------------------------|
| `Exp_A.jl` | HMC only | random | NUTS | *Why not just HMC?* → expected failure (collapse / partial fit / 10⁵ explosion) |
| `Exp_B.jl` | MAP only | — | Adam MAP, no sampling | *Why not just MAP?* → good point fit, **no uncertainty** |
| `Exp_C.jl` | **MAP→HMC (ours)** | MAP | NUTS | the method |
| `Exp_D.jl` | Plain NeuralODE | random | Adam MSE | *Why be Bayesian (BNODE vs NODE)?* |

Key contrasts: **A vs C** = MAP is necessary for the sampler to find the orbit.
**B vs C** = same point estimate, but only C adds calibrated uncertainty (the §2.7
decoupling claim). **C vs D** = what the Bayesian treatment buys over a plain NODE.

## Run

```bash
cd paper-experiments/AISTAATS/map-tests
julia --project=../../.. Exp_A.jl   # then Exp_B / Exp_C / Exp_D
```

Each run appends a row to `results.csv` (the benchmark table) and writes plots to
`outputs/<arm>/`. `smoke_test.jl` runs the whole pipeline at tiny budgets (~1 min
after precompile) to verify the API.

## Budgets ("smaller set of heuristics")

The arm scripts default to a **small** budget for fast iteration:
NUTS `50/50`, MAP `phaseA=1500 / phaseB=300`. For the camera-ready numbers, bump
the `NSAMP/NADPT` consts to `250/250` (matching the main paper) and MAP to
`6000/800`. Nothing else changes.

## Metrics in `results.csv`

`map_rmse`, `map_rel_err`, `coverage_g/_v` (90% PP coverage), `post_mse_mean`,
`sigma_hat_mean/_std`, `accept` (acceptance rate), `ebfmi`, `treedepth`,
`ndiverge`, `ess_min`, `rhat_max`, `runtime_s`. Arms without a posterior (B, D)
leave the UQ/sampler columns `missing` — that absence *is* the point.

## Not yet wired (next steps for full reviewer coverage)

- **Three failure modes for Arm A** (Fig. 1): sweep `init_seed` and the MAP-loss
  vs Gaussian-LL objective to land collapse / partial-fit / explosion deterministically.
- **Multi-seed** runs per arm (≥3 seeds) for error bars — addresses the single-chain
  Monte-Carlo-uncertainty limitation in §3.
- **σ and split sweep** reusing `make_lv_problem(; σ_obs=…, n_train=…)`.
