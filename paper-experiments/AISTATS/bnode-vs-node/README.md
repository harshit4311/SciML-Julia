# BNODE vs NODE — forecasting accuracy

A focused side-by-side run that answers one question: **on the same data and the
same dev budget, how does a plain NeuralODE compare to our BNODE (MAP→HMC) on
point forecast accuracy over the validation window?**

Uses the same shared harness (`../map-tests/lv_bnode_common.jl`) and the same
dev configs as Exp_C and Exp_D in `../map-tests/`:

| | Setting |
|---|---|
| σ_obs | 0.2 |
| Train / val split | 100 / 100 (200 pts) |
| NODE | Adam (MSE), 4000 iters |
| BNODE | MAP (1500 + 300) → NUTS (50 / 50, max_depth=6, tol=1e-6) |

## What's reported

For NODE: the deterministic prediction on the validation window.
For BNODE: the **posterior mean** trajectory on the validation window — the
Bayes-optimal point predictor under the inferred posterior. The 90% credible
band is plotted alongside for visual context (it's the thing NODE can't give).

Both methods report `val_rmse`, `val_rel_err`, `val_mse` into
`forecasting_results.csv`. The combined plot is `outputs/forecast_compare/validation_compare.png`.

## Run

```bash
cd paper-experiments/AISTAATS/bnode-vs-node
julia --project=../../.. forecast_compare.jl
```

CSV is rewritten each run (this folder is for the head-to-head, not accumulation).

## Reading the result

Don't expect BNODE to win on RMSE. The point is the *opposite* — that BNODE is
**competitive** with plain NODE on the deterministic point forecast, while
*also* producing a calibrated band. If RMSEs are within the same ballpark, the
argument is "you get uncertainty at essentially no point-accuracy cost," which
is the BNODE pitch the paper actually makes.
