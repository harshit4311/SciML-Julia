#=
Arm B — MAP ONLY (no HMC).

Two-phase MAP pre-training, then STOP. Reports the MAP point estimate as a
standalone predictor. There is no posterior, hence no coverage and no σ̂: this is
the "why not just MAP?" arm. Note that this MAP point is exactly the NUTS
initialisation used by Arm C — so B vs C isolates what the sampling step adds
(calibrated uncertainty) on top of an identical point estimate. This is the §2.7
"MAP accuracy decoupled from calibration" comparison.

Run:  julia --project=../../.. Exp_B.jl
=#

include("lv_bnode_common.jl")

const ARM   = "B_map_only"
const σ_obs = 0.2            # match the other arms (paper's main regime)
const SPLIT = 100
const NTOT  = 200

outdir = ensure_outdir(ARM)
prob = make_lv_problem(; σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT)
fns  = build_fns(prob)

# Small MAP budget for fast iteration; bump for the camera-ready run.
p_map, mmetrics = run_map(prob, fns; phaseA_iters=1500, phaseB_iters=300)
plot_point_fit(prob, fns, p_map; outdir=outdir, label="Arm B (MAP only)")

println("\n----- Arm B summary -----")
@show mmetrics

append_result!(RESULTS_CSV, (;
    arm=ARM, method="MAP only (no sampling)", init="random",
    σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT, n_samples=0, n_adapts=0,
    map_rmse=mmetrics.map_rmse, map_rel_err=mmetrics.map_rel_err,
    coverage_g=missing, coverage_v=missing,           # no posterior → no UQ
    post_mse_mean=mmetrics.map_val_mse,
    sigma_hat_mean=missing, sigma_hat_std=missing,
    accept=missing, ebfmi=missing, treedepth=missing,
    ndiverge=missing, ess_min=missing, rhat_max=missing,
    runtime_s=missing,
))
