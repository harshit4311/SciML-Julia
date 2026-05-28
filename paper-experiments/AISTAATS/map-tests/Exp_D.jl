#=
Arm D — PLAIN NeuralODE (non-Bayesian baseline).

Standard NeuralODE trained by Adam on the weighted MSE loss — no weight prior, no
noise scale, no sampling. This is the conventional NODE the BNODE is benchmarked
against ("why be Bayesian at all?"). It produces a point forecast with NO
calibrated uncertainty; the contrast with Arm C is the BNODE-vs-NODE argument.

Run:  julia --project=../../.. Exp_D.jl
=#

include("lv_bnode_common.jl")

const ARM   = "D_plain_node"
const σ_obs = 0.2            # match the other arms (paper's main regime)
const SPLIT = 100
const NTOT  = 200

outdir = ensure_outdir(ARM)
prob = make_lv_problem(; σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT)
fns  = build_fns(prob)

p_node, nmetrics = run_node(prob, fns; iters=4000, lr=5e-3)
plot_point_fit(prob, fns, p_node; outdir=outdir, label="Arm D (plain NeuralODE)")

println("\n----- Arm D summary -----")
@show nmetrics

append_result!(RESULTS_CSV, (;
    arm=ARM, method="Plain NeuralODE (Adam MSE)", init="random",
    σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT, n_samples=0, n_adapts=0,
    map_rmse=nmetrics.node_rmse, map_rel_err=nmetrics.node_rel_err,
    coverage_g=missing, coverage_v=missing,           # deterministic → no UQ
    post_mse_mean=nmetrics.node_val_mse,
    sigma_hat_mean=missing, sigma_hat_std=missing,
    accept=missing, ebfmi=missing, treedepth=missing,
    ndiverge=missing, ess_min=missing, rhat_max=missing,
    runtime_s=missing,
))
