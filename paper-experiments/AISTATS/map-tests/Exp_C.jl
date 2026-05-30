#=
Arm C — MAP → HMC (OURS).

Two-phase MAP pre-training, then NUTS started from the MAP checkpoint. This is the
method in the paper. Compared against Arm A it shows MAP is necessary for the
sampler to find the orbit; compared against Arm B it shows HMC adds calibrated
uncertainty on top of the same point estimate; compared against Arm D it shows
what the Bayesian treatment buys over a plain NeuralODE.

Run:  julia --project=../../.. Exp_C.jl
=#

include("lv_bnode_common.jl")

const ARM   = "C_map_then_hmc"
const σ_obs = 0.2            # paper's main regime (σ=0.5 is the pathological case, App. A.3)
const SPLIT = 100
const NTOT  = 200
const NSAMP = 50             # small "heuristic" budget; bump to 250/250 for final
const NADPT = 50
const MAXDEPTH = 6           # DEV: caps NUTS trajectory (~16× faster). Set 10 for camera-ready.
const DEV_TOL  = 1e-6        # DEV: looser ODE tolerance. Set 1e-8 for camera-ready.

outdir = ensure_outdir(ARM)
prob = make_lv_problem(; σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT,
                         solver_reltol=DEV_TOL, solver_abstol=DEV_TOL)
fns  = build_fns(prob)

# 1) MAP checkpoint.
p_map, mmetrics = run_map(prob, fns; phaseA_iters=1500, phaseB_iters=300)
plot_point_fit(prob, fns, p_map; outdir=outdir, label="Arm C MAP checkpoint")

# 2) NUTS from the MAP checkpoint.
samples, stats, runtime = run_nuts(prob, fns, p_map;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)

diag = nuts_diagnostics(samples, stats)
post = analyze_posterior(prob, fns, samples; outdir=outdir, label="Arm C (MAP→HMC)")

println("\n----- Arm C summary -----")
@show mmetrics diag post

append_result!(RESULTS_CSV, (;
    arm=ARM, method="MAP→HMC (ours)", init="MAP",
    σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT, n_samples=NSAMP, n_adapts=NADPT,
    map_rmse=mmetrics.map_rmse, map_rel_err=mmetrics.map_rel_err,
    coverage_g=post.coverage_g, coverage_v=post.coverage_v,
    post_mse_mean=post.post_mse_mean,
    sigma_hat_mean=post.sigma_hat_mean, sigma_hat_std=post.sigma_hat_std,
    accept=diag.accept, ebfmi=diag.ebfmi, treedepth=diag.treedepth,
    ndiverge=diag.ndiverge, ess_min=diag.ess_min, rhat_max=diag.rhat_max,
    runtime_s=runtime,
))
