#=
Arm A — HMC ONLY (no MAP pre-training).

NUTS is started from the RANDOM network initialisation (prob.p_flat_init), with
no MAP checkpoint. This is the honest "why not just HMC?" baseline: it is expected
to fail (baseline collapse / partial fit / 10^5 phase-space explosion), reproducing
Figure 1 (top two rows) of the paper.

Run:  julia --project=../../.. Exp_A.jl
=#

include("lv_bnode_common.jl")

const ARM   = "A_hmc_only"
const σ_obs = 0.2            # match the other arms (paper's main regime)
const SPLIT = 100            # train points (of n_total)
const NTOT  = 200
# Defaults below are DEV (fast iteration). For the paper, run with:
#   NSAMP=250 NADPT=250 MAXDEPTH=10 DEV_TOL=1e-8 julia --project=../../.. Exp_A.jl
const NSAMP    = parse(Int,     get(ENV, "NSAMP",    "50"))
const NADPT    = parse(Int,     get(ENV, "NADPT",    "50"))
const MAXDEPTH = parse(Int,     get(ENV, "MAXDEPTH", "6"))
const DEV_TOL  = parse(Float64, get(ENV, "DEV_TOL",  "1e-6"))

outdir = ensure_outdir(ARM)
prob = make_lv_problem(; σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT,
                         solver_reltol=DEV_TOL, solver_abstol=DEV_TOL)
fns  = build_fns(prob)

# NUTS straight from random init — NO MAP.
samples, stats, runtime = run_nuts(prob, fns, prob.p_flat_init;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)

diag = nuts_diagnostics(samples, stats)
post = analyze_posterior(prob, fns, samples; outdir=outdir, label="Arm A (HMC only)")

println("\n----- Arm A summary -----")
@show diag post

append_result!(RESULTS_CSV, (;
    arm=ARM, method="HMC only (random init)", init="random",
    σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT, n_samples=NSAMP, n_adapts=NADPT,
    map_rmse=missing, map_rel_err=missing,
    coverage_g=post.coverage_g, coverage_v=post.coverage_v,
    post_mse_mean=post.post_mse_mean,
    sigma_hat_mean=post.sigma_hat_mean, sigma_hat_std=post.sigma_hat_std,
    accept=diag.accept, ebfmi=diag.ebfmi, treedepth=diag.treedepth,
    ndiverge=diag.ndiverge, ess_min=diag.ess_min, rhat_max=diag.rhat_max,
    runtime_s=runtime,
))
