#=
BNODE vs NODE — pure forecasting-accuracy comparison.

Reuses the same harness, data, architecture, likelihood, MAP scheduler, and dev
budgets used by Exp_C and Exp_D in ../map-tests/. Runs the two methods back-to-back
on IDENTICAL data and writes:

  - forecasting_results.csv   (one row per method: val_rmse / val_rel_err / val_mse)
  - outputs/forecast_compare/ (point-fit plots + a validation-window comparison)

For BNODE the reported point forecast is the POSTERIOR MEAN trajectory on the
validation window — the Bayes-optimal point predictor under the inferred posterior.
The 90% credible band is overlaid on the comparison plot so the cost-vs-benefit
trade is visible (BNODE gives you the band; NODE doesn't).

Run:  julia --project=../../.. forecast_compare.jl
=#

include("../map-tests/lv_bnode_common.jl")

# === Config — IDENTICAL to map-tests dev settings (Exp_C / Exp_D) =============
const σ_obs    = 0.2
const SPLIT    = 100
const NTOT     = 200
const NSAMP    = 50
const NADPT    = 50
const MAXDEPTH = 6
const DEV_TOL  = 1e-6
# NODE_ITERS: 4000 = NODE's natural convergence ("as-deployed").
#             1800 = matched to MAP's iter budget ("apples-to-apples").
# Override from shell:  NODE_ITERS=1800 julia --project=../../.. forecast_compare.jl
const NODE_ITERS = parse(Int, get(ENV, "NODE_ITERS", "4000"))
# FRESH=1 wipes the CSV first; default accumulates rows so both NODE budgets land
# in one table after two runs.
const FRESH = get(ENV, "FRESH", "0") == "1"

# Use this script's directory, not the common module's (ensure_outdir resolves
# @__DIR__ at the common file's location, which would put plots in map-tests/).
outdir = joinpath(@__DIR__, "outputs", "forecast_compare_$(NODE_ITERS)"); mkpath(outdir)
csv_path = joinpath(@__DIR__, "forecasting_results.csv")
(FRESH && isfile(csv_path)) && rm(csv_path)

prob = make_lv_problem(; σ_obs=σ_obs, n_train=SPLIT, n_total=NTOT,
                         solver_reltol=DEV_TOL, solver_abstol=DEV_TOL)
fns  = build_fns(prob)
ntime_val = length(prob.t_val)

# ============================================================================
# NODE — plain deterministic NeuralODE (Exp_D config)
# ============================================================================
println("\n=== NODE (Adam MSE, no prior, no σ) ===")
node_runtime = @elapsed begin
    p_node, nm = run_node(prob, fns; iters=NODE_ITERS, lr=5e-3)
end
θ_node    = fns.unflatten_p(p_node[1:end-1])
node_val  = fns.solve_valid_forecast(θ_node)        # (2 × ntime_val)
node_full = fns.predict(θ_node)

node_val_mse  = mean((prob.data_val .- node_val) .^ 2)
node_val_rmse = sqrt(node_val_mse)
node_val_rel  = norm(prob.data_val .- node_val) / norm(prob.data_val)

plot_point_fit(prob, fns, p_node; outdir=outdir, label="NODE")

# ============================================================================
# BNODE — MAP → HMC (Exp_C config)
# ============================================================================
println("\n=== BNODE (MAP → HMC) ===")
p_map, mm = run_map(prob, fns; phaseA_iters=1500, phaseB_iters=300)
samples, stats, nuts_rt = run_nuts(prob, fns, p_map;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)
bnode_runtime = nuts_rt + 0.0    # MAP wall-clock isn't tracked separately; bias toward NUTS

# Posterior-mean validation forecast + 90% PP CI on validation window.
n_post = size(samples, 2)
latent_g = fill(NaN, ntime_val, n_post); latent_v = similar(latent_g)
pp_g     = fill(NaN, ntime_val, n_post); pp_v     = similar(pp_g)
for k in 1:n_post
    col = samples[:, k]
    p = try fns.solve_valid_forecast(col[1:end-1]) catch; nothing end
    (p === nothing || size(p) != (2, ntime_val) || !all(isfinite, p)) && continue
    σ_b = exp(col[end]); σ_r = σ_b / sqrt(W2)
    latent_g[:, k] .= p[1, :]; latent_v[:, k] .= p[2, :]
    pp_g[:, k] .= p[1, :] .+ randn(ntime_val) .* σ_b
    pp_v[:, k] .= p[2, :] .+ randn(ntime_val) .* σ_r
end

_mean_row(M) = [(f = filter(isfinite, M[i, :]); isempty(f) ? NaN : mean(f))
                for i in 1:size(M, 1)]
mean_g_val = _mean_row(latent_g);  mean_v_val = _mean_row(latent_v)
lo_g = [_finite_q(pp_g[i, :], 0.05) for i in 1:ntime_val]
hi_g = [_finite_q(pp_g[i, :], 0.95) for i in 1:ntime_val]
lo_v = [_finite_q(pp_v[i, :], 0.05) for i in 1:ntime_val]
hi_v = [_finite_q(pp_v[i, :], 0.95) for i in 1:ntime_val]

bnode_mean    = vcat(reshape(mean_g_val, 1, :), reshape(mean_v_val, 1, :))
bnode_val_mse  = mean((prob.data_val .- bnode_mean) .^ 2)
bnode_val_rmse = sqrt(bnode_val_mse)
bnode_val_rel  = norm(prob.data_val .- bnode_mean) / norm(prob.data_val)

# ============================================================================
# Write CSV (one row per method)
# ============================================================================
append_result!(csv_path, (;
    method="NODE (Adam MSE)", config="iters=$NODE_ITERS",
    val_rmse=node_val_rmse, val_rel_err=node_val_rel, val_mse=node_val_mse,
    has_uncertainty=false, runtime_s=node_runtime,
))
append_result!(csv_path, (;
    method="BNODE (MAP→HMC, posterior mean)",
    config="MAP=1500/300, NUTS=$NSAMP/$NADPT, max_depth=$MAXDEPTH, tol=$DEV_TOL",
    val_rmse=bnode_val_rmse, val_rel_err=bnode_val_rel, val_mse=bnode_val_mse,
    has_uncertainty=true, runtime_s=bnode_runtime,
))

# ============================================================================
# Combined validation-window plot
# ============================================================================
try
    tv = prob.t_val
    pl = Plots.plot(title="Validation-window forecast: NODE vs BNODE",
                    xlabel="Time", ylabel="State", legend=:outertopright)
    Plots.scatter!(pl, tv, prob.data_val[1, :], color=:blue, alpha=0.5, label="Data: Growth")
    Plots.scatter!(pl, tv, prob.data_val[2, :], color=:red,  alpha=0.5, label="Data: Value")
    # NODE — single deterministic line
    Plots.plot!(pl, tv, node_val[1, :], color=:blue, lw=2, ls=:dash, label="NODE: Growth")
    Plots.plot!(pl, tv, node_val[2, :], color=:red,  lw=2, ls=:dash, label="NODE: Value")
    # BNODE — posterior mean + 90% PP band
    Plots.plot!(pl, tv, mean_g_val, ribbon=(mean_g_val .- lo_g, hi_g .- mean_g_val),
                color=:blue, lw=2, fillalpha=0.20, label="BNODE mean+90% CI (Growth)")
    Plots.plot!(pl, tv, mean_v_val, ribbon=(mean_v_val .- lo_v, hi_v .- mean_v_val),
                color=:red,  lw=2, fillalpha=0.20, label="BNODE mean+90% CI (Value)")
    Plots.savefig(pl, joinpath(outdir, "validation_compare.png"))
catch e
    @warn "validation comparison plot failed" exception=e
end

println("\n----- BNODE vs NODE forecasting summary -----")
println(@sprintf("NODE   val_rmse=%.4f  rel_err=%.3f%%", node_val_rmse, 100*node_val_rel))
println(@sprintf("BNODE  val_rmse=%.4f  rel_err=%.3f%%  (posterior mean)",
                 bnode_val_rmse, 100*bnode_val_rel))
println("CSV   → $csv_path")
println("Plots → $outdir/")
