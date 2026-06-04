#=
Hudson Bay lynx-hare case study — BNODE+MAP+HMC on real ecological data.

This applies the *same* harness used in map-tests/Exp_C and bnode-vs-node/
(architecture, MAP scheduler, NUTS config) to the canonical Hudson Bay
Company lynx-hare pelt record (1900–1920). The point: demonstrate that the
calibrated-band machinery characterised on synthetic Lotka–Volterra data in §2
of the paper transfers to a real ecological dataset whose noise structure is
unknown.

Key differences from the synthetic harness:
  - Data is loaded from CSV, not synthesised; no ground-truth trajectory exists.
  - Calibration check is "do held-out OBSERVED counts fall in the 90% band?".
  - Reports σ̂ as the inferred noise scale (no σ_true to compare against).
  - Includes a "decision-relevance" plot: 5th/95th percentile forecast quantiles
    overlaid against the MAP trajectory, framed as a precautionary-management band.

Run (DEV, ~10 min after precompile):
  julia --project=../../.. hudson_bay.jl

Run (PAPER, ~24h):
  NSAMP=250 NADPT=250 MAXDEPTH=10 DEV_TOL=1e-8 \
  MAP_PHASEA=6000 MAP_PHASEB=800 \
    julia --project=../../.. hudson_bay.jl
=#

include("../map-tests/lv_bnode_common.jl")

# === Config — overrides via env vars =========================================
const NSAMP      = parse(Int,     get(ENV, "NSAMP",      "50"))
const NADPT      = parse(Int,     get(ENV, "NADPT",      "50"))
const MAXDEPTH   = parse(Int,     get(ENV, "MAXDEPTH",   "6"))
const DEV_TOL    = parse(Float64, get(ENV, "DEV_TOL",    "1e-6"))
const MAP_PHASEA = parse(Int,     get(ENV, "MAP_PHASEA", "4000"))
const MAP_PHASEB = parse(Int,     get(ENV, "MAP_PHASEB", "500"))
const INIT_SEED  = parse(Int,     get(ENV, "INIT_SEED",  "42"))
const TRAIN_FRAC = parse(Float64, get(ENV, "TRAIN_FRAC", "0.70"))
const TMAX       = parse(Float64, get(ENV, "TMAX",       "7.0"))    # rescale years → [0, TMAX]

outdir = joinpath(@__DIR__, "outputs", "hudson_bay")
mkpath(outdir)
csv_out = joinpath(@__DIR__, "hudson_bay_results.csv")

# === Load + normalise the dataset ============================================
#
# Source: Hudson Bay Company pelt-trade records (Elton & Nicholson 1942);
# 21-year subset 1900–1920 as used in the Stan case study (Carpenter et al.).
# Columns: year, hare (prey, thousands), lynx (predator, thousands).
#
println("Loading Hudson Bay lynx-hare data…")
df = CSV.read(joinpath(@__DIR__, "data", "lynx_hare.csv"), DataFrames.DataFrame)
years = df.year
hare  = Vector{Float64}(df.hare)
lynx  = Vector{Float64}(df.lynx)
n_total = length(years)
n_train = round(Int, TRAIN_FRAC * n_total)
println(@sprintf("  N=%d total, train=%d (%.0f%%), forecast=%d",
                 n_total, n_train, 100*TRAIN_FRAC, n_total - n_train))

# Normalise each channel by the training-window mean. Keeps the architecture
# (2-32-32-32-2 with tanh, sized for synthetic LV) in its operating range.
hare_scale = mean(hare[1:n_train])
lynx_scale = mean(lynx[1:n_train])
hare_n = hare ./ hare_scale
lynx_n = lynx ./ lynx_scale
println(@sprintf("  Normalisation: hare_scale=%.2f, lynx_scale=%.2f", hare_scale, lynx_scale))

# Rescale years → [0, TMAX]. Matches the temporal scale of the synthetic LV
# (≈2 cycles over 7 time units on the dataset's ≈9-yr period).
tsteps  = collect(range(0.0, TMAX, length=n_total))
t_train = tsteps[1:n_train]
t_val   = tsteps[n_train+1:end]

# Build ode_data matrix: rows = (hare, lynx); columns = time.
ode_data   = permutedims(hcat(hare_n, lynx_n))         # 2 × n_total
data_train = ode_data[:, 1:n_train]
data_val   = ode_data[:, n_train+1:end]
u0         = ode_data[:, 1]

# === Build network + initial params (mirror lv_bnode_common.make_lv_problem) =
dudt2 = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 2),
)
Random.seed!(INIT_SEED)
rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
p_struct  = ComponentArrays.ComponentArray{Float64}(p)
p_flat_nn = vec(collect(p_struct))
logσ_init = log(0.1)
p_flat_init = vcat(p_flat_nn, logσ_init)
println(@sprintf("  Params: %d (+1 logσ)", length(p_flat_nn)))

# === Construct prob NamedTuple in the shape build_fns expects ================
prob = (;
    tsteps, t_train, t_val,
    ode_data,
    ode_data_clean = ode_data,         # no clean reference for real data
    data_train, data_val, u0,
    dudt2, st, p_struct,
    p_flat_init, logσ_init,
    n_train, n_total,
    σ_obs = NaN,                       # unknown for real data
    solver_reltol = DEV_TOL,
    solver_abstol = DEV_TOL,
)
fns = build_fns(prob)

# === MAP pre-training ========================================================
println("\n=== MAP pre-training ===")
p_map, mm = run_map(prob, fns; phaseA_iters=MAP_PHASEA, phaseB_iters=MAP_PHASEB)
plot_point_fit(prob, fns, p_map; outdir=outdir, label="Hudson Bay MAP")

# === NUTS ====================================================================
println("\n=== NUTS sampling ===")
samples, stats, nuts_rt = run_nuts(prob, fns, p_map;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)

# === Posterior analysis (uses shared analyze_posterior) =====================
diag = nuts_diagnostics(samples, stats)
post = analyze_posterior(prob, fns, samples; outdir=outdir, label="Hudson Bay BNODE")

# === Forecast-window-only coverage (the real-data calibration check) =========
# The shared analyze_posterior computes coverage over the FULL trajectory.
# For Hudson Bay the relevant calibration claim is the FORECAST window only.
n_post = size(samples, 2)
ntime_val = length(t_val)
pp_hare_val = fill(NaN, ntime_val, n_post)
pp_lynx_val = fill(NaN, ntime_val, n_post)
for k in 1:n_post
    col = samples[:, k]
    p_v = try fns.solve_valid_forecast(col[1:end-1]) catch; nothing end
    (p_v === nothing || size(p_v) != (2, ntime_val) || !all(isfinite, p_v)) && continue
    σ_b = exp(col[end]); σ_r = σ_b / sqrt(W2)
    pp_hare_val[:, k] .= p_v[1, :] .+ randn(ntime_val) .* σ_b
    pp_lynx_val[:, k] .= p_v[2, :] .+ randn(ntime_val) .* σ_r
end

lo_hare_val = [_finite_q(pp_hare_val[i, :], 0.05) for i in 1:ntime_val]
hi_hare_val = [_finite_q(pp_hare_val[i, :], 0.95) for i in 1:ntime_val]
lo_lynx_val = [_finite_q(pp_lynx_val[i, :], 0.05) for i in 1:ntime_val]
hi_lynx_val = [_finite_q(pp_lynx_val[i, :], 0.95) for i in 1:ntime_val]
mean_hare_val = [(f = filter(isfinite, pp_hare_val[i, :]); isempty(f) ? NaN : mean(f))
                 for i in 1:ntime_val]
mean_lynx_val = [(f = filter(isfinite, pp_lynx_val[i, :]); isempty(f) ? NaN : mean(f))
                 for i in 1:ntime_val]

covg(d, lo, hi) = (m = isfinite.(lo) .& isfinite.(hi);
                   any(m) ? mean((d[m] .>= lo[m]) .& (d[m] .<= hi[m])) : NaN)
forecast_coverage_hare = covg(data_val[1, :], lo_hare_val, hi_hare_val)
forecast_coverage_lynx = covg(data_val[2, :], lo_lynx_val, hi_lynx_val)

# === Decision-relevance plot ================================================
# This is the plot that makes the case study a *case study*: it overlays MAP
# and 5th/95th percentile forecast trajectories on the held-out data,
# explicitly framing the lower quantile as a precautionary-management bound.
try
    pl = Plots.plot(t_val, mean_hare_val, color=:blue, lw=2,
                    ribbon=(mean_hare_val .- lo_hare_val, hi_hare_val .- mean_hare_val),
                    fillalpha=0.2, label="Hare: posterior mean (90% PP CI)",
                    xlabel="Rescaled time", ylabel="Normalised population",
                    title="Decision-relevant forecast: Hudson Bay BNODE")
    Plots.plot!(pl, t_val, lo_hare_val, color=:blue, lw=2, ls=:dash,
                label="Hare: 5th percentile (precautionary lower bound)")
    Plots.plot!(pl, t_val, mean_lynx_val, color=:red, lw=2,
                ribbon=(mean_lynx_val .- lo_lynx_val, hi_lynx_val .- mean_lynx_val),
                fillalpha=0.2, label="Lynx: posterior mean (90% PP CI)")
    Plots.plot!(pl, t_val, lo_lynx_val, color=:red, lw=2, ls=:dash,
                label="Lynx: 5th percentile")
    Plots.scatter!(pl, t_val, data_val[1, :], color=:blue, alpha=0.6,
                   label="Hare: observed (held out)")
    Plots.scatter!(pl, t_val, data_val[2, :], color=:red, alpha=0.6,
                   label="Lynx: observed (held out)")
    Plots.savefig(pl, joinpath(outdir, "decision_relevance.png"))
catch e
    @warn "decision-relevance plot failed" exception=e
end

# === Console summary =========================================================
println("\n----- Hudson Bay BNODE summary -----")
println(@sprintf("MAP  val_rmse  : %.4f  rel_err %.3f%%", mm.map_rmse, 100*mm.map_rel_err))
println(@sprintf("Posterior σ̂   : mean=%.4f  std=%.4f",
                 post.sigma_hat_mean, post.sigma_hat_std))
println(@sprintf("Coverage (full): hare=%.3f  lynx=%.3f", post.coverage_g, post.coverage_v))
println(@sprintf("Coverage (val) : hare=%.3f  lynx=%.3f",
                 forecast_coverage_hare, forecast_coverage_lynx))
println(@sprintf("NUTS diag      : accept=%.3f  EBFMI=%.3f  treedepth=%.2f  divergences=%d",
                 diag.accept, diag.ebfmi, diag.treedepth, diag.ndiverge))
println(@sprintf("NUTS quality   : ESS_min=%.2f  R̂_max=%.3f  runtime=%.1fs",
                 diag.ess_min, diag.rhat_max, nuts_rt))

# === Write results row ======================================================
append_result!(csv_out, (;
    dataset = "Hudson Bay lynx-hare 1900–1920",
    n_total, n_train,
    config = "MAP=$MAP_PHASEA/$MAP_PHASEB, NUTS=$NSAMP/$NADPT, " *
             "max_depth=$MAXDEPTH, tol=$DEV_TOL, train_frac=$TRAIN_FRAC",
    init_seed = INIT_SEED,
    hare_scale, lynx_scale,
    map_rmse        = mm.map_rmse,
    map_rel_err     = mm.map_rel_err,
    sigma_hat_mean  = post.sigma_hat_mean,
    sigma_hat_std   = post.sigma_hat_std,
    coverage_full_hare    = post.coverage_g,
    coverage_full_lynx    = post.coverage_v,
    coverage_forecast_hare = forecast_coverage_hare,
    coverage_forecast_lynx = forecast_coverage_lynx,
    post_mse_mean   = post.post_mse_mean,
    accept   = diag.accept,
    ebfmi    = diag.ebfmi,
    treedepth = diag.treedepth,
    ndiverge = diag.ndiverge,
    ess_min  = diag.ess_min,
    rhat_max = diag.rhat_max,
    runtime_s = nuts_rt,
))

println("\nCSV   → $csv_out")
println("Plots → $outdir/")
