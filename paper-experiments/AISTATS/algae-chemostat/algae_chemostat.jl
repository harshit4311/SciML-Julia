#=
Rotifer–algae chemostat case study — BNODE+MAP+HMC on real predator–prey data.

Applies the *same* harness used in map-tests/Exp_C and the Hudson Bay / HIV
case studies (architecture, MAP scheduler, NUTS config) to the Blasius et al.
(2020) rotifer–algae chemostat time series — the experimental realisation of
Lotka–Volterra. The point: show that the calibrated-band machinery characterised
on synthetic LV data in §2 transfers to a *dense, long-horizon, low-but-unknown
noise* real dataset, completing the spectrum next to Hudson Bay (short, heavy
noise) and HIV (sparse, pooled).

State variables (the two LV channels):
  z = [algae, rotifers]   algae = prey (10⁶ cells/ml), rotifers = predator (/ml)

Key differences from the synthetic harness (same as Hudson Bay, plus two new):
  - Data loaded from CSV; no ground-truth trajectory exists (σ_obs = NaN).
  - Calibration check is "do held-out OBSERVED counts fall in the 90% band?".
  - Reports σ̂ as the inferred noise scale.
  - [NEW] One experiment is selected from the pooled 10-experiment CSV (EXPT,
    default C1 — the ~374-day flagship run; the single-trajectory analogue of
    the Hudson Bay fit).
  - [NEW] Measurement-days with a MISSING algae or rotifer value (coded NaN in
    the source, empty in the CSV) are dropped so the Gaussian likelihood stays
    finite; the irregular surviving sample times are used directly as `saveat`.

Time rescaling note: unlike the 21-point lynx–hare record (≈2 cycles → [0,7]),
C1 contains many predator–prey cycles over its ~374 days. `TMAX` rescales the
record to ODE time [0, TMAX]; the default (30) keeps the per-cycle resolution in
the same range the 2-32-32-32-2 network was sized for on synthetic LV (≈2
cycles / 7 units). Lower TMAX compresses more cycles into the synthetic window
(harder for the net); higher TMAX lengthens integration (slower, harder NUTS).

Run (DEV, ~20–40 min after precompile):
  julia --project=../../.. algae_chemostat.jl

Run a different experiment:
  EXPT=6 julia --project=../../.. algae_chemostat.jl

Run (PAPER, long — relax tol, deepen tree, more MAP):
  NSAMP=250 NADPT=250 MAXDEPTH=10 DEV_TOL=1e-8 \
  MAP_PHASEA=6000 MAP_PHASEB=800 \
    julia --project=../../.. algae_chemostat.jl
=#

include("../map-tests/lv_bnode_common.jl")
include("plot_helpers.jl")

# === Config — overrides via env vars =========================================
const EXPT       = parse(Int,     get(ENV, "EXPT",       "1"))      # which chemostat (1–10)
const NSAMP      = parse(Int,     get(ENV, "NSAMP",      "200"))
const NADPT      = parse(Int,     get(ENV, "NADPT",      "200"))
const MAXDEPTH   = parse(Int,     get(ENV, "MAXDEPTH",   "8"))
const DEV_TOL    = parse(Float64, get(ENV, "DEV_TOL",    "1e-6"))
const MAP_PHASEA = parse(Int,     get(ENV, "MAP_PHASEA", "4000"))
const MAP_PHASEB = parse(Int,     get(ENV, "MAP_PHASEB", "500"))
const INIT_SEED  = parse(Int,     get(ENV, "INIT_SEED",  "42"))
const TRAIN_FRAC = parse(Float64, get(ENV, "TRAIN_FRAC", "0.70"))
const TMAX       = parse(Float64, get(ENV, "TMAX",       "30.0"))   # rescale days → [0, TMAX]
const NWIN       = parse(Int,     get(ENV, "NWIN",       "5"))      # time-window rows in faceted plot

outdir = joinpath(@__DIR__, "outputs", "algae_chemostat")
mkpath(outdir)
csv_out = joinpath(@__DIR__, "algae_chemostat_results.csv")

# === Load + select experiment + drop missing days ============================
#
# Source: Blasius et al. (2020), Nature 577:226–230. figshare 10045976.
# Pooled CSV columns: experiment, day, algae, rotifers, egg_ratio, eggs, dead,
# medium_N. We model the two LV channels (algae, rotifers) for one experiment.
#
println("Loading Blasius rotifer–algae chemostat data (experiment C$EXPT)…")
df_all = CSV.read(joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv"),
                  DataFrames.DataFrame; missingstring="")
df = df_all[df_all.experiment .== EXPT, :]
DataFrames.nrow(df) > 0 || error("No rows for experiment $EXPT (valid: 1–10).")

# Drop measurement-days missing either channel so the likelihood stays finite.
keep = .!ismissing.(df.algae) .& .!ismissing.(df.rotifers)
n_dropped = sum(.!keep)
df = df[keep, :]
DataFrames.sort!(df, :day)

day      = Vector{Float64}(df.day)
algae    = Vector{Float64}(df.algae)        # prey, 10⁶ cells/ml
rotifers = Vector{Float64}(df.rotifers)     # predator, animals/ml
n_total  = length(day)
n_train  = round(Int, TRAIN_FRAC * n_total)
println(@sprintf("  N=%d usable days (%d dropped for missing), train=%d (%.0f%%), forecast=%d",
                 n_total, n_dropped, n_train, 100*TRAIN_FRAC, n_total - n_train))
println(@sprintf("  Time span: %.1f–%.1f days → rescaled to [0, %.1f]",
                 first(day), last(day), TMAX))

# Normalise each channel by its training-window mean (keeps the tanh network,
# sized for synthetic LV, in its operating range). Algae ~O(1) in 10⁶/ml,
# rotifers ~O(10–100)/ml, so this also balances the two channels.
algae_scale = mean(algae[1:n_train])
rotif_scale = mean(rotifers[1:n_train])
algae_n = algae ./ algae_scale
rotif_n = rotifers ./ rotif_scale
println(@sprintf("  Normalisation: algae_scale=%.3f, rotifer_scale=%.3f", algae_scale, rotif_scale))

# Rescale the (irregular) observation days → [0, TMAX], preserving spacing.
day0, dayT = first(day), last(day)
tsteps  = (day .- day0) ./ (dayT - day0) .* TMAX
t_train = tsteps[1:n_train]
t_val   = tsteps[n_train+1:end]

# Build ode_data matrix: rows = (algae, rotifers); columns = time.
ode_data   = permutedims(hcat(algae_n, rotif_n))       # 2 × n_total
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
plot_point_fit(prob, fns, p_map; outdir=outdir, label="Chemostat C$EXPT MAP",
               ch1_label="Algae", ch2_label="Rotifers")

# === NUTS ====================================================================
println("\n=== NUTS sampling ===")
samples, stats, nuts_rt = run_nuts(prob, fns, p_map;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)

# === Posterior analysis (uses shared analyze_posterior) =====================
diag = nuts_diagnostics(samples, stats)
post = analyze_posterior(prob, fns, samples; outdir=outdir, label="Chemostat C$EXPT BNODE",
                          ch1_label="Algae", ch2_label="Rotifers")

# === Faceted posterior-predictive plot (legible for the dense, many-cycle data)
# analyze_posterior's single-axis PP plot crams ~360 points + the posterior cloud
# onto one set of twinned axes; for C1 that is unreadable. This splits the two
# channels into columns and time into NWIN window-rows (each y-zoomed), drawing
# the held-out-aware data ON TOP of a light 90% PP band so it stays visible.
n_post = size(samples, 2)
ntime  = length(tsteps)
pp_a = fill(NaN, ntime, n_post); pp_r = fill(NaN, ntime, n_post)
for (k, col) in enumerate(eachcol(samples))
    θ = fns.unflatten_p(col[1:end-1])
    σ_b = exp(col[end]); σ_r = σ_b / sqrt(W2)
    pred = try fns.predict(θ) catch; fill(NaN, 2, ntime) end
    size(pred, 2) == ntime || (pred = fill(NaN, 2, ntime))
    pp_a[:, k] .= pred[1, :] .+ randn(ntime) .* σ_b
    pp_r[:, k] .= pred[2, :] .+ randn(ntime) .* σ_r
end
band(M) = ([_finite_q(M[i, :], 0.05) for i in 1:ntime],
           [_finite_q(M[i, :], 0.95) for i in 1:ntime],
           [(f = filter(isfinite, M[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:ntime])
try
    faceted_grid(joinpath(outdir, "posterior_faceted.png"), day,
                 [ode_data[1, :], ode_data[2, :]];
                 labels=["algae (normalised)", "rotifers (normalised)"],
                 colors=[:seagreen, :firebrick], nwin=NWIN,
                 split_day=day[n_train], bands=[band(pp_a), band(pp_r)],
                 title="C$EXPT BNODE — 90% posterior predictive (band) vs data (points); dashed = train/forecast")
    println("→ posterior_faceted.png")
catch e
    @warn "faceted posterior plot failed" exception=e
end

# === Forecast-window-only coverage (the real-data calibration check) =========
# analyze_posterior computes coverage over the FULL trajectory; the relevant
# calibration claim for this case study is the held-out FORECAST window only.
n_post = size(samples, 2)
ntime_val = length(t_val)
pp_algae_val = fill(NaN, ntime_val, n_post)
pp_rotif_val = fill(NaN, ntime_val, n_post)
for k in 1:n_post
    col = samples[:, k]
    p_v = try fns.solve_valid_forecast(col[1:end-1]) catch; nothing end
    (p_v === nothing || size(p_v) != (2, ntime_val) || !all(isfinite, p_v)) && continue
    σ_b = exp(col[end]); σ_r = σ_b / sqrt(W2)
    pp_algae_val[:, k] .= p_v[1, :] .+ randn(ntime_val) .* σ_b
    pp_rotif_val[:, k] .= p_v[2, :] .+ randn(ntime_val) .* σ_r
end

lo_algae_val = [_finite_q(pp_algae_val[i, :], 0.05) for i in 1:ntime_val]
hi_algae_val = [_finite_q(pp_algae_val[i, :], 0.95) for i in 1:ntime_val]
lo_rotif_val = [_finite_q(pp_rotif_val[i, :], 0.05) for i in 1:ntime_val]
hi_rotif_val = [_finite_q(pp_rotif_val[i, :], 0.95) for i in 1:ntime_val]
mean_algae_val = [(f = filter(isfinite, pp_algae_val[i, :]); isempty(f) ? NaN : mean(f))
                  for i in 1:ntime_val]
mean_rotif_val = [(f = filter(isfinite, pp_rotif_val[i, :]); isempty(f) ? NaN : mean(f))
                  for i in 1:ntime_val]

covg(d, lo, hi) = (m = isfinite.(lo) .& isfinite.(hi);
                   any(m) ? mean((d[m] .>= lo[m]) .& (d[m] .<= hi[m])) : NaN)
forecast_coverage_algae = covg(data_val[1, :], lo_algae_val, hi_algae_val)
forecast_coverage_rotif = covg(data_val[2, :], lo_rotif_val, hi_rotif_val)

# === Decision-relevance plot ================================================
# Overlays MAP / posterior-mean and 5th/95th percentile forecast trajectories on
# the held-out data — the forecast band a manager would act on (e.g. precautionary
# low-prey / high-predator bounds for a bioreactor or fishery analogue).
try
    pl = Plots.plot(t_val, mean_algae_val, color=:seagreen, lw=2,
                    ribbon=(mean_algae_val .- lo_algae_val, hi_algae_val .- mean_algae_val),
                    fillalpha=0.2, label="Algae: posterior mean (90% PP CI)",
                    xlabel="Rescaled time", ylabel="Normalised population",
                    title="Decision-relevant forecast: chemostat C$EXPT BNODE")
    Plots.plot!(pl, t_val, lo_algae_val, color=:seagreen, lw=2, ls=:dash,
                label="Algae: 5th percentile (precautionary lower bound)")
    Plots.plot!(pl, t_val, mean_rotif_val, color=:firebrick, lw=2,
                ribbon=(mean_rotif_val .- lo_rotif_val, hi_rotif_val .- mean_rotif_val),
                fillalpha=0.2, label="Rotifers: posterior mean (90% PP CI)")
    Plots.plot!(pl, t_val, hi_rotif_val, color=:firebrick, lw=2, ls=:dash,
                label="Rotifers: 95th percentile")
    Plots.scatter!(pl, t_val, data_val[1, :], color=:seagreen, alpha=0.6,
                   label="Algae: observed (held out)")
    Plots.scatter!(pl, t_val, data_val[2, :], color=:firebrick, alpha=0.6,
                   label="Rotifers: observed (held out)")
    Plots.savefig(pl, joinpath(outdir, "decision_relevance.png"))
catch e
    @warn "decision-relevance plot failed" exception=e
end

# === Console summary =========================================================
println("\n----- Chemostat C$EXPT BNODE summary -----")
println(@sprintf("MAP  val_rmse  : %.4f  rel_err %.3f%%", mm.map_rmse, 100*mm.map_rel_err))
println(@sprintf("Posterior σ̂   : mean=%.4f  std=%.4f",
                 post.sigma_hat_mean, post.sigma_hat_std))
println(@sprintf("Coverage (full): algae=%.3f  rotifers=%.3f", post.coverage_g, post.coverage_v))
println(@sprintf("Coverage (val) : algae=%.3f  rotifers=%.3f",
                 forecast_coverage_algae, forecast_coverage_rotif))
println(@sprintf("NUTS diag      : accept=%.3f  EBFMI=%.3f  treedepth=%.2f  divergences=%d",
                 diag.accept, diag.ebfmi, diag.treedepth, diag.ndiverge))
println(@sprintf("NUTS quality   : ESS_min=%.2f  R̂_max=%.3f  runtime=%.1fs",
                 diag.ess_min, diag.rhat_max, nuts_rt))

# === Write results row ======================================================
append_result!(csv_out, (;
    dataset = "Blasius 2020 rotifer–algae chemostat C$EXPT",
    experiment = EXPT,
    n_total, n_train, n_dropped,
    config = "MAP=$MAP_PHASEA/$MAP_PHASEB, NUTS=$NSAMP/$NADPT, " *
             "max_depth=$MAXDEPTH, tol=$DEV_TOL, train_frac=$TRAIN_FRAC, tmax=$TMAX",
    init_seed = INIT_SEED,
    algae_scale, rotif_scale,
    map_rmse        = mm.map_rmse,
    map_rel_err     = mm.map_rel_err,
    sigma_hat_mean  = post.sigma_hat_mean,
    sigma_hat_std   = post.sigma_hat_std,
    coverage_full_algae    = post.coverage_g,
    coverage_full_rotif    = post.coverage_v,
    coverage_forecast_algae = forecast_coverage_algae,
    coverage_forecast_rotif = forecast_coverage_rotif,
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
