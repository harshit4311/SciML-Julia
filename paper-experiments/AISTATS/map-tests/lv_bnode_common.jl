#=
lv_bnode_common.jl — shared harness for the MAP / HMC ablation (AISTATS).

Defines the Lotka–Volterra data, the Neural-ODE model, the Gaussian likelihood,
the MAP pre-training driver, the NUTS driver, a deterministic NeuralODE driver,
and all benchmark metrics (coverage, acceptance rate, EBFMI, ESS/Rhat, runtime).

Each arm script (Exp_A..Exp_D) only sets a config and calls these drivers, so the
four arms share IDENTICAL data, architecture, likelihood, and budget. That is what
makes the "why MAP / why HMC / BNODE vs NODE" comparison honest.

No top-level work runs on include(); this file only defines functions.
=#

import SciMLSensitivity as SMS
import DifferentialEquations as DE
import Lux
import Zygote
import Optimization
import OptimizationOptimisers
import Random
import Plots
import AdvancedHMC
import MCMCChains
import StatsPlots
import ComponentArrays
import CSV
import DataFrames
using Statistics: mean, std, quantile, var
using LinearAlgebra: norm
using Printf: @sprintf

# Value-channel likelihood is down-weighted by 1/W2 (predator amplitude is smaller).
const W2 = 3.0

# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

"""
    make_lv_problem(; σ_obs, n_train, n_total, tmax, data_seed, init_seed)

Generate the noisy Lotka–Volterra dataset and a random network initialisation.
`data_seed` controls the observation noise (shared across arms so the data is
identical); `init_seed` controls the random network weights (shared so arms A and
D start from the same random point). Returns a NamedTuple `prob`.
"""
function make_lv_problem(; σ_obs::Float64=0.2,
                           n_train::Int=100,
                           n_total::Int=200,
                           tmax::Float64=7.0,
                           data_seed::Int=42,
                           init_seed::Int=42,
                           solver_reltol::Float64=1e-8,
                           solver_abstol::Float64=1e-8)
    p_lv = [1.5, 1.0, 3.0, 1.0]
    u0_lv = [1.0, 1.0]
    tspan = (0.0, tmax)
    tsteps = range(0.0, tmax, length=n_total)

    prob_lv = DE.ODEProblem(lotka_volterra!, u0_lv, tspan, p_lv)
    sol_lv = DE.solve(prob_lv, DE.Tsit5(), saveat=tsteps)
    ode_data_clean = Array(sol_lv)

    Random.seed!(data_seed)
    ode_data = ode_data_clean .+ σ_obs .* randn(size(ode_data_clean))

    t_train = tsteps[1:n_train]
    t_val   = tsteps[n_train+1:end]
    data_train = ode_data[:, 1:n_train]
    data_val   = ode_data[:, n_train+1:end]
    u0 = ode_data[:, 1]

    dudt2 = Lux.Chain(
        Lux.Dense(2, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 2),
    )
    Random.seed!(init_seed)
    rng = Random.default_rng()
    p, st = Lux.setup(rng, dudt2)
    p_struct = ComponentArrays.ComponentArray{Float64}(p)
    p_flat_nn = vec(collect(p_struct))

    logσ_init = log(0.1)
    p_flat_init = vcat(p_flat_nn, logσ_init)

    println(@sprintf("Problem: σ_obs=%.3f, split=%d/%d, params=%d (+1 logσ)",
                     σ_obs, n_train, n_total - n_train, length(p_flat_nn)))

    return (; tsteps, t_train, t_val, ode_data, ode_data_clean,
              data_train, data_val, u0, dudt2, st, p_struct,
              p_flat_init, logσ_init, n_train, n_total, σ_obs,
              solver_reltol, solver_abstol)
end

# ---------------------------------------------------------------------------
# Solvers, loss, likelihood (closures over `prob`)
# ---------------------------------------------------------------------------
"""
    build_fns(prob)

Return a NamedTuple of functions closing over `prob`: solvers, the weighted MSE
training loss, the Gaussian log-posterior `l(θ)` that NUTS samples, its gradient
`dldθ`, and `validation_metrics`. Kept as closures (not globals) so multiple
problems can coexist without clobbering shared state.
"""
function build_fns(prob)
    dudt2, st = prob.dudt2, prob.st
    p_struct = prob.p_struct
    t_train, t_val = prob.t_train, prob.t_val
    tsteps = prob.tsteps
    data_train, data_val = prob.data_train, prob.data_val
    u0 = prob.u0
    reltol, abstol = prob.solver_reltol, prob.solver_abstol

    unflatten_p(pf) = ComponentArrays.ComponentVector(pf, ComponentArrays.getaxes(p_struct))

    neuralodefunc(u, p, t) = dudt2(u, p, st)[1]

    function solve_train(u0_, p)
        p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
        prob_ = DE.ODEProblem(neuralodefunc, u0_, (t_train[1], t_train[end]), p)
        Array(DE.solve(prob_, DE.Tsit5(), saveat=t_train,
                       abstol=abstol, reltol=reltol, maxiters=Int(1e5)))
    end

    function solve_full_trajectory(u0_, p)
        p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
        prob_ = DE.ODEProblem(neuralodefunc, u0_, (tsteps[1], tsteps[end]), p)
        Array(DE.solve(prob_, DE.Tsit5(), saveat=tsteps,
                       abstol=abstol, reltol=reltol, maxiters=Int(1e5)))
    end

    function solve_valid_forecast(p)
        p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
        u0_val = data_train[:, end]
        prob_ = DE.ODEProblem(neuralodefunc, u0_val, (t_train[end], t_val[end]), p)
        Array(DE.solve(prob_, DE.Tsit5(), saveat=t_val,
                       abstol=abstol, reltol=reltol, maxiters=Int(1e5)))
    end

    predict(p) = solve_full_trajectory(u0, p)

    function loss_neuralode(p)
        pred = solve_train(u0, p)
        w1, w2 = 1.0, W2
        loss = w1 * sum(abs2, data_train[1, :] .- pred[1, :]) +
               w2 * sum(abs2, data_train[2, :] .- pred[2, :])
        return loss, pred
    end

    function validation_metrics(p)
        pred_val = solve_valid_forecast(p)
        mse = mean((data_val .- pred_val) .^ 2)
        rmse = sqrt(mse)
        rel_err = norm(data_val - pred_val) / norm(data_val)
        return mse, rmse, rel_err
    end

    # Gaussian log-posterior (likelihood + N(0,1) weight prior). NUTS samples this.
    function l(θ_flat)
        θ_nn = unflatten_p(θ_flat[1:end-1])
        logσ = θ_flat[end]
        σ_blue = exp(logσ)
        σ_red  = exp(logσ) / sqrt(W2)

        pred = solve_train(u0, θ_nn)
        ll  = -0.5 * sum(((data_train[1, :] .- pred[1, :]) ./ σ_blue) .^ 2)
        ll += -0.5 * sum(((data_train[2, :] .- pred[2, :]) ./ σ_red) .^ 2)
        ll -= length(t_train) * log(σ_blue)
        ll -= length(t_train) * log(σ_red)

        lp  = -0.5 * sum(θ_flat[1:end-1] .^ 2)
        lp -= 0.5 * logσ^2
        return ll + lp
    end

    function dldθ(θ_flat)
        x, back = Zygote.pullback(l, θ_flat)
        grad = first(back(1))
        return x, grad
    end

    return (; unflatten_p, neuralodefunc, solve_train, solve_full_trajectory,
              solve_valid_forecast, predict, loss_neuralode, validation_metrics,
              l, dldθ)
end

# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
"""
    run_map(prob, fns; phaseA_iters, phaseB_iters, lrA, lrB)

Two-phase MAP pre-training (Adam). σ is FROZEN at logσ_init during MAP, so
minimising −l is equivalent to a weight-prior-regularised weighted MSE — the
"MSE-MAP" objective of Appendix A.1. Returns `(p_flat_map, map_metrics)`.
"""
function run_map(prob, fns; phaseA_iters::Int=6000, phaseB_iters::Int=800,
                 lrA::Float64=5e-3, lrB::Float64=1e-3,
                 cliff_thresh::Float64=1.5, cliff_max_loss::Float64=500.0,
                 plateau_thresh::Float64=1e-4, verbose::Bool=true)
    logσ_init = prob.logσ_init
    opt_func = Optimization.OptimizationFunction(
        (x, _) -> -fns.l(vcat(x, logσ_init)),
        Optimization.AutoZygote(),
    )

    # Phase A: discover dynamics (fast LR, cliff-detection stop).
    verbose && println("🔵 Phase A — discover dynamics")
    opt_prob_A = Optimization.OptimizationProblem(opt_func, prob.p_flat_init[1:end-1])
    iterA = Ref(0); prevA = Ref(NaN)
    cbA = function (state, loss_val)
        iterA[] += 1
        if verbose && iterA[] % 100 == 0
            vmse, _, _ = fns.validation_metrics(state.u)
            println(@sprintf("  A iter %d  loss %.4g  valMSE %.4g", iterA[], loss_val, vmse))
        end
        # Cliff fires only if (a) loss dropped by exp(cliff_thresh)× AND (b) the
        # post-cliff loss is already plausibly small (< cliff_max_loss).
        # Condition (b) prevents transient gradient-noise spikes from masquerading
        # as real convergence — the seed-314 failure mode.
        if iterA[] > 200 && !isnan(prevA[]) && prevA[] > 0 && loss_val > 0 &&
           (log(prevA[]) - log(loss_val) > cliff_thresh) &&
           (loss_val < cliff_max_loss)
            verbose && println("  cliff detected — stopping Phase A at iter $(iterA[]) (loss=$(round(loss_val, digits=1)))")
            prevA[] = loss_val; return true
        end
        prevA[] = loss_val; return false
    end
    res_A = Optimization.solve(opt_prob_A, OptimizationOptimisers.Adam(lrA),
                               maxiters=phaseA_iters, callback=cbA)

    # Phase B: widen the basin (slow LR, plateau-detection stop).
    verbose && println("🟢 Phase B — stabilize geometry")
    opt_prob_B = Optimization.OptimizationProblem(opt_func, res_A.u)
    iterB = Ref(0); prevB = Ref(NaN)
    cbB = function (state, loss_val)
        iterB[] += 1
        if verbose && iterB[] % 100 == 0
            vmse, _, _ = fns.validation_metrics(state.u)
            println(@sprintf("  B iter %d  loss %.4g  valMSE %.4g", iterB[], loss_val, vmse))
        end
        if iterB[] > 100 && !isnan(prevB[]) && abs(prevB[] - loss_val) < plateau_thresh
            verbose && println("  plateau detected — stopping Phase B at iter $(iterB[])")
            prevB[] = loss_val; return true
        end
        prevB[] = loss_val; return false
    end
    res_B = Optimization.solve(opt_prob_B, OptimizationOptimisers.Adam(lrB),
                               maxiters=phaseB_iters, callback=cbB)

    p_flat_map = vcat(res_B.u, logσ_init)
    mse, rmse, rel = fns.validation_metrics(fns.unflatten_p(p_flat_map[1:end-1]))
    map_metrics = (; map_loss=res_B.objective, map_val_mse=mse, map_rmse=rmse, map_rel_err=rel)
    verbose && println(@sprintf("MAP done. loss=%.4g  valRMSE=%.4g  relErr=%.3f%%",
                                res_B.objective, rmse, 100rel))
    return p_flat_map, map_metrics
end

"""
    run_node(prob, fns; iters, lr)

Deterministic NeuralODE baseline: plain Adam on the weighted MSE training loss,
NO weight prior and NO noise scale. The conventional (non-Bayesian) NODE the BNODE
is benchmarked against. Returns `(p_flat, metrics)` with no uncertainty.
"""
function run_node(prob, fns; iters::Int=4000, lr::Float64=5e-3, verbose::Bool=true)
    verbose && println("⚪ NeuralODE — deterministic Adam (MSE only)")
    opt_func = Optimization.OptimizationFunction(
        (x, _) -> fns.loss_neuralode(x)[1],
        Optimization.AutoZygote(),
    )
    opt_prob = Optimization.OptimizationProblem(opt_func, prob.p_flat_init[1:end-1])
    it = Ref(0)
    cb = function (state, loss_val)
        it[] += 1
        if verbose && it[] % 200 == 0
            vmse, _, _ = fns.validation_metrics(state.u)
            println(@sprintf("  node iter %d  loss %.4g  valMSE %.4g", it[], loss_val, vmse))
        end
        return false
    end
    res = Optimization.solve(opt_prob, OptimizationOptimisers.Adam(lr),
                             maxiters=iters, callback=cb)
    p_flat = vcat(res.u, prob.logσ_init)
    mse, rmse, rel = fns.validation_metrics(fns.unflatten_p(p_flat[1:end-1]))
    metrics = (; node_train_loss=res.objective, node_val_mse=mse, node_rmse=rmse, node_rel_err=rel)
    verbose && println(@sprintf("NeuralODE done. valRMSE=%.4g  relErr=%.3f%%", rmse, 100rel))
    return p_flat, metrics
end

"""
    run_nuts(prob, fns, p_init; n_samples, n_adapts, target_accept)

Single-chain NUTS from `p_init`. Returns `(samples, stats, runtime_s)` where
`samples` is a (nparams × n_samples) matrix.
"""
function run_nuts(prob, fns, p_init; n_samples::Int=250, n_adapts::Int=250,
                  target_accept::Float64=0.80, max_depth::Int=10)
    D = length(p_init)
    metric = AdvancedHMC.DiagEuclideanMetric(D)
    h = AdvancedHMC.Hamiltonian(metric, fns.l, fns.dldθ)
    integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_init))
    # max_depth caps the NUTS trajectory length: 2^max_depth leapfrog steps per
    # sample. Lower it (e.g. 6) for fast dev passes, restore 10 for camera-ready.
    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integrator, AdvancedHMC.GeneralisedNoUTurn(max_depth=max_depth)))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(target_accept, integrator))

    println(@sprintf("Sampling NUTS: %d samples, %d adapts, max_depth=%d (D=%d)…",
                     n_samples, n_adapts, max_depth, D))
    local samples, stats
    runtime = @elapsed begin
        samples, stats = AdvancedHMC.sample(h, kernel, p_init, n_samples, adaptor, n_adapts;
                                            progress=true)
    end
    return hcat(samples...), stats, runtime
end

# ---------------------------------------------------------------------------
# Diagnostics & metrics
# ---------------------------------------------------------------------------
_field_or(stats, name, default=NaN) =
    try mean(getproperty(s, name) for s in stats) catch; default end

"""EBFMI from the per-sample Hamiltonian energy trace (paper quotes 0.08–0.55)."""
function ebfmi(stats)
    E = try [s.hamiltonian_energy for s in stats] catch; return NaN end
    length(E) < 2 && return NaN
    den = sum((E .- mean(E)) .^ 2)
    den == 0 && return NaN
    return sum(diff(E) .^ 2) / den
end

"""
    nuts_diagnostics(samples, stats)

Acceptance rate, EBFMI, mean tree depth, divergence count, and ESS/Rhat over a
representative parameter subset (first 5 weights + logσ). ESS/Rhat are wrapped in
try/catch so an exploding chain (Arm A) still returns a row.
"""
function nuts_diagnostics(samples, stats)
    accept = _field_or(stats, :acceptance_rate)
    treedepth = _field_or(stats, :tree_depth)
    ndiverge = try sum(s.numerical_error for s in stats) catch; NaN end
    ebf = ebfmi(stats)

    ess_min, rhat_max = NaN, NaN
    try
        n_par, n_samp = size(samples)
        idx = vcat(1:min(5, n_par), n_par)            # 5 weights + logσ
        sub = permutedims(samples[idx, :])            # (samples × params)
        chn = MCMCChains.Chains(reshape(sub, n_samp, length(idx), 1))
        er = MCMCChains.ess_rhat(chn)
        ess_min = minimum(skipmissing(er[:, :ess]))
        rhat_max = maximum(skipmissing(er[:, :rhat]))
    catch e
        @warn "ess_rhat failed (expected if the chain degenerated)" exception=e
    end
    return (; accept, ebfmi=ebf, treedepth, ndiverge, ess_min, rhat_max)
end

_finite_q(v, q) = (f = filter(isfinite, v); length(f) < 5 ? NaN : quantile(f, q))

"""
    analyze_posterior(prob, fns, samples; outdir, label, plot=true)

Posterior predictive over the full trajectory: 90% coverage (Growth/Value),
mean posterior validation MSE, σ̂ mean/std, and plots. NaN/Inf-robust so Arm A
does not crash. Returns a NamedTuple of metrics.
"""
function analyze_posterior(prob, fns, samples; outdir::String, label::String, plot::Bool=true,
                            ch1_label::String="Growth", ch2_label::String="Value")
    tsteps = prob.tsteps
    ode_data = prob.ode_data
    ntime = length(tsteps)
    npost = size(samples, 2)

    latent_g = fill(NaN, ntime, npost); latent_v = similar(latent_g)
    pp_g = similar(latent_g); pp_v = similar(latent_g)
    for (k, col) in enumerate(eachcol(samples))
        θ = fns.unflatten_p(col[1:end-1])
        σ_blue = exp(col[end]); σ_red = σ_blue / sqrt(W2)
        pred = try fns.predict(θ) catch; fill(NaN, 2, ntime) end
        size(pred, 2) == ntime || (pred = fill(NaN, 2, ntime))
        latent_g[:, k] .= pred[1, :]; latent_v[:, k] .= pred[2, :]
        pp_g[:, k] .= pred[1, :] .+ randn(ntime) .* σ_blue
        pp_v[:, k] .= pred[2, :] .+ randn(ntime) .* σ_red
    end

    lo_g = [_finite_q(pp_g[i, :], 0.05) for i in 1:ntime]
    hi_g = [_finite_q(pp_g[i, :], 0.95) for i in 1:ntime]
    lo_v = [_finite_q(pp_v[i, :], 0.05) for i in 1:ntime]
    hi_v = [_finite_q(pp_v[i, :], 0.95) for i in 1:ntime]
    mean_g = [(f = filter(isfinite, pp_g[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:ntime]
    mean_v = [(f = filter(isfinite, pp_v[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:ntime]

    covg(d, lo, hi) = (m = isfinite.(lo) .& isfinite.(hi);
                       any(m) ? mean((d[m] .>= lo[m]) .& (d[m] .<= hi[m])) : NaN)
    coverage_g = covg(ode_data[1, :], lo_g, hi_g)
    coverage_v = covg(ode_data[2, :], lo_v, hi_v)

    σ_samples = exp.(samples[end, :])
    post_mse = [fns.validation_metrics(c[1:end-1])[1] for c in eachcol(samples)]
    post_mse_finite = filter(isfinite, post_mse)

    metrics = (; coverage_g, coverage_v,
                 post_mse_mean = isempty(post_mse_finite) ? NaN : mean(post_mse_finite),
                 sigma_hat_mean = mean(σ_samples), sigma_hat_std = std(σ_samples))

    if plot
        try
            dmin, dmax = minimum(ode_data), maximum(ode_data)
            pad = (dmax - dmin) * 0.2
            pl = Plots.plot(tsteps, mean_g, ribbon=(mean_g .- lo_g, hi_g .- mean_g),
                            fillalpha=0.25, color=:blue, lw=2, label="90% PP CI ($ch1_label)",
                            xlabel="Time", title="$label posterior predictive",
                            ylims=(dmin - pad, dmax + pad))
            Plots.plot!(tsteps, mean_v, ribbon=(mean_v .- lo_v, hi_v .- mean_v),
                        fillalpha=0.25, color=:red, lw=2, label="90% PP CI ($ch2_label)")
            Plots.scatter!(tsteps, ode_data[1, :], color=:blue, alpha=0.6, label="Data: $ch1_label")
            Plots.scatter!(tsteps, ode_data[2, :], color=:red, alpha=0.6, label="Data: $ch2_label")
            Plots.vline!([prob.t_train[end]], color=:black, ls=:dash, label="Train/Val")
            Plots.savefig(pl, joinpath(outdir, "posterior_predictive.png"))

            ps = Plots.scatter(ode_data[1, :], ode_data[2, :], color=:blue, label="Data",
                               xlabel=ch1_label, ylabel=ch2_label, title="$label phase space")
            for _ in 1:min(300, npost)
                resol = samples[1:end-1, rand(1:npost)]
                r = try fns.predict(resol) catch; continue end
                size(r, 2) == ntime && Plots.plot!(r[1, :], r[2, :], alpha=0.04, color=:purple, label="")
            end
            Plots.plot!(mean_g, mean_v, color=:black, lw=2, label="Posterior Mean")
            Plots.savefig(ps, joinpath(outdir, "phase_space.png"))
        catch e
            @warn "posterior predictive / phase plotting failed" exception=e
        end
        # Trace + ACF separately: these need enough samples (skip cleanly otherwise).
        try
            n_show = min(5, size(samples, 1))
            chn = MCMCChains.Chains(reshape(permutedims(samples[1:n_show, :]),
                                            size(samples, 2), n_show, 1))
            Plots.savefig(Plots.plot(chn), joinpath(outdir, "chain_trace.png"))
            Plots.savefig(MCMCChains.autocorplot(chn), joinpath(outdir, "autocor.png"))
        catch e
            @warn "trace/autocorrelation plotting skipped (too few samples?)" exception=e
        end
    end
    return metrics
end

"""Point-estimate fit plot for the no-posterior arms (MAP-only, NeuralODE)."""
function plot_point_fit(prob, fns, p_flat; outdir::String, label::String,
                         ch1_label::String="Growth", ch2_label::String="Value")
    try
        pred = fns.predict(fns.unflatten_p(p_flat[1:end-1]))
        pl = Plots.plot(prob.tsteps, pred[1, :], color=:blue, lw=2, label=ch1_label,
                        xlabel="Time", title="$label point fit")
        Plots.plot!(prob.tsteps, pred[2, :], color=:red, lw=2, label=ch2_label)
        Plots.scatter!(prob.tsteps, prob.ode_data[1, :], color=:blue, alpha=0.5, label="Data: $ch1_label")
        Plots.scatter!(prob.tsteps, prob.ode_data[2, :], color=:red, alpha=0.5, label="Data: $ch2_label")
        Plots.vline!([prob.t_train[end]], color=:black, ls=:dash, label="Train/Val")
        Plots.savefig(pl, joinpath(outdir, "point_fit.png"))
    catch e
        @warn "point-fit plotting failed" exception=e
    end
end

# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------
"""
    append_result!(csv_path, row)

Append one NamedTuple row to the shared benchmark CSV (creating it with a header
on first write). Missing columns across arms are filled so the table stays
rectangular — this CSV is the benchmark table for the paper.
"""
function append_result!(csv_path::String, row::NamedTuple)
    df_new = DataFrames.DataFrame([row])
    if isfile(csv_path)
        df_old = CSV.read(csv_path, DataFrames.DataFrame)
        df = vcat(df_old, df_new; cols=:union)
    else
        df = df_new
    end
    CSV.write(csv_path, df)
    println("→ appended result to $csv_path")
end

const RESULTS_CSV = joinpath(@__DIR__, "results.csv")
ensure_outdir(name) = (d = joinpath(@__DIR__, "outputs", name); mkpath(d); d)
