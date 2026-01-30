# dataset with 2 curves (200 datapoints total)
# training on CPU (on my local machine)

# same as exp_20, just increased no.of samples & adapts for HMC

# SciML Libraries
import SciMLSensitivity as SMS
import DifferentialEquations as DE

# ML Tools
import Lux
import Zygote
import Optimization
import OptimizationOptimisers

# External Tools
import Random
Random.seed!(42)

import Plots
import AdvancedHMC
import MCMCChains
import StatsPlots
import ComponentArrays
import CSV
import DataFrames
using Statistics: mean, std, quantile
using LinearAlgebra: norm
# import JLD2 --- (ignore for now)

const w2 = 3.0

# ---------------------------------------
# Generate Lotka-Volterra dataset
# ---------------------------------------
function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# LV equation parameter. p = [α, β, δ, γ]
p_lv = [1.5, 1.0, 3.0, 1.0]

# -----------------------------
# Generate 2 Cycles (200 points)
# -----------------------------
u0_lv = [1.0, 1.0]
tspan = (0.0, 7.0)
tsteps = range(0.0, 7.0, length=200)

prob_lv = DE.ODEProblem(lotka_volterra!, u0_lv, tspan, p_lv)
sol_lv = DE.solve(prob_lv, DE.Tsit5(), saveat=tsteps)
ode_data = Array(sol_lv)

# Update Neural ODE initial condition & datasize
u0 = ode_data[:, 1]
# datasize = length(tsteps) * size(ode_data, 1)

# -------------------------------
# Data Splitting for Forecasting
# -------------------------------
n_total = length(tsteps)
n_train = Int(round(0.7 * n_total))   # 140
n_val   = n_total - n_train           # 60

t_train = tsteps[1:n_train]
t_val   = tsteps[n_train+1:end]

data_train = ode_data[:, 1:n_train]
data_val   = ode_data[:, n_train+1:end]

println("Data split: $n_train training points, $n_val validation points")

# -------------------------------
# Neural ODE definition
# -------------------------------
dudt2 = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 2)
)

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
const _st = st

function neuralodefunc(u, p, t)
    # dudt2(u, p, _st)[1] .* 0.1   # SCALED by 0.1
    # dudt2(u, p, _st)[1]   # NOT scaled
    out = dudt2(u, p, _st)[1]
    # out .* [0.1, 0.2] # give red more room 
    out
end

# (A) Train solver
function solve_train(u0, p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    prob = DE.ODEProblem(neuralodefunc, u0, (t_train[1], t_train[end]), p)
    Array(DE.solve(prob, DE.Tsit5(), saveat=t_train, abstol = 1e-8, reltol = 1e-8, maxiters=1e5))
end

# (B) Forecast solver
function solve_forecast(u0, p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    prob = DE.ODEProblem(neuralodefunc, u0, (tsteps[1], tsteps[end]), p)
    Array(DE.solve(prob, DE.Tsit5(), saveat=tsteps, abstol = 1e-8, reltol = 1e-8, maxiters=1e5))
end

# Alias for compatibility with plotting code (behaves as full forecast)
predict_neuralode(p) = solve_forecast(u0, p)


# -------------------------------
# Parameter handling
# -------------------------------
p_struct = ComponentArrays.ComponentArray{Float64}(p)
p_flat = vec(collect(p_struct))

logσ_init = log(0.1)
p_flat = vcat(p_flat, logσ_init)

function unflatten_p(p_flat)
    ComponentArrays.ComponentVector(p_flat, ComponentArrays.getaxes(p_struct))
end


# -------------------------------
# Prediction & loss
# -------------------------------

function loss_neuralode(p)
    # Use solve_train for loss calculation
    pred = solve_train(u0, p)
    
    w1 = 1.0
    w2 = 3.0  # or 5.0
    # Compare against data_train
    loss = w1 * sum(abs2, data_train[1, :] .- pred[1, :]) + w2 * sum(abs2, data_train[2, :] .- pred[2, :])

    return loss, pred
end

function validation_metrics(p)
    # Solve full trajectory
    pred_full = solve_forecast(u0, p)

    # Extract validation part
    pred_val = pred_full[:, n_train+1:end]

    mse = mean((data_val .- pred_val).^2)
    rmse = sqrt(mse)
    
    # Relative error of the trajectory
    rel_err = norm(data_val - pred_val) / norm(data_val)

    return mse, rmse, rel_err
end



function l(θ_flat)
    θ_nn = unflatten_p(θ_flat[1:end-1])

    logσ = θ_flat[end]
    σ_blue = exp(logσ)
    σ_red  = exp(logσ) / sqrt(w2)

    # Use solve_train
    pred = solve_train(u0, θ_nn)

    # Use data_train
    ll  = -0.5 * sum(((data_train[1,:] .- pred[1,:]) ./ σ_blue).^2)
    ll += -0.5 * sum(((data_train[2,:] .- pred[2,:]) ./ σ_red ).^2)

    # Use length(t_train)
    ll -= length(t_train) * log(σ_blue)
    ll -= length(t_train) * log(σ_red)

    lp  = -0.5 * sum(θ_flat[1:end-1].^2)
    lp -= 0.5 * logσ^2

    return ll + lp
end


function dldθ(θ_flat)
    x, lambda = Zygote.pullback(l, θ_flat)
    grad = first(lambda(1))
    return x, grad
end


# -------------------------------
# MAP Pre-training
# -------------------------------
println("Starting MAP pre-training...")
# Define the optimization function (minimize negative log-posterior)
opt_func = Optimization.OptimizationFunction(
    (x, p) -> -l(vcat(x, logσ_init)),
    Optimization.AutoZygote()
)

# -------------------------------
# Phase A: Discover dynamics
# -------------------------------
println("🔵 Phase A — Discover dynamics")
opt_prob_A = Optimization.OptimizationProblem(opt_func, p_flat[1:end-1])

iter_A = 0
loss_prev_A = NaN

callback_A = function (state, loss_val)
    global iter_A, loss_prev_A
    iter_A += 1
    
    stop = false
    current_loss = loss_val
    
    if iter_A % 100 == 0
        val_mse, val_rmse, _ = validation_metrics(state.u)
        println("Phase A Iteration: $iter_A, Loss: $current_loss, Val MSE: $val_mse")
    end

    # Concrete stopping logic: stop if we find a cliff
    if iter_A > 200 && !isnan(loss_prev_A)
        # log(loss_prev) - log(loss_curr) > 1.5
        # Ensure valid log domain
        if loss_prev_A > 0 && current_loss > 0
             if log(loss_prev_A) - log(current_loss) > 1.5
                 println("Cliff detected — stopping Phase A at iter $iter_A")
                 stop = true
             end
        end
    end
    
    loss_prev_A = current_loss
    return stop
end

# Adam(0.005), Max iters: 8000
res_A = Optimization.solve(opt_prob_A, OptimizationOptimisers.Adam(0.005), maxiters=6700, callback=callback_A)
p_phase_A = res_A.u
println("Phase A complete. Final loss: ", res_A.objective)


# -------------------------------
# Phase B: Stabilize geometry
# -------------------------------
println("🟢 Phase B — Stabilize geometry")
# Initialize with result from Phase A
opt_prob_B = Optimization.OptimizationProblem(opt_func, p_phase_A)

iter_B = 0
loss_prev_B = NaN

callback_B = function (state, loss_val)
    global iter_B, loss_prev_B
    iter_B += 1
    
    stop = false
    current_loss = loss_val

    if iter_B % 100 == 0
         val_mse, val_rmse, _ = validation_metrics(state.u)
         println("Phase B Iteration: $iter_B, Loss: $current_loss, Val MSE: $val_mse")
    end
    
    # STOP when loss plateaus smoothly
    if iter_B > 100 && !isnan(loss_prev_B)
         # Plateau check: small absolute change
         if abs(loss_prev_B - current_loss) < 1e-4
             println("Loss plateau detected — stopping Phase B at iter $iter_B")
             stop = true
         end
    end

    loss_prev_B = current_loss
    return stop
end

# Adam(0.001), Iters: 500-1000
res = Optimization.solve(opt_prob_B, OptimizationOptimisers.Adam(0.001), maxiters=800, callback=callback_B)
p_flat = vcat(res.u, logσ_init)

println("MAP pre-training (Phase B) complete. Final loss: ", res.objective)

# Calculate validation metrics for MAP
map_params_opt = unflatten_p(p_flat[1:end-1])
mse_val, rmse_val, rel_err_val = validation_metrics(map_params_opt)
println("MAP Validation MSE: $mse_val")
println("MAP Validation RMSE: $rmse_val")
println("MAP Validation Rel Err: $rel_err_val")

# -------------------------------
# Plot MAP Result
# -------------------------------
println("Plotting MAP pre-training results...")
map_params = unflatten_p(p_flat[1:end-1])
map_prediction = predict_neuralode(map_params)

pl_map = Plots.plot(tsteps, map_prediction[1,:], color=:blue, xlabel="Time", title="MAP Pre-training Fit")
Plots.plot!(tsteps, map_prediction[2,:], color=:red)
Plots.scatter!(tsteps, ode_data[1, :], color=:blue, alpha=0.5, label="Data")
Plots.scatter!(tsteps, ode_data[2, :], color=:red, alpha=0.5, label="")
Plots.vline!([t_train[end]], label="Train/Val Split", color=:black, linestyle=:dash)
Plots.savefig("map_pretraining_fit.png")
println("Saved map_pretraining_fit.png")


# -------------------------------
# HMC setup
# -------------------------------
n_samples = 50
n_adapts = 100

metric = AdvancedHMC.DiagEuclideanMetric(length(p_flat))
h = AdvancedHMC.Hamiltonian(metric, l, dldθ)
integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_flat))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.80, integrator))

samples, stats = AdvancedHMC.sample(h, kernel, p_flat, n_samples, adaptor, n_adapts; progress = true)


# -------------------------------
# MCMC diagnostics
# -------------------------------
samples = hcat(samples...)
#= comment the following lines(till plot.save autocorr_plot.fig) if we want to run for smaller sample sizes =#
# samples_reduced = samples[1:5, :]
# samples_reshape = reshape(samples_reduced, (n_samples, 5, 1))
n_show = min(5, size(samples, 1))
samples_reduced = samples[1:n_show, :]
samples_reshape = reshape(samples_reduced, (n_samples, n_show, 1))


Chain_Spiral = MCMCChains.Chains(samples_reshape)

Plots.plot(Chain_Spiral)
Plots.savefig("chain_spiral_plot.png")

MCMCChains.autocorplot(Chain_Spiral)
Plots.savefig("autocor_plot.png")


# -------------------------------
# Plot time-series fit with confidence intervals
# -------------------------------

# 1. Generate posterior predictive samples
println("Generating posterior predictive samples...")
n_post = size(samples, 2)
ntime  = length(tsteps)

pp_growth = zeros(ntime, n_post)
pp_value  = zeros(ntime, n_post)

for (k, p_col) in enumerate(eachcol(samples))
    # Unpack parameters
    θ = unflatten_p(p_col[1:end-1])
    logσ_val = p_col[end]
    
    # Calculate sigmas based on the likelihood definition
    σ_blue = exp(logσ_val)
    σ_red  = exp(logσ_val) / sqrt(w2)

    # Solve Neural ODE → u(k)(t)
    pred = predict_neuralode(θ)  # 2 × T

    # Sample observation noise and add to trajectory
    # Using the same noise model as in the likelihood function l(θ)
    ε_growth = randn(ntime) .* σ_blue
    ε_value  = randn(ntime) .* σ_red

    # Posterior predictive draw
    pp_growth[:, k] .= pred[1, :] .+ ε_growth
    pp_value[:, k]  .= pred[2, :] .+ ε_value
end

# 2. Compute posterior predictive bands (90% CI)
lower_growth_pp = [quantile(pp_growth[i, :], 0.05) for i in 1:ntime]
upper_growth_pp = [quantile(pp_growth[i, :], 0.95) for i in 1:ntime]
mean_growth_pp  = vec(mean(pp_growth, dims=2))

lower_value_pp = [quantile(pp_value[i, :], 0.05) for i in 1:ntime]
upper_value_pp = [quantile(pp_value[i, :], 0.95) for i in 1:ntime]
mean_value_pp  = vec(mean(pp_value, dims=2))


# 3. Plot results
# Set y-axis limits based on the actual data range for better visualization
data_min = minimum(ode_data)
data_max = maximum(ode_data)
y_padding = (data_max - data_min) * 0.2 
ylims = (data_min - y_padding, data_max + y_padding)

# Plot Growth
pl = Plots.plot(tsteps, mean_growth_pp, 
    ribbon=(mean_growth_pp .- lower_growth_pp, upper_growth_pp .- mean_growth_pp),
    fillalpha=0.25, color=:blue, label="90% PP CI (Growth)",
    xlabel="Time", title="Russell Predator-Prey Posterior Predictive",
    ylims=ylims, lw=2)

# Plot Value
Plots.plot!(tsteps, mean_value_pp, 
    ribbon=(mean_value_pp .- lower_value_pp, upper_value_pp .- mean_value_pp),
    fillalpha=0.25, color=:red, label="90% PP CI (Value)", lw=2)

# Scatter plot of the original data
Plots.scatter!(tsteps, ode_data[1, :], color = :blue, label = "Data: Growth", alpha=0.6)
Plots.scatter!(tsteps, ode_data[2, :], color = :red, label = "Data: Value", alpha=0.6)

Plots.vline!([t_train[end]], label="Train/Val Split", color=:black, linestyle=:dash)

Plots.savefig("russell_fit_with_ci.png")
println("Saved russell_fit_with_ci.png")


# -------------------------------
# Phase-space plot
# -------------------------------
pl = Plots.scatter(
    ode_data[1, :], ode_data[2, :],
    color = :blue, label = "Data",
    xlabel = "Growth", ylabel = "Value",
    title = "Russell Predator-Prey Phase Space"
)

# Plot posterior samples (latent trajectories)
n_plot_samples = min(300, size(samples, 2))
for k in 1:n_plot_samples
    idx_s = rand(1:size(samples, 2))
    # Extract only NN parameters (exclude logsigma)
    p_nn = samples[1:end-1, idx_s]
    resol = predict_neuralode(p_nn)
    Plots.plot!(resol[1, :], resol[2, :], alpha = 0.04, color = :purple, label = "")
end

# Plot Posterior Mean (using the mean PP computed earlier, which approximates latent mean)
Plots.plot!(mean_growth_pp, mean_value_pp, color = :black, w = 2, label = "Posterior Mean")
Plots.savefig("russell_phase_space.png")


# -------------------------------
# Training Complete! --- Loss statistics & plot 
# -------------------------------

# Compute losses for posterior samples (Training Loss)
train_losses = map(col -> loss_neuralode(col[1:end-1])[1], eachcol(samples))

# Compute validation metrics for posterior samples
val_metrics  = map(col -> validation_metrics(col[1:end-1]), eachcol(samples))
val_mse      = [m[1] for m in val_metrics]

println()
println("----- Training complete! -----")
println()

println("Training Loss statistics across posterior samples:")
println("Min loss: ", minimum(train_losses))
println("Mean loss: ", mean(train_losses))
println("Std dev of loss: ", std(train_losses))

println("Validation MSE statistics across posterior samples:")
println("Min MSE: ", minimum(val_mse))
println("Mean MSE: ", mean(val_mse))
println("Std dev MSE: ", std(val_mse))

Plots.histogram(train_losses, label="Train Loss", alpha=0.5,
    xlabel="Loss", ylabel="Frequency",
    title="Distribution of Loss over Posterior Samples")
Plots.histogram!(val_mse, label="Val MSE", alpha=0.5)
Plots.savefig("loss_distribution.png")