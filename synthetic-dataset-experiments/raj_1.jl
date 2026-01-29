# dataset with 2 curves (200 datapoints total)
# training on CPU (on my local machine)

# Raj's predator-prey dataset and code - NO validation loss 

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
# import JLD2 --- (ignore for now)


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
# Generate Dataset (36 points) with Noise
# -----------------------------
# Using settings from reference snippet
u0_lv = [1.0, 1.0]
tspan = (0.0, 3.5)
tsteps = 0.0:0.1:3.5

prob_lv = DE.ODEProblem(lotka_volterra!, u0_lv, tspan, p_lv)
sol_lv = DE.solve(prob_lv, DE.Tsit5(), saveat=tsteps)
clean_ode_data = Array(sol_lv)

# Add noise (0.1 sigma)
ode_data = clean_ode_data .+ 0.1 .* randn(size(clean_ode_data))

# Update Neural ODE initial condition & datasize
# Use true initial condition for training (as in reference)
u0 = u0_lv 
datasize = length(tsteps) * size(ode_data, 1)

# -------------------------------
# Neural ODE definition
# -------------------------------
dudt2 = Lux.Chain(
    Lux.Dense(2, 20, Lux.relu),
    Lux.Dense(20, 20, Lux.relu),
    Lux.Dense(20, 20, Lux.relu),
    Lux.Dense(20, 2)
)

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
const _st = st

function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1]   # NO scaling
end

function prob_neuralode(u0, p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    DE.solve(prob, DE.Tsit5(), saveat=tsteps, abstol = 1e-8, reltol = 1e-8, maxiters=1e5)
end


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
function predict_neuralode(p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end


# -------------------------------
# HMC: log-likelihood & gradient
# -------------------------------
function l(θ_flat)
    θ_nn = unflatten_p(θ_flat[1:end-1])
    logσ = θ_flat[end]
    σ = exp(logσ)

    pred = predict_neuralode(θ_nn)

    # Likelihood with normalization
    ll = -sum(abs2, ode_data .- pred) / (2σ^2) - length(ode_data) * logσ

    # Prior
    lp = -sum(θ_flat[1:end-1].^2) / 2  # standard Gaussian prior on weights
    lp += -0.5 * logσ^2                # standard Gaussian prior on logσ

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
    (x, p) -> -l(x),
    Optimization.AutoZygote()
)

opt_prob = Optimization.OptimizationProblem(opt_func, p_flat)

iter = 0
callback = function (state, loss_val)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Iteration: $iter, Loss: $loss_val")
    end
    return false
end

# Adam(0.01), Max iters: 1500
res = Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.01), maxiters=1500, callback=callback)
p_flat = res.u

println("MAP pre-training complete. Final loss: ", res.objective)

# -------------------------------
# Plot MAP Result
# -------------------------------
println("Plotting MAP pre-training results...")
map_params = unflatten_p(p_flat[1:end-1])
map_prediction = predict_neuralode(map_params)

pl_map = Plots.plot(tsteps, map_prediction[1,:], color=:blue, xlabel="Time", title="MAP Pre-training Fit")
Plots.plot!(tsteps, map_prediction[2,:], color=:red)
Plots.scatter!(tsteps, ode_data[1, :], color=:blue, alpha=0.5)
Plots.scatter!(tsteps, ode_data[2, :], color=:red, alpha=0.5)
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
    σ = exp(logσ_val)

    # Solve Neural ODE → u(k)(t)
    pred = predict_neuralode(θ)  # 2 × T

    # Sample observation noise and add to trajectory
    # Using the same noise model as in the likelihood function l(θ)
    ε_growth = randn(ntime) .* σ
    ε_value  = randn(ntime) .* σ

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

# Compute losses for posterior samples
losses = map(col -> loss_neuralode(col[1:end-1])[1], eachcol(samples))

println()
println("----- Training complete! -----")
println()

println("Loss statistics across posterior samples:")
println("Min loss: ", minimum(losses))
println("Mean loss: ", mean(losses))
println("Std dev of loss: ", std(losses))

Plots.histogram(losses, label="Loss Distribution",
    xlabel="Loss", ylabel="Frequency",
    title="Distribution of Loss over Posterior Samples")
Plots.savefig("loss_distribution.png")


# -------------------------------
# Extrapolation / Forecasting
# -------------------------------
println("Generating extrapolation plots...")

# Extrapolation settings
tspan_extrap = (0.0, 4.5)
tsteps_extrap = 0.0:0.1:4.5 

function predict_neuralode_extrap(p, tspan, tsteps)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    # Re-create problem with extrapolation tspan
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    sol = DE.solve(prob, DE.Tsit5(), saveat=tsteps, abstol = 1e-8, reltol = 1e-8, maxiters=1e5)
    return Array(sol)
end

# Generate posterior predictive for extrapolation
ntime_extrap = length(tsteps_extrap)
pp_growth_extrap = zeros(ntime_extrap, n_post)
pp_value_extrap  = zeros(ntime_extrap, n_post)

for (k, p_col) in enumerate(eachcol(samples))
    θ = unflatten_p(p_col[1:end-1])
    logσ_val = p_col[end]
    σ = exp(logσ_val)
    
    # Prediction on extended range
    pred = predict_neuralode_extrap(θ, tspan_extrap, tsteps_extrap)
    
    # Add noise
    ε_growth = randn(ntime_extrap) .* σ
    ε_value  = randn(ntime_extrap) .* σ
    
    if size(pred, 2) == ntime_extrap
        pp_growth_extrap[:, k] .= pred[1, :] .+ ε_growth
        pp_value_extrap[:, k]  .= pred[2, :] .+ ε_value
    end
end

# Compute CI for extrapolation
lower_growth_ex = [quantile(pp_growth_extrap[i, :], 0.05) for i in 1:ntime_extrap]
upper_growth_ex = [quantile(pp_growth_extrap[i, :], 0.95) for i in 1:ntime_extrap]
mean_growth_ex  = vec(mean(pp_growth_extrap, dims=2))

lower_value_ex = [quantile(pp_value_extrap[i, :], 0.05) for i in 1:ntime_extrap]
upper_value_ex = [quantile(pp_value_extrap[i, :], 0.95) for i in 1:ntime_extrap]
mean_value_ex  = vec(mean(pp_value_extrap, dims=2))

# Plotting Extrapolation
# Calculate true solution for extrapolation comparison
prob_lv_extrap = DE.ODEProblem(lotka_volterra!, u0_lv, tspan_extrap, p_lv)
sol_lv_extrap = DE.solve(prob_lv_extrap, DE.Tsit5(), saveat=tsteps_extrap)
true_data_extrap = Array(sol_lv_extrap)

pl_ex = Plots.plot(tsteps_extrap, mean_growth_ex, 
    ribbon=(mean_growth_ex .- lower_growth_ex, upper_growth_ex .- mean_growth_ex),
    fillalpha=0.25, color=:blue, label="90% PP CI (Growth)",
    xlabel="Time", title="Extrapolation: Russell Predator-Prey",
    lw=2, legend=:topright)
Plots.plot!(tsteps_extrap, mean_value_ex, 
    ribbon=(mean_value_ex .- lower_value_ex, upper_value_ex .- mean_value_ex),
    fillalpha=0.25, color=:red, label="90% PP CI (Value)", lw=2)

# Scatter training data
Plots.scatter!(tsteps, ode_data[1, :], color = :black, label = "Training Data", alpha=0.6, markershape=:circle)
Plots.scatter!(tsteps, ode_data[2, :], color = :black, label = "", alpha=0.6, markershape=:rect)

# Plot True Dynamics (Ground Truth)
Plots.plot!(tsteps_extrap, true_data_extrap[1, :], color=:black, linestyle=:dash, label="True Growth")
Plots.plot!(tsteps_extrap, true_data_extrap[2, :], color=:black, linestyle=:dot, label="True Value")

# Mark extrapolation start
Plots.vline!([3.5], color=:black, label="Training End")

Plots.savefig("russell_extrapolation.png")
println("Saved russell_extrapolation.png")

