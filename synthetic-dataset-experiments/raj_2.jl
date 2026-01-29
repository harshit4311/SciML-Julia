# dataset with 2 curves (200 datapoints total)
# training on CPU (on my local machine)

# Raj's predator-prey dataset and code - Added validation loss here

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

# -----------------------------
# Train / Validation Split (time-based)
# -----------------------------
train_ratio = 0.8
n_train = Int(round(length(tsteps) * train_ratio))

tsteps_train = tsteps[1:n_train]
tsteps_val   = tsteps[n_train+1:end]

ode_data_train = ode_data[:, 1:n_train]
ode_data_val   = ode_data[:, n_train+1:end]

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
function predict_neuralode_on_grid(p, tspan_local, tgrid)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan_local, p)
    sol = DE.solve(prob, DE.Tsit5(), saveat=tgrid, abstol=1e-8, reltol=1e-8, maxiters=1e5)
    return Array(sol)
end

function predict_neuralode(p)
    # Wrapper for full range (for plotting compatibility)
    predict_neuralode_on_grid(p, tspan, tsteps)
end

function loss_train(p)
    pred = predict_neuralode_on_grid(p, (tsteps_train[1], tsteps_train[end]), tsteps_train)
    return sum(abs2, ode_data_train .- pred)
end


# -------------------------------
# HMC: log-likelihood & gradient
# -------------------------------
function l(θ_flat)
    θ_nn = unflatten_p(θ_flat[1:end-1])
    logσ = θ_flat[end]
    σ = exp(logσ)

    pred = predict_neuralode_on_grid(θ_nn, (tsteps_train[1], tsteps_train[end]), tsteps_train)

    # Likelihood with normalization (Training Data)
    ll = -sum(abs2, ode_data_train .- pred) / (2σ^2) - length(ode_data_train) * logσ

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
# Improved Validation Metrics
# -------------------------------
function validation_loss_nll_correct(p_col)
    θ = unflatten_p(p_col[1:end-1])
    logσ = p_col[end]
    σ = exp(logσ)

    # initial condition = last TRAIN state (ground truth)
    u0_val = ode_data_train[:, end]

    # Start simulation from the time of the last training point
    # We must bridge the gap between train_end and val_start
    t_start = tsteps_train[end]
    t_end   = tsteps_val[end]
    tspan_val = (t_start, t_end)

    prob = DE.ODEProblem(neuralodefunc, u0_val, tspan_val, θ)
    # We solve over the gap, but saveat only the validation points
    sol = DE.solve(prob, DE.Tsit5(), saveat=tsteps_val,
                   abstol=1e-8, reltol=1e-8, maxiters=1e5)
    
    # Check if solve succeeded for full range
    if length(sol.t) != length(tsteps_val)
         return Inf 
    end
    
    pred_val = Array(sol)

    return mean(((ode_data_val .- pred_val).^2) ./ (2σ^2) .+ log(σ))
end

function validation_loss_dynamics(p_col)
    θ = unflatten_p(p_col[1:end-1])
    
    u0_val = ode_data_train[:, end]
    tspan_val = (tsteps_train[end], tsteps_val[end])

    prob = DE.ODEProblem(neuralodefunc, u0_val, tspan_val, θ)
    sol = DE.solve(prob, DE.Tsit5(), saveat=tsteps_val,
                   abstol=1e-8, reltol=1e-8, maxiters=1e5)
    
    if length(sol.t) != length(tsteps_val)
         return Inf
    end
    pred_val = Array(sol)

    # Pure MSE (Dynamics Error, ignoring sigma)
    return mean((ode_data_val .- pred_val).^2)
end

function bayesian_validation_nll(samples)
    vals = [validation_loss_nll_correct(col) for col in eachcol(samples)]
    return mean(vals)
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
        # Compute validation NLL (expensive, so only do it periodically)
        val_nll = validation_loss_nll_correct(state.u)
        println("Iteration: $iter, Train Loss (NegPost): $(round(loss_val, digits=4)), Val NLL: $(round(val_nll, digits=4))")
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
Plots.vline!([tsteps_train[end]], color=:black, label="Train End")
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
# Training Complete! --- Loss statistics & plot 
# -------------------------------

# Compute losses for posterior samples
train_losses = map(col -> loss_train(col[1:end-1]), eachcol(samples))
val_losses_nll = map(col -> validation_loss_nll_correct(col), eachcol(samples))
val_losses_mse = map(col -> validation_loss_dynamics(col), eachcol(samples))

println()
println("----- Training complete! -----")
println()

println("Train Loss statistics:")
println("Min loss: ", minimum(train_losses))
println("Mean loss: ", mean(train_losses))
println("Std dev of loss: ", std(train_losses))

println("Validation Loss (NLL - Bayesian Correct) statistics:")
println("Min val loss: ", minimum(val_losses_nll))
println("Mean val loss: ", mean(val_losses_nll))
println("Std val loss: ", std(val_losses_nll))

println("Validation MSE (Dynamics Only) statistics:")
println("Min val MSE: ", minimum(val_losses_mse))
println("Mean val MSE: ", mean(val_losses_mse))

println("Bayesian Validation NLL (Expectation): ", mean(val_losses_nll))

Plots.histogram(train_losses, label="Train Loss", xlabel="Loss", ylabel="Frequency", title="Posterior Loss Distribution")
Plots.vline!([mean(val_losses_nll)], label="Mean Val NLL", color=:red)
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

