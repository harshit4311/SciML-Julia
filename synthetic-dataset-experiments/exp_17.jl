# dataset with 2 curves (200 datapoints total)
# training on CPU (on my local machine)

# same as Exp_1, just updated code to save HMC checkpoint and resume from it

# SciML Libraries
import SciMLSensitivity as SMS
import DifferentialEquations as DE

# ML Tools
import Lux
import Zygote

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
import JLD2

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
datasize = length(tsteps)

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
    dudt2(u, p, _st)[1] .* 0.1   # SCALED by 0.1
    # dudt2(u, p, _st)[1]   # NOT scaled
end

function prob_neuralode(u0, p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    DE.solve(prob, DE.Tsit5(), saveat=tsteps, maxiters=1e5)
end


# -------------------------------
# Parameter handling
# -------------------------------
p_struct = ComponentArrays.ComponentArray{Float64}(p)
p_flat = vec(collect(p_struct))

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
l(θ_flat) = begin
    θ = unflatten_p(θ_flat)
    # Normalize total loss by datasize to keep gradient magnitude consistent
    # (-sum(abs2, ode_data .- predict_neuralode(θ)) - 0.01 * sum(θ_flat .* θ_flat)) / datasize
    -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
end

function dldθ(θ_flat)
    x, lambda = Zygote.pullback(l, θ_flat)
    grad = first(lambda(1))
    return x, grad
end


# -------------------------------
# HMC setup
# -------------------------------
n_samples = 50

CHECKPOINT_FILE = "hmc_checkpoint.jld2"
CONTINUE_FROM_CHECKPOINT = true

if CONTINUE_FROM_CHECKPOINT && isfile(CHECKPOINT_FILE)
    println("Loading checkpoint from $CHECKPOINT_FILE...")
    checkpoint = JLD2.load(CHECKPOINT_FILE)
    existing_samples = checkpoint["samples"]
    initial_theta = existing_samples[end]
    saved_step_size = get(checkpoint, "step_size", nothing)
    saved_metric = get(checkpoint, "metric", nothing)
    println("Resuming from last sample...")
    n_adapts = 0
else
    initial_theta = p_flat
    existing_samples = nothing
    saved_step_size = nothing
    saved_metric = nothing
    println("Starting fresh sampling...")
    n_adapts = 200
end

if saved_metric !== nothing
    println("Using saved metric...")
    metric = saved_metric
else
    metric = AdvancedHMC.DiagEuclideanMetric(length(p_flat))
end

h = AdvancedHMC.Hamiltonian(metric, l, dldθ)

if saved_step_size !== nothing
    println("Using saved step size: $saved_step_size")
    integrator = AdvancedHMC.Leapfrog(saved_step_size)
else
    println("Finding good step size...")
    integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, initial_theta))
end

kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))

if n_adapts > 0
    adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.80, integrator))
else
    adaptor = AdvancedHMC.NoAdaptation()
end

samples, stats = AdvancedHMC.sample(h, kernel, initial_theta, n_samples, adaptor, n_adapts; progress = true)

# Enforce type safety
samples = Vector{Vector{Float64}}(samples)

if existing_samples !== nothing
    samples = vcat(existing_samples, samples)
end

# Extract final step size from stats
final_step_size = integrator.ϵ

println("Saving checkpoint to $CHECKPOINT_FILE...")
JLD2.save(
    CHECKPOINT_FILE, "samples", 
    samples, "step_size", 
    final_step_size, "metric", metric
    )


# -------------------------------
# MCMC diagnostics
# -------------------------------
samples = hcat(samples...)
total_samples = size(samples, 2)
#= comment the following lines(till plot.save autocorr_plot.fig) if we want to run for smaller sample sizes =#
samples_reduced = samples[1:5, :]
# Correctly format for MCMCChains: (samples, params, chains)
samples_t = permutedims(samples_reduced, (2, 1))
samples_reshape = reshape(samples_t, (total_samples, 5, 1))

Chain_Spiral = MCMCChains.Chains(samples_reshape)

Plots.plot(Chain_Spiral)
Plots.savefig("chain_spiral_plot.png")

MCMCChains.autocorplot(Chain_Spiral)
Plots.savefig("autocor_plot.png")


# -------------------------------
# Plot time-series fit with confidence intervals
# -------------------------------

# Generate predictions for all posterior samples
predictions = [predict_neuralode(p) for p in eachcol(samples)];

# Separate predictions for each species (Growth and Value)
growth_preds = hcat([p[1, :] for p in predictions]...);
value_preds = hcat([p[2, :] for p in predictions]...);

# Calculate quantiles for the 90% confidence interval
lower_growth = [quantile(growth_preds[i, :], 0.05) for i in 1:datasize];
upper_growth = [quantile(growth_preds[i, :], 0.95) for i in 1:datasize];
lower_value = [quantile(value_preds[i, :], 0.05) for i in 1:datasize];
upper_value = [quantile(value_preds[i, :], 0.95) for i in 1:datasize];

# Find the best fit prediction (lowest loss)
losses = map(x -> loss_neuralode(x)[1], eachcol(samples));
idx = findmin(losses)[2];
best_fit_prediction = predict_neuralode(samples[:, idx]);

# Set y-axis limits based on the actual data range for better visualization
data_min = minimum(ode_data)
data_max = maximum(ode_data)
y_padding = (data_max - data_min) * 0.1 # Add 10% padding
ylims = (data_min - y_padding, data_max + y_padding)

# Plot the results
pl = Plots.plot(tsteps, best_fit_prediction[1,:], ribbon=(best_fit_prediction[1,:] .- lower_growth, upper_growth .- best_fit_prediction[1,:]),
    fillalpha=0.2, color=:blue, label="Growth Prediction",
    xlabel="Time", title="Russell Predator-Prey Neural ODE",
    ylims=ylims);
Plots.plot!(tsteps, best_fit_prediction[2,:], ribbon=(best_fit_prediction[2,:] .- lower_value, upper_value .- best_fit_prediction[2,:]),
    fillalpha=0.2, color=:red, label="Value Prediction");

# Scatter plot of the original data
Plots.scatter!(tsteps, ode_data[1, :], color = :blue, label = "Data: Growth");
Plots.scatter!(tsteps, ode_data[2, :], color = :red, label = "Data: Value");

Plots.savefig("russell_fit_with_ci.png");


# -------------------------------
# Phase-space plot
# -------------------------------
pl = Plots.scatter(
    ode_data[1, :], ode_data[2, :],
    color = :blue, label = "Data",
    xlabel = "Growth", ylabel = "Value",
    title = "Russell Predator-Prey Phase Space"
)

for k in 1:300
    resol = predict_neuralode(samples[:, 1:end][:, rand(1:size(samples, 2))])
    Plots.plot!(resol[1, :], resol[2, :], alpha = 0.04, color = :purple, label = "")
end

# Plots.plot!(prediction[1, :], prediction[2, :], color = :black, w = 2, label = "Best fit prediction")
Plots.plot!(best_fit_prediction[1, :], best_fit_prediction[2, :], color = :black, w = 2, label = "Best fit prediction")
Plots.savefig("russell_phase_space.png")


# -------------------------------
# Training Complete! --- Loss statistics & plot 
# -------------------------------

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