# dataset with 2 curves (200 datapoints total)
# training on CPU (on my local machine)

# same as exp_20, just dividing MAP pre-training into 2 phases

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
datasize = length(tsteps) * size(ode_data, 1)

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
    out .* [0.1, 0.2] # give red more room 
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
    # loss = sum(abs2, ode_data .- pred)
    w1 = 1.0
    w2 = 3.0  # or 5.0
    loss = w1 * sum(abs2, ode_data[1, :] .- pred[1, :]) + w2 * sum(abs2, ode_data[2, :] .- pred[2, :])

    return loss, pred
end


# -------------------------------
# HMC: log-likelihood & gradient
# -------------------------------
# l(θ_flat) = begin
#     θ = unflatten_p(θ_flat)
#     -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
# end

function l(θ_flat)
    θ_nn = unflatten_p(θ_flat[1:end-1])

    logσ = θ_flat[end]
    σ_blue = exp(logσ)
    σ_red  = exp(logσ) / sqrt(w2)

    pred = predict_neuralode(θ_nn)

    ll  = -0.5 * sum(((ode_data[1,:] .- pred[1,:]) ./ σ_blue).^2)
    ll += -0.5 * sum(((ode_data[2,:] .- pred[2,:]) ./ σ_red ).^2)

    ll -= length(tsteps) * log(σ_blue)
    ll -= length(tsteps) * log(σ_red)

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

# --- Phase A: Dynamics Discovery ---
println("--- Phase A: Dynamics Discovery ---")
opt_prob_A = Optimization.OptimizationProblem(opt_func, p_flat[1:end-1])

iter = 0
losses = Float64[]
callback_A = function (state, loss_val)
    global iter, losses
    iter += 1
    push!(losses, loss_val)
    
    if iter % 100 == 0
        println("Phase A Iteration: $iter, Loss: $loss_val")
    end
    
    # Stop early if loss drops by >10x within ~200 iters
    if iter > 200
        prev_loss = losses[iter - 200]
        if prev_loss / loss_val > 10.0
            println("Detected loss cliff (drop > 10x in 200 iters). Stopping Phase A early at iter $iter.")
            return true 
        end
    end
    return false
end

res_A = Optimization.solve(opt_prob_A, OptimizationOptimisers.Adam(0.005), maxiters=7000, callback=callback_A)
p_phase_A = res_A.u
println("Phase A complete.")

# --- Phase B: Geometry Stabilization ---
println("--- Phase B: Geometry Stabilization ---")
opt_prob_B = Optimization.OptimizationProblem(opt_func, p_phase_A)

iter = 0
callback_B = function (state, loss_val)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Phase B Iteration: $iter, Loss: $loss_val")
    end
    return false
end

res_B = Optimization.solve(opt_prob_B, OptimizationOptimisers.Adam(0.001), maxiters=1000, callback=callback_B)
p_flat = vcat(res_B.u, logσ_init)

println("MAP pre-training complete (Phase A + B). Final loss: ", res_B.objective)

# -------------------------------
# Plot MAP Result
# -------------------------------
println("Plotting MAP pre-training results...")
map_params = unflatten_p(p_flat[1:end-1])
map_prediction = predict_neuralode(map_params)

pl_map = Plots.plot(tsteps, map_prediction[1,:], label="Growth Prediction (MAP)", color=:blue, xlabel="Time", title="MAP Pre-training Fit")
Plots.plot!(tsteps, map_prediction[2,:], label="Value Prediction (MAP)", color=:red)
Plots.scatter!(tsteps, ode_data[1, :], color=:blue, label="Data: Growth", alpha=0.5)
Plots.scatter!(tsteps, ode_data[2, :], color=:red, label="Data: Value", alpha=0.5)
Plots.savefig("map_pretraining_fit.png")
println("Saved map_pretraining_fit.png")


# -------------------------------
# HMC setup
# -------------------------------
n_samples = 200
n_adapts = 500

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

# Generate predictions for all posterior samples
# predictions = [predict_neuralode(p) for p in eachcol(samples)];
predictions = [
    predict_neuralode(unflatten_p(p[1:end-1]))
    for p in eachcol(samples)
]
    

# Separate predictions for each species (Growth and Value)
growth_preds = hcat([p[1, :] for p in predictions]...);
value_preds = hcat([p[2, :] for p in predictions]...);

# Calculate quantiles for the 90% confidence interval
# lower_growth = [quantile(growth_preds[i, :], 0.05) for i in 1:datasize];
# upper_growth = [quantile(growth_preds[i, :], 0.95) for i in 1:datasize];
# lower_value = [quantile(value_preds[i, :], 0.05) for i in 1:datasize];
# upper_value = [quantile(value_preds[i, :], 0.95) for i in 1:datasize];
ntime = length(tsteps)

lower_growth = [quantile(growth_preds[i, :], 0.05) for i in 1:ntime]
upper_growth = [quantile(growth_preds[i, :], 0.95) for i in 1:ntime]
lower_value  = [quantile(value_preds[i, :], 0.05) for i in 1:ntime]
upper_value  = [quantile(value_preds[i, :], 0.95) for i in 1:ntime]

# Find the best fit prediction (lowest loss)
losses = map(x -> loss_neuralode(x)[1], eachcol(samples));
idx = findmin(losses)[2];
# best_fit_prediction = predict_neuralode(samples[:, idx]);
best_fit_prediction = predict_neuralode(unflatten_p(samples[1:end-1, idx]))


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