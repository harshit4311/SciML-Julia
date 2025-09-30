# SciML Libraries
import SciMLSensitivity as SMS
import DifferentialEquations as DE

# ML Tools
import Lux
import Zygote

# External Tools
import Random
import Plots
import AdvancedHMC
import MCMCChains
import StatsPlots
import ComponentArrays
import CSV
import DataFrames
using Statistics: mean, std
# import JLD2 --- (ignore for now)

# -------------------------------
# Load Russell dataset
# -------------------------------
df = CSV.read("/Users/harshit/Downloads/Research-Commons-Quant/SciML-Julia/russell-dataset/russell_predator_prey.csv", DataFrames.DataFrame)
ode_data = [df.growth_ret'; df.value_ret']

# Initial conditions and data
u0 = ode_data[:, 1]
datasize = size(ode_data, 2)
tspan = (0.0, Float64(datasize-1))
tsteps = range(tspan[1], tspan[2], length = datasize)


# -------------------------------
# Neural ODE definition
# -------------------------------
dudt2 = Lux.Chain(
    x -> x .^ 3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
const _st = st

function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1]
end

function prob_neuralode(u0, p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    DE.solve(prob, DE.Rodas5(), saveat = tsteps, maxiters=1e5)
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
    -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ_flat .* θ_flat)
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
n_adapts = 50

metric = AdvancedHMC.DiagEuclideanMetric(length(p_flat))
h = AdvancedHMC.Hamiltonian(metric, l, dldθ)
integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_flat))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.70, integrator))

samples, stats = AdvancedHMC.sample(h, kernel, p_flat, n_samples, adaptor, n_adapts; progress = true)


# -------------------------------
# MCMC diagnostics
# -------------------------------
samples = hcat(samples...)
samples_reduced = samples[1:5, :]
samples_reshape = reshape(samples_reduced, (n_samples, 5, 1))

Chain_Spiral = MCMCChains.Chains(samples_reshape)

Plots.plot(Chain_Spiral)
Plots.savefig("chain_spiral_plot.png")

MCMCChains.autocorplot(Chain_Spiral)
Plots.savefig("autocor_plot.png")


# -------------------------------
# Plot time-series fit
# -------------------------------
pl = Plots.scatter(
    tsteps, ode_data[1, :],
    color = :blue, label = "Data: Growth",
    xlabel = "Time", title = "Russell Predator-Prey Neural ODE"
)
Plots.scatter!(
    tsteps, ode_data[2, :],
    color = :red, label = "Data: Value"
)

for k in 1:300
    resol = predict_neuralode(samples[:, 1:end][:, rand(1:size(samples, 2))])
    Plots.plot!(tsteps, resol[1, :], alpha = 0.04, color = :blue, label = "")
    Plots.plot!(tsteps, resol[2, :], alpha = 0.04, color = :red, label = "")
end

losses = map(x -> loss_neuralode(x)[1], eachcol(samples))
idx = findmin(losses)[2]
prediction = predict_neuralode(samples[:, idx])

Plots.plot!(tsteps, prediction[1, :], color = :black, w = 2, label = "")
Plots.plot!(tsteps, prediction[2, :], color = :black, w = 2, label = "Best fit prediction")
Plots.savefig("lotka_volterra_fit.png")


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

Plots.plot!(prediction[1, :], prediction[2, :], color = :black, w = 2, label = "Best fit prediction")
Plots.savefig("lotka_volterra_phase.png")