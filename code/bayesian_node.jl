# Step 1: Import Libraries
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

# Setup: Get the data from the Spiral ODE example
u0 = [2.0; 0.0]
datasize = 40
tspan = (0.0, 1)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = DE.ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(DE.solve(prob_trueode, DE.Tsit5(), saveat = tsteps))

# Step 2: Define the Neural ODE architecture.
dudt2 = Lux.Chain(x -> x .^ 3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2))

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
const _st = st
function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1]
end
function prob_neuralode(u0, p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    sol = DE.solve(prob, DE.Tsit5(), saveat = tsteps)
end
p = ComponentArrays.ComponentArray{Float64}(p)
const _p = p

# Step 3: Define the loss function for the Neural ODE.
function predict_neuralode(p)
    p = p isa ComponentArrays.ComponentArray ? p : convert(typeof(_p), p)
    Array(prob_neuralode(u0, p))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Step 4: Integrate the Bayesian estimation workflow (AdvancedHMC)
# Define log density and its gradient. The θ*θ term denotes Gaussian priors.
l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
function dldθ(θ)
    x, lambda = Zygote.pullback(l, θ)
    grad = first(lambda(1))
    return x, grad
end

metric = AdvancedHMC.DiagEuclideanMetric(length(p))
h = AdvancedHMC.Hamiltonian(metric, l, dldθ)

# NUTS sampler setup and sampling parameters
integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.45, integrator))
samples, stats = AdvancedHMC.sample(h, kernel, p, 500, adaptor, 500; progress = true)

# Step 5: Plot diagnostics
samples = hcat(samples...)
samples_reduced = samples[1:5, :]
samples_reshape = reshape(samples_reduced, (500, 5, 1))
Chain_Spiral = MCMCChains.Chains(samples_reshape)
Plots.plot(Chain_Spiral)

# Autocorrelation plot
MCMCChains.autocorplot(Chain_Spiral)

# Retrodiction / time-series plotting over posterior samples
pl = Plots.scatter(tsteps, ode_data[1, :], color = :red, label = "Data: Var1", xlabel = "t",
    title = "Spiral Neural ODE")
Plots.scatter!(tsteps, ode_data[2, :], color = :blue, label = "Data: Var2")
for k in 1:300
    resol = predict_neuralode(samples[:, 100:end][:, rand(1:400)])
    Plots.plot!(tsteps, resol[1, :], alpha = 0.04, color = :red, label = "")
    Plots.plot!(tsteps, resol[2, :], alpha = 0.04, color = :blue, label = "")
end
