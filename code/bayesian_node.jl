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

# Define the Lotka-Volterra equations
function lotka_volterra_func(du, u, p, t)
    α, β, δ, γ = 1.5, 1.0, 3.0, 1.0
    du[1] = dR = α * u[1] - β * u[1] * u[2] # Prey (Rabbits)
    du[2] = dW = -δ * u[2] + γ * u[1] * u[2] # Predator (Wolves)
end

# Set initial conditions, time span, and generate data
u0 = [1.0; 1.0]
datasize = 100
tspan = (0.0, 10.0)
tsteps = range(tspan[1], tspan[2], length = datasize)

prob_trueode = DE.ODEProblem(lotka_volterra_func, u0, tspan)
ode_data = Array(DE.solve(prob_trueode, DE.Tsit5(), saveat = tsteps))

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

# Save the structure for unflattening
p_struct = ComponentArrays.ComponentArray{Float64}(p)
p_flat = vec(collect(p_struct))

function unflatten_p(p_flat)
    return ComponentArrays.ComponentVector(p_flat, ComponentArrays.getaxes(p_struct))
end

function predict_neuralode(p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Log-likelihood and gradient for HMC (accepts flat vector)
l(θ_flat) = begin
    θ = unflatten_p(θ_flat)
    -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ_flat .* θ_flat)
end

function dldθ(θ_flat)
    x, lambda = Zygote.pullback(l, θ_flat)
    grad = first(lambda(1))
    return x, grad
end

metric = AdvancedHMC.DiagEuclideanMetric(length(p_flat))
h = AdvancedHMC.Hamiltonian(metric, l, dldθ)

integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_flat))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.2, integrator))
samples, stats = AdvancedHMC.sample(h, kernel, p_flat, 500, adaptor, 500; progress = true)

samples = hcat(samples...)
samples_reduced = samples[1:5, :]
samples_reshape = reshape(samples_reduced, (500, 5, 1))
Chain_Spiral = MCMCChains.Chains(samples_reshape)
Plots.plot(Chain_Spiral)
Plots.savefig("chain_spiral_plot.png")

MCMCChains.autocorplot(Chain_Spiral)
Plots.savefig("autocor_plot.png")

pl = Plots.scatter(tsteps, ode_data[1, :], color = :blue, label = "Data: Rabbits", xlabel = "Time",
    title = "Lotka-Volterra Neural ODE")
Plots.scatter!(tsteps, ode_data[2, :], color = :red, label = "Data: Wolves")
for k in 1:300
    resol = predict_neuralode(samples[:, 100:end][:, rand(1:400)])
    Plots.plot!(tsteps, resol[1, :], alpha = 0.04, color = :blue, label = "")
    Plots.plot!(tsteps, resol[2, :], alpha = 0.04, color = :red, label = "")
end

losses = map(x -> loss_neuralode(x)[1], eachcol(samples))
idx = findmin(losses)[2]
prediction = predict_neuralode(samples[:, idx])
Plots.plot!(tsteps, prediction[1, :], color = :black, w = 2, label = "")
Plots.plot!(tsteps, prediction[2, :], color = :black, w = 2, label = "Best fit prediction")
Plots.savefig("lotka_volterra_fit.png")

pl = Plots.scatter(ode_data[1, :], ode_data[2, :], color = :blue, label = "Data", xlabel = "Rabbits",
    ylabel = "Wolves", title = "Lotka-Volterra Phase Space")
for k in 1:300
    resol = predict_neuralode(samples[:, 100:end][:, rand(1:400)])
    Plots.plot!(resol[1, :], resol[2, :], alpha = 0.04, color = :purple, label = "")
end
Plots.plot!(prediction[1, :], prediction[2, :], color = :black, w = 2,
    label = "Best fit prediction")
Plots.savefig("lotka_volterra_phase.png")