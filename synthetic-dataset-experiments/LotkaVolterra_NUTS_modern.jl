# Modernized version of Raj's code for current Julia environment
# Uses Lux instead of FastChain, and manual Flux loop instead of sciml_train

using DifferentialEquations
using Flux
using Lux
using Plots
using AdvancedHMC
using MCMCChains
using Random
using ComponentArrays
using Statistics
using Distributions
using Zygote
using SciMLSensitivity


# ---------------------------------------
# 1. Data Generation
# ---------------------------------------
function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 3.5)
tsteps = 0.0:0.1:3.5
datasize = length(tsteps)

# LV equation parameter. p = [α, β, δ, γ]
p_lv = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p_lv)
mean_ode_data = Array(solve(prob_ode, Tsit5(), saveat = tsteps))

# Generate 30 realizations of noisy data (as in Raj's code)
# ode_data shape: (2, 36, 30)
ode_data = mean_ode_data .+ 0.1 .* randn(size(mean_ode_data)..., 30)


# ---------------------------------------
# 2. Neural ODE Definition (Lux)
# ---------------------------------------
# Architecture: 2 -> 20 -> 20 -> 20 -> 2
dudt2 = Lux.Chain(
    Lux.Dense(2, 20, Lux.relu),
    Lux.Dense(20, 20, Lux.relu),
    Lux.Dense(20, 20, Lux.relu),
    Lux.Dense(20, 2)
)

rng = Random.default_rng()
p_model, st = Lux.setup(rng, dudt2)
const _st = st

function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1]
end

prob_neuralode = ODEProblem(neuralodefunc, u0, tspan, p_model)

function predict_neuralode(p)
    # Solve ODE
    # Note: We use saveat=tsteps to match data points
    Array(solve(prob_neuralode, Tsit5(), p=p, saveat=tsteps))
end


# ---------------------------------------
# 3. Parameter Handling
# ---------------------------------------
p_struct = ComponentArrays.ComponentArray{Float64}(p_model)
p_flat = vec(collect(p_struct))

# Augment parameters with prior_std (Raj's θ[end])
# We initialize it to 1.0
p_combined = [p_flat; 1.0]

function unflatten_p(p_flat)
    ComponentArrays.ComponentVector(p_flat, ComponentArrays.getaxes(p_struct))
end


# ---------------------------------------
# 4. Log-Posterior
# ---------------------------------------
function logposterior(θ_combined)
    # θ_combined = [weights...; prior_std]
    weights_flat = θ_combined[1:end-1]
    prior_std = θ_combined[end]
    
    # Ensure prior_std is positive
    sigma = abs(prior_std) + 1e-6 
    
    weights = unflatten_p(weights_flat)
    
    # 1. Prior: Weights ~ N(0, sigma^2)
    lp = logpdf(MvNormal(zeros(length(weights_flat)), sigma), weights_flat)
    
    # 2. Likelihood: Data ~ N(pred, 1) (Implicitly, since sum(abs2) is used)
    # Sum over all 30 realizations
    pred = predict_neuralode(weights)
    ll = -sum(abs2, ode_data .- pred)
    
    return lp + ll
end

function dlogposterior(θ_combined)
    val, back = Zygote.pullback(logposterior, θ_combined)
    grad = first(back(1))
    return val, grad
end


# ---------------------------------------
# 5. Pre-training (MAP)
# ---------------------------------------
println("Starting MAP pre-training with ADAM...")

opt = Flux.ADAM(0.05)
p_opt = copy(p_combined)
opt_state = Flux.setup(opt, p_opt)

# Raj used 1500 iterations
for i in 1:1500
    val, grad = dlogposterior(p_opt)
    Flux.update!(opt_state, p_opt, -grad)
    if i % 100 == 0
        println("Epoch $i: Log-posterior = $val")
    end
end

p_min = p_opt
println("Pre-training complete.")


# ---------------------------------------
# 6. HMC Sampling
# ---------------------------------------
n_samples = 200
n_adapts = 200

metric = AdvancedHMC.DiagEuclideanMetric(length(p_min))
h = AdvancedHMC.Hamiltonian(metric, logposterior, dlogposterior)
integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_min))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.5, integrator))

println("Starting HMC sampling...")
samples, stats = AdvancedHMC.sample(h, kernel, p_min, n_samples, adaptor, n_adapts; progress=true)


# ---------------------------------------
# 7. Analysis & Plotting
# ---------------------------------------
samples_mat = hcat(samples...) # (n_params, n_samples)

# Calculate losses for samples
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

losses = [loss_neuralode(unflatten_p(samples[i][1:end-1])) for i in 1:length(samples)]

# Plot Loss
scatter(losses, ylabel = "Loss", label = "Architecture1: 500 warmup, 500 sample")
savefig("ExtendedLV_Loss_100_500_Arch1.pdf")

# Plot Retrodicted Data (using first realization of data for visualization)
pl = scatter(tsteps, mean_ode_data[1,:], color = :red, label = "Data: Var1", title = "Lotka Volterra Neural ODE")
scatter!(tsteps, mean_ode_data[2,:], color = :blue, label = "Data: Var2", xlabel = "t", ylims = (0, 10))

# Plot posterior samples
for _ in 1:300
    idx = rand(1:length(samples))
    p_samp = unflatten_p(samples[idx][1:end-1])
    resol = predict_neuralode(p_samp)
    plot!(tsteps, resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps, resol[2,:], alpha=0.04, color = :blue, label = "")
end

# Best fit
idx_best = findmin(losses)[2]
p_best = unflatten_p(samples[idx_best][1:end-1])
prediction = predict_neuralode(p_best)

plot!(tsteps, prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps, prediction[2,:], color = :black, w = 2, label = "Best fit prediction")
savefig("ExtendedLV_Fit.pdf")

# Contour / Phase Space
pl = scatter(
    mean_ode_data[1,:],
    mean_ode_data[2,:],
    color = :blue, label = "Data",  xlabel = "Var1",
    ylabel = "Var2", title = "Lotka Volterra Neural ODE",
    legend = (0.85, 0.95), legendfontsize = 5,
)

# Plot all 30 data realizations
for k in 1:size(ode_data, 3)
    scatter!(
        ode_data[1,:,k],
        ode_data[2,:,k],
        color = :blue, label = "",
    )
end

# Plot posterior predictive
# Use the last 50% of samples or whatever is available
start_idx = max(1, length(samples) - 100)
for k1 in start_idx:length(samples)
    p_samp = unflatten_p(samples[k1][1:end-1])
    sigma_samp = abs(samples[k1][end]) # Ensure positive
    
    resol = predict_neuralode(p_samp)
    
    for k2 in 1:10
        _resol = resol .+ sigma_samp .* randn(size(resol))
        plot!(_resol[1,:], _resol[2,:], alpha=0.04, color = :red, label = "")
    end
end

plot!(prediction[1,:], prediction[2,:], color = :red, w = 2, label = "Simulated data")
plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction")

savefig("ExtendedLV_Contour_Retrodicted_500_500_Arch2.pdf")

println("Done!")