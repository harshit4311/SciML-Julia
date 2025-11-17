# gpu_bayesian_node_lightning.jl
# GPU forward solves (posterior predictions) + CPU HMC (AdvancedHMC)
# Single-file runnable script. Designed to avoid device-mismatch issues.

# used T4 & L40s GPUs on lightning.ai

using CSV, DataFrames
using Random
using Statistics: mean, std, quantile
using Plots
using ComponentArrays
using MCMCChains
using StatsPlots  
using AdvancedHMC
using DiffEqBase
import SciMLSensitivity
import DifferentialEquations as DE
using Lux
using Zygote
using CUDA

# -------------------------------
# Global clean setup
# -------------------------------
ENV["LUX_DEBUG_MODE"] = "false"
# Disable Enzyme-based VJPs if available (to avoid sensitivity-related warnings)
if isdefined(SciMLSensitivity, :ENZYME_VJP)
    SciMLSensitivity.ENZYME_VJP[] = false
end

CUDA.allowscalar(false)                # Disallow accidental CPU fallbacks on GPU

# -------------------------------
# Quick GPU status
# -------------------------------
println("=== GPU status ===")
try
    @show CUDA.has_cuda()
    @show CUDA.functional()
    @show CUDA.device()
catch e
    println("CUDA check failed: ", e)
end
println("==================")

# -------------------------------
# Load dataset
# -------------------------------
csv_path = "russell-datasets/ideal_synthetic_LV_russell_growth_value_200.csv"
df = CSV.read(csv_path, DataFrame)
growth_f = Float32.(df.Growth)
value_f  = Float32.(df.Value)

datasize = length(growth_f)
tsteps = collect(0f0:Float32(1):Float32(datasize-1))   # keep same spacing as your earlier script
tspan = (tsteps[1], tsteps[end])

# stack data as 2 x T (for convenience)
data_cpu = hcat(growth_f, value_f)'   # 2 x datasize
u0_cpu = data_cpu[:, 1]               # initial condition on CPU

# -------------------------------
# Build Lux neural network (CPU)
# -------------------------------
rng = Random.default_rng()
dudt_cpu = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 2)
)

p_init_cpu, st_cpu = Lux.setup(rng, dudt_cpu)     # CPU params & state

# Flatten cpu parameter structure into Float32 vector for HMC parameterization
p_struct_cpu = ComponentArrays.ComponentArray{Float32}(p_init_cpu |> Lux.cpu_device())
p_flat_cpu = vec(collect(p_struct_cpu))           # Float32 vector that represents parameters
n_params = length(p_flat_cpu)
println("n_params = ", n_params)

# unflatten helper (rebuild same ComponentVector axes)
function unflatten_p_cpu(flatvec::AbstractVector{<:AbstractFloat})
    ComponentArrays.ComponentVector(flatvec, ComponentArrays.getaxes(p_struct_cpu))
end

# -------------------------------
# CPU neural ODE function (out-of-place, plain CPU)
# This is used for HMC (gradients via Zygote)
# -------------------------------
function neuralode_cpu!(du, u, p, t)
    # u : Vector{Float32} (2-element)
    # p : ComponentVector for parameters (CPU)
    # Evaluate network (Lux returns tuple (output, state) sometimes)
    out = dudt_cpu(u, p, st_cpu)   # returns a tuple-like object
    dud = out[1]
    du .= dud .* 0.1f0
end

# CPU ODE solve wrapper returning 2 x datasize matrix (Float32)
function solve_cpu_forward(p_flat::Vector{Float32}; saveat = tsteps)
    p_comp = unflatten_p_cpu(p_flat)
    prob = DE.ODEProblem(neuralode_cpu!, u0_cpu, tspan, p_comp)
    sol = DE.solve(prob, DE.Rodas5(), saveat = saveat, reltol=1e-6, abstol=1e-8, maxiters=Int(1e7))
    # convert solution to 2 x T matrix (Float32)
    mat = hcat([Array(sol[i]) for i in 1:length(sol)]...)  # each sol[i] is vector length 2
    return mat
end

# -------------------------------
# Loss / log-likelihood (CPU)
# We'll use a simple Gaussian observation model with sigma_obs
# -------------------------------
sigma_obs = 0.1f0

function loglik_cpu(p_flat64::Vector{Float64})
    # AdvancedHMC expects Float64 vectors. Convert to Float32 for CPU solver.
    p32 = Float32.(p_flat64)
    pred = solve_cpu_forward(p32; saveat=tsteps)
    # pred is 2 x T
    # compute negative sum-of-squares log-likelihood (up to constant)
    resid = (data_cpu .- pred) ./ sigma_obs
    ll = -0.5 * sum(resid .^ 2)
    # add a modest gaussian prior on parameters (zero-mean, sigma 10)
    ll -= 0.5 * sum((p_flat64 ./ 10.0) .^ 2)
    return ll
end

# wrapper for AdvancedHMC objective (returns Float64 scalar)
function target_logprob(θ::Vector{Float64})
    return loglik_cpu(θ)
end

# gradient using Zygote (on the Float64 input) — AdvancedHMC expects gradient w.r.t Float64
function target_and_gradient(θ::Vector{Float64})
    val, back = Zygote.pullback(target_logprob, θ)
    g = first(back(1.0))
    return val, g
end

# A small adapter returning (value) for AdvancedHMC.. but we'll supply gradient separately.
function l_scalar(θ::Vector{Float64})
    return target_logprob(θ)
end

function dldθ(θ::Vector{Float64})
    val, grad = target_and_gradient(θ)
    return val, grad
end

# -------------------------------
# HMC / AdvancedHMC setup (CPU)
# -------------------------------
p0_64 = Float64.(p_flat_cpu)   # initial point in Float64
dim = length(p0_64)
n_samples = 4   # small for example; increase for real inference
n_adapts  = 10

metric = AdvancedHMC.DiagEuclideanMetric(dim)
ham = AdvancedHMC.Hamiltonian(metric, l_scalar, dldθ)

# find a reasonable step size
println("Finding good stepsize (this calls l & gradient a few times)...")
step_size = AdvancedHMC.find_good_stepsize(ham, p0_64)
integrator = AdvancedHMC.Leapfrog(step_size)
trajectory = AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn())
kernel = AdvancedHMC.HMCKernel(trajectory)
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.8, integrator))

println("Starting HMC sampling: n_samples=", n_samples, ", n_adapts=", n_adapts)
samples64, stats = AdvancedHMC.sample(ham, kernel, p0_64, n_samples, adaptor, n_adapts; progress = true)
println("HMC finished")

# samples64 is a Vector{Vector{Float64}} (n_samples vectors). Convert into matrix
samples64_mat = hcat(samples64...)   # dim x n_samples
samples32 = Float32.(samples64_mat)  # Float32 for reuse with CPU solver if desired

# -------------------------------
# Simple diagnostics (MCMCChains)
# -------------------------------
# Build minimal chains object for a small subset of parameters (first 5 param dims)
n_show = min(5, size(samples32, 1))
samples_reduced = samples32[1:n_show, :]
# reshape to (n_iter, n_chains, n_params_shown)
samples_reshape = reshape(samples_reduced', (n_samples, 1, n_show))
chain = MCMCChains.Chains(samples_reshape)
plot(chain) |> savefig("chain_spiral_plot.png")
MCMCChains.autocorplot(chain) |> savefig("autocor_plot.png")

# -------------------------------
# Posterior predictions (GPU) — move parameters to GPU and run solves without adjoint
# -------------------------------
# Build a GPU copy of the network (a separate network pinned for GPU prediction)
dudt_gpu = dudt_cpu |> Lux.gpu_device()
# Setup GPU parameter structure (initial)
p_init_gpu, st_gpu = Lux.setup(rng, dudt_gpu)   # this gives shapes consistent with dudt_gpu
# We'll use unflatten->pack mapping from CPU flat vector into component vector matching p_struct_cpu,
# then move each subarray to GPU before calling dudt_gpu.

# helper: convert CPU flat (ComponentVector) to GPU ComponentVector with same axes
function cpuflat_to_gpu_component(p_flat_vec::AbstractVector{Float32})
    cpu_comp = unflatten_p_cpu(p_flat_vec)  # CPU ComponentVector
    # convert each array in cpu_comp into cu() equivalent
    gpu_comp = map(x -> cu(Array(x)), cpu_comp)  # ensures arrays are moved to GPU
    return gpu_comp
end

# GPU solve wrapper: sol -> 2 x T matrix on CPU (collect at end)
function solve_gpu_forward(p_flat::Vector{Float32}; saveat = tsteps)
    # build GPU parameter comp vector
    p_gpu_comp = cpuflat_to_gpu_component(p_flat)
    # define in-place GPU f!
    function neuralode_gpu!(du, u, p, t)
        # u, du are CuArray{Float32}
        out = dudt_gpu(u, p_gpu_comp, st_gpu)
        dud = out[1]
        du .= dud .* 0.1f0
    end
    u0_gpu = cu(u0_cpu)
    prob_gpu = DE.ODEProblem(neuralode_gpu!, u0_gpu, tspan, nothing)
    # Important: set sensealg = Nothing to avoid adjoint/AD attempts on GPU
    sol = DE.solve(prob_gpu, DE.Rodas5(), saveat = saveat, sensealg = nothing, reltol=1e-6, abstol=1e-8, maxiters = Int(1e7))
    # collect results to CPU
    mat_cpu = hcat([Array(sol[i]) for i in 1:length(sol)]...)
    return mat_cpu
end

# choose a few posterior samples (or all) for prediction
nsamps = size(samples32, 2)
take_idx = 1: max(1, Int(round(nsamps/ min(nsamps,50)))) : nsamps  # subsample up to ~50 preds
chosen_idx = collect(take_idx)
println("Posterior sample count for GPU prediction: ", length(chosen_idx))

predictions = Vector{Matrix{Float32}}(undef, length(chosen_idx))
for (i, j) in enumerate(chosen_idx)
    p_flat_sample = vec(samples32[:, j])
    @info "Running GPU forward for sample $i / $(length(chosen_idx))"

    # Run forward solve on GPU
    predictions[i] = solve_gpu_forward(p_flat_sample; saveat = tsteps)

    # Force synchronization so GPU work completes before moving on
    CUDA.synchronize()

    # Optional: print timing info
    @info "Finished GPU sample $i"
end


# Build CI bands across predictions
# For each time index, compute quantiles
growth_preds = hcat([pred[1, :] for pred in predictions]...)  # T x nsamps_pred? actually each pred[1,:] is length T
value_preds  = hcat([pred[2, :] for pred in predictions]...)

lower_growth = [quantile(growth_preds[i, :], 0.05) for i in 1:datasize]
upper_growth = [quantile(growth_preds[i, :], 0.95) for i in 1:datasize]
median_growth = [quantile(growth_preds[i, :], 0.50) for i in 1:datasize]

lower_value = [quantile(value_preds[i, :], 0.05) for i in 1:datasize]
upper_value = [quantile(value_preds[i, :], 0.95) for i in 1:datasize]
median_value = [quantile(value_preds[i, :], 0.50) for i in 1:datasize]

# find best-fit sample by CPU loss evaluation for clarity
losses = [ sum((data_cpu .- solve_cpu_forward(vec(samples32[:, j]))) .^ 2) for j in 1:nsamps ]
min_loss, min_idx = findmin(losses)
best_fit_cpu_pred = solve_cpu_forward(vec(samples32[:, min_idx]))   # 2 x T

# -------------------------------
# Save plots
# -------------------------------
# Time-series with CI
plt = plot(Float32.(tsteps), median_growth, ribbon = (median_growth .- lower_growth, upper_growth .- median_growth), label = "Growth median", xlabel = "Time", ylabel = "Value", title = "Growth & Value predictions (median + 90% CI)")
plot!(Float32.(tsteps), median_value, ribbon = (median_value .- lower_value, upper_value .- median_value), label = "Value median")
scatter!(Float32.(tsteps), data_cpu[1, :], label = "Data: Growth", alpha=0.7)
scatter!(Float32.(tsteps), data_cpu[2, :], label = "Data: Value", alpha=0.7)
savefig(plt, "russell_fit_with_ci.png")

# Phase space plot
plt2 = scatter(data_cpu[1, :], data_cpu[2, :], label = "Data", xlabel = "Growth", ylabel = "Value", title="Phase space")
for k in 1:length(predictions)
    plot!(predictions[k][1, :], predictions[k][2, :], alpha=0.06, label = "")
end
plot!(best_fit_cpu_pred[1, :], best_fit_cpu_pred[2, :], color=:black, linewidth=2, label="Best fit (CPU eval)")
savefig(plt2, "russell_phase_space.png")

# Loss histogram
plt3 = histogram(losses, bins=20, label="Loss", xlabel="Loss", ylabel="Frequency", title="Loss distribution (posterior samples)")
savefig(plt3, "loss_distribution.png")

println("Done. Saved plots: russell_fit_with_ci.png, russell_phase_space.png, loss_distribution.png")
