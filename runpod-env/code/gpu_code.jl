# code for Float64-compatible GPUs on runpod 

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
using Statistics: mean, std, quantile
import CUDA

# -------------------------------
# Load Russell dataset
# -------------------------------
# df = CSV.read("/root/SciML-Julia/russell-datasets/ideal_synthetic_LV_russell_growth_value_200.csv", DataFrames.DataFrame)
df = CSV.read("dataset/ideal_synthetic_LV_russell_growth_value_200.csv", DataFrames.DataFrame)

# Extract columns
growth = df.Growth
value  = df.Value

# Convert to Neural ODE input format
ode_data_cpu = Float64.([growth'; value'])
ode_data = CUDA.cu(ode_data_cpu)

# Initial condition & timespan
u0 = ode_data[:, 1]
datasize = size(ode_data, 2)
tspan = (0.0, Float64(datasize - 1))
tsteps = range(tspan[1], tspan[2], length = datasize)

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
p = ComponentArrays.ComponentArray{Float64}(p) # Ensure initial p is Float64
st = CUDA.cu(st)
const _st = st

function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1] .* 0.1   # SCALED by 0.1
    # dudt2(u, p, _st)[1]   # NOT scaled
end

function prob_neuralode(u0, p)
    prob = DE.ODEProblem(neuralodefunc, u0, tspan, p)
    # Using Vern7 (explicit, high order) which is often more robust for Neural ODEs than Tsit5
    # Increased maxiters to 50000 to allow for more steps if dynamics are complex
    # Relaxed tolerances slightly to aid convergence
    DE.solve(prob, DE.Vern7(), saveat = tsteps, maxiters=50000, abstol=1e-5, reltol=1e-5, sensealg=SMS.InterpolatingAdjoint(autojacvec=SMS.ZygoteVJP()))
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
    
    # Safe transfer/conversion that works with Zygote
    # We use Zygote.ignore to prevent Zygote from trying to differentiate the GPU transfer logic
    # which involves low-level CUDA calls that Zygote cannot handle.
    p_gpu = Zygote.ignore() do
        CUDA.cu(Float64.(p))
    end

    sol = prob_neuralode(u0, p_gpu)
    
    # Check for solver failure (e.g. MaxIters reached)
    # We use Zygote.ignore to safely check the retcode without confusing the AD
    failure = Zygote.ignore() do
        # Check if solver failed to produce the expected number of time steps
        # This handles MaxIters, instability, etc.
        length(sol.u) != length(tsteps) || (sol.retcode != :Success && sol.retcode != :Terminated)
    end

    if failure
        Zygote.ignore() do
             # Only print occasionally to avoid spamming if many fail
             if rand() < 0.1 
                 println("  [Warn] ODE Solver failed/maxiters. Rejecting sample.")
                 flush(stdout)
             end
        end
        # Return an infinite prediction to force high loss and rejection by HMC
        return CUDA.fill(Inf, size(ode_data))
    end

    reduce(hcat, sol.u)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return Float64(loss), pred
end


# -------------------------------
# HMC: log-likelihood & gradient
# -------------------------------
function l_gpu(θ_flat)
    θ = unflatten_p(θ_flat)
    pred = predict_neuralode(θ)
    -sum(abs2, ode_data .- pred) - 0.01 * sum(abs2, θ_flat)
end

l(θ_flat) = begin
    θ_gpu = CUDA.cu(θ_flat)
    Float64(l_gpu(θ_gpu))
end

# Global counter for GC
gc_counter = 0

function dldθ(θ_flat)
    global gc_counter
    gc_counter += 1
    # Trigger GC every 50 gradient evaluations to prevent system memory accumulation
    if gc_counter % 50 == 0
        GC.gc()
    end
    
    # Ensure input is Float64 before moving to GPU
    θ_gpu = CUDA.cu(Float64.(θ_flat))
    val, back = Zygote.pullback(l_gpu, θ_gpu)
    grad = back(1.0)[1]

    # Print progress to show activity
    if gc_counter % 10 == 0
        println("Gradient evaluation: $gc_counter | Log-density: $val")
        flush(stdout)
    end

    return Float64(val), Array{Float64}(Array(grad))
end


# -------------------------------
# HMC setup
# -------------------------------
println("Initializing HMC...")
flush(stdout)

n_samples = 50
n_adapts = 50

metric = AdvancedHMC.DiagEuclideanMetric(length(p_flat))
h = AdvancedHMC.Hamiltonian(metric, l, dldθ)
integrator = AdvancedHMC.Leapfrog(AdvancedHMC.find_good_stepsize(h, p_flat))
kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn()))
adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.8, integrator))

println("Starting Warmup (Compilation)...")
println("  1. Compiling forward pass...")
flush(stdout)
try
    l(p_flat)
    println("     Forward pass compiled.")
catch e
    println("     Forward pass warmup warning: ", e)
end

println("  2. Compiling backward pass (this may take a minute)...")
flush(stdout)
try
    dldθ(p_flat)
    println("     Backward pass compiled.")
catch e
    println("     Backward pass warmup warning: ", e)
end

println("Warmup complete. Starting Sampling...")
flush(stdout)

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
# Plot time-series fit with confidence intervals
# -------------------------------

# Generate predictions for all posterior samples
predictions = [Array(predict_neuralode(p)) for p in eachcol(samples)];

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
best_fit_prediction = Array(predict_neuralode(samples[:, idx]));

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
    resol = Array(predict_neuralode(samples[:, 1:end][:, rand(1:size(samples, 2))]))
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