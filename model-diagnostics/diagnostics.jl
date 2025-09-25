using MCMCChains, Plots, StatsPlots, JLD2

# Load the MCMC samples and stats
data = load("model-diagnostics/mcmc_results.jld2")
samples = data["samples"]
stats = data["stats"]
tsteps = data["tsteps"]
ode_data = data["ode_data"]
predict_neuralode = include("../code/bayesian_node.jl").predict_neuralode


println("--- MCMC Diagnostics ---")

# Create a Chains object
# The shape of samples is (n_params, n_samples), we need to transpose it
# and add a dimension for the chain
samples_reshaped = reshape(samples', size(samples, 2), size(samples, 1), 1)
chain = MCMCChains.Chains(samples_reshaped)

# 1. Summary Statistics
println("Summary Statistics:")
display(summarystats(chain))
println("\n")

# 2. Geweke Diagnostic
println("Geweke Diagnostic:")
display(gewekediag(chain))
println("\n")

# 3. Effective Sample Size
println("Effective Sample Size:")
display(ess(chain))
println("\n")

# 4. Trace and Autocorrelation plots
println("Generating Trace and Autocorrelation plots...")
trace_plot = Plots.plot(chain)
Plots.savefig(trace_plot, "model-diagnostics/trace_plot.png")

autocor_plot = MCMCChains.autocorplot(chain)
Plots.savefig(autocor_plot, "model-diagnostics/autocor_plot.png")
println("Plots saved in model-diagnostics/")


# 5. Posterior Predictive Check
println("Generating Posterior Predictive Check plot...")
pl = Plots.scatter(
    tsteps, ode_data[1, :],
    color = :blue, label = "Data: Rabbits",
    xlabel = "Time", title = "Posterior Predictive Check"
)
Plots.scatter!(
    tsteps, ode_data[2, :],
    color = :red, label = "Data: Wolves"
)

# Plot some posterior samples
for k in 1:100
    sample_col = samples[:, rand(1:size(samples, 2))]
    resol = predict_neuralode(sample_col)
    Plots.plot!(tsteps, resol[1, :], alpha = 0.1, color = :blue, label = "")
    Plots.plot!(tsteps, resol[2, :], alpha = 0.1, color = :red, label = "")
end
Plots.savefig("model-diagnostics/posterior_predictive_check.png")
println("Posterior predictive check plot saved.")

# 6. Parameter Posteriors (for a subset of parameters)
println("Generating posterior distribution plots for first 5 parameters...")
param_posterior_plot = corner(chain, 1:5)
Plots.savefig(param_posterior_plot, "model-diagnostics/parameter_posteriors.png")
println("Parameter posterior plot saved.")

println("Diagnostics complete.")
