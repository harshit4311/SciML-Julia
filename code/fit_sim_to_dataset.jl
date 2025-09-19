#############################
# Lotka–Volterra Parameter Fitting in Julia
#############################

using DifferentialEquations
using Optimization
using OptimizationPolyalgorithms
using SciMLSensitivity
using ForwardDiff
using Plots

# ---------------------------
# 1️⃣ Experimental data
# ---------------------------
t_data = 0:10
x_data = [1.000, 2.773, 6.773, 0.971, 1.886, 6.101, 1.398, 1.335, 4.353, 3.247, 1.034]
y_data = [1.000, 0.259, 2.015, 1.908, 0.323, 0.629, 3.458, 0.508, 0.314, 4.547, 0.906]

# Combine into 11×2 matrix for easier handling
xy_data = hcat(x_data, y_data)

# Quick scatter plot of data
scatter(t_data, xy_data, label=["x Data" "y Data"], xlabel="Time", ylabel="Population", title="Experimental Data")

# ---------------------------
# 2️⃣ Define Lotka–Volterra ODE
# ---------------------------
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# Initial guess for parameters [α, β, δ, γ]
pguess = [1.0, 1.2, 2.5, 1.2]

# Setup ODE problem
prob = ODEProblem(lotka_volterra!, u0, tspan, pguess)

# Solve with initial guess
initial_sol = solve(prob, saveat=1)

# Plot initial guess solution with data
plt = plot(initial_sol, label=["x Prediction" "y Prediction"], xlabel="Time", ylabel="Population", title="Initial Guess vs Data")
scatter!(plt, t_data, xy_data, label=["x Data" "y Data"])
display(plt)

# ---------------------------
# 3️⃣ Define loss function
# ---------------------------
function loss(newp)
    newprob = remake(prob, p=newp)
    sol = solve(newprob, saveat=1)
    sol_array = hcat(sol.u...)'
    return sum(abs2, sol_array .- xy_data)
end

# ---------------------------
# 4️⃣ Define callback to monitor optimization
# ---------------------------
function callback(state, l)
    display(l)
    newprob = remake(prob, p=state.u)
    sol = solve(newprob, saveat=1)
    sol_array = hcat(sol.u...)'
    plt = plot(sol_array, ylim=(0,6), label=["Current x Prediction" "Current y Prediction"], xlabel="Time", ylabel="Population", title="Optimization Progress")
    scatter!(plt, t_data, xy_data, label=["x Data" "y Data"])
    display(plt)
    return false
end

# ---------------------------
# 5️⃣ Set up optimization
# ---------------------------
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, _) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pguess)

# ---------------------------
# 6️⃣ Run optimization
# ---------------------------
pfinal = Optimization.solve(optprob, OptimizationPolyalgorithms.PolyOpt(),
                            callback=callback,
                            maxiters=200)

# Extract and round parameters
α, β, δ, γ = round.(pfinal.u, digits=2)
println("Optimized parameters:")
println("α = $α, β = $β, δ = $δ, γ = $γ")

# ---------------------------
# 7️⃣ Plot final fitted solution
# ---------------------------
final_prob = remake(prob, p=pfinal.u)
final_sol = solve(final_prob, saveat=1)
plt_final = plot(final_sol, label=["x Prediction" "y Prediction"], xlabel="Time", ylabel="Population", title="Fitted Lotka-Volterra Model")
scatter!(plt_final, t_data, xy_data, label=["x Data" "y Data"])
display(plt_final)
