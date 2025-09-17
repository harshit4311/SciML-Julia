import DifferentialEquations as DE
import Optimization as OPT
import OptimizationPolyalgorithms
import SciMLSensitivity
import ForwardDiff
import Plots

# Define experimental data
t_data = 0:10
x_data = [1.000 2.773 6.773 0.971 1.886 6.101 1.398 1.335 4.353 3.247 1.034]
y_data = [1.000 0.259 2.015 1.908 0.323 0.629 3.458 0.508 0.314 4.547 0.906]
xy_data = vcat(x_data, y_data)

# Plot the provided data
Plots.scatter(t_data, xy_data', label = ["x Data" "y Data"])

# Setup the ODE function
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval
tspan = (0.0, 10.0)

# LV equation parameter. p = [α, β, δ, γ]
pguess = [1.0, 1.2, 2.5, 1.2]

# Set up the ODE problem with our guessed parameter values
prob = DE.ODEProblem(lotka_volterra!, u0, tspan, pguess)

# Solve the ODE problem with our guessed parameter values
initial_sol = DE.solve(prob, saveat = 1)

# View the guessed model solution
plt = Plots.plot(initial_sol, label = ["x Prediction" "y Prediction"])
Plots.scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])

# Define a loss metric function to be minimized
function loss(newp)
    newprob = DE.remake(prob, p = newp)
    sol = DE.solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- xy_data)
    return loss
end

# Define a callback function to monitor optimization progress
function callback(state, l)
    display(l)
    newprob = DE.remake(prob, p = state.u)
    sol = DE.solve(newprob, saveat = 1)
    plt = Plots.plot(sol, ylim = (0, 6), label = ["Current x Prediction" "Current y Prediction"])
    Plots.scatter!(plt, t_data, xy_data', label = ["x Data" "y Data"])
    display(plt)
    return false
end

# Set up the optimization problem with our loss function and initial guess
adtype = OPT.AutoForwardDiff()
pguess = [1.0, 1.2, 2.5, 1.2]
optf = OPT.OptimizationFunction((x, _) -> loss(x), adtype)
optprob = OPT.OptimizationProblem(optf, pguess)

# Optimize the ODE parameters for best fit to our data
pfinal = OPT.solve(optprob, OptimizationPolyalgorithms.PolyOpt(),
    callback = callback,
    maxiters = 200)
α, β, γ, δ = round.(pfinal, digits = 1)