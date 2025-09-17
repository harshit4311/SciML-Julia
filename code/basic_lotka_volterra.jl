using Pkg
Pkg.activate(".")

using DifferentialEquations
using ModelingToolkit
using Plots
using ModelingToolkit: t_nounits as t, D_nounits as D,
                       @variables, @parameters, @named, @mtkbuild

# Define variables
@variables x(t)=1 y(t)=1 z(t)
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define equations
eqs = [
    D(x) ~ α * x - β * x * y,
    D(y) ~ -γ * y + δ * x * y,
    z ~ x + y
]

# Build ODE system
@mtkbuild sys = ODESystem(eqs, t)

# Problem + solve
tspan = (0.0, 10.0)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob)

# Make plots
p1 = plot(sol, title = "Rabbits vs Wolves")
p2 = plot(sol, idxs = z, title = "Total Animals")

# Combine into final figure
finalplot = plot(p1, p2, layout = (2, 1))

# Save figure
savefig(finalplot, "rabbits_vs_wolves.png")

println("✅ Plot saved as rabbits_vs_wolves.png")
