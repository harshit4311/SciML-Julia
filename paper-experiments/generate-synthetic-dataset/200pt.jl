using DifferentialEquations
using CSV
using DataFrames
using Random

# LV system
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α*x - β*x*y
    du[2] = -δ*y + γ*x*y
end

# Parameters
p_lv = [1.5, 1.0, 3.0, 1.0]

# Initial condition + time
u0_lv = [1.0, 1.0]
tspan = (0.0, 7.0)
tsteps = range(0.0, 7.0, length=200)

# Solve ODE
prob = ODEProblem(lotka_volterra!, u0_lv, tspan, p_lv)
sol = solve(prob, saveat=tsteps)

# Extract solution
t = sol.t
x = [u[1] for u in sol.u]
y = [u[2] for u in sol.u]

df = DataFrame(
    time = t,
    prey = x,
    predator = y
)

Random.seed!(42)

noise_level = 0.2

df.prey_noisy = df.prey .+ noise_level .* randn(length(df.prey))
df.predator_noisy = df.predator .+ noise_level .* randn(length(df.predator))

CSV.write("lotka_volterra_dataset.csv", df)
