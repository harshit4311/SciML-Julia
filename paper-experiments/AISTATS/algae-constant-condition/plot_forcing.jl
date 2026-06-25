#=
plot_forcing.jl — visualise the external nutrient inflow Nᵢₙ(t) per experiment.

The point: C8/C9 are driven by a square-wave Nᵢₙ (0 ↔ 160, ~8-day period) while the
others are held constant. An autonomous neural ODE can fit the latter but not the
former. Output: outputs/forcing.png (a grid, one panel per experiment).

Run:  julia --project=../../.. plot_forcing.jl
=#

import CSV
import DataFrames
import Plots

const DATA = joinpath(@__DIR__, "..", "algae-chemostat", "data", "blasius_rotifer_algae.csv")
const OUT  = joinpath(@__DIR__, "outputs"); mkpath(OUT)

df = CSV.read(DATA, DataFrames.DataFrame; missingstring="")
expts = sort(unique(df.experiment))

panels = map(expts) do e
    sub = df[df.experiment .== e, :]
    ok = .!ismissing.(sub.medium_N)
    d  = Float64.(sub.day[ok]); n = Float64.(sub.medium_N[ok])
    forced = length(unique(round.(n))) > 1
    Plots.plot(d, n; lw=1.5, legend=false, grid=false,
               color = forced ? :crimson : :steelblue,
               title = "C$e " * (forced ? "(FORCED)" : "(constant)"),
               titlefontsize=8, ylabel="Nᵢₙ", xlabel="day",
               ylims=(-10, 175), seriestype=:steppre)
end

Plots.savefig(Plots.plot(panels...; layout=(5, 2), size=(1100, 1300),
              plot_title="External nutrient inflow Nᵢₙ(t): C8/C9 are square-wave forced",
              plot_titlefontsize=11), joinpath(OUT, "forcing.png"))
println("→ outputs/forcing.png")
