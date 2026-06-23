#=
periodicity.jl — rank the 10 chemostat experiments by how clean / fittable their
predator–prey cycle is. For a smooth 2-D BNODE to fit an experiment, it needs a
coherent low-frequency oscillation that stands above the day-to-day noise.

For each experiment we linearly interpolate the (irregular, ~daily) algae and
rotifer series onto a regular 1-day grid, mean-centre, and compute the
autocorrelation function. A clean cycle shows a strong positive ACF secondary
peak at the cycle period; pure noise has an ACF that just decays to ~0. We report
that peak's lag (≈ period in days) and height (≈ cycle "cleanliness", 0–1).

Run:  julia --project=../../.. periodicity.jl
=#

import CSV
import DataFrames
import Plots
using Statistics: mean

const MINLAG = 4     # ignore the trivial short-lag autocorrelation
const MAXLAG = 120   # longest period we'd consider a "cycle"
const OUT    = joinpath(@__DIR__, "outputs", "data_explore")
mkpath(OUT)

# Linear interpolation of (t, y) onto integer-day grid spanning the data.
function to_grid(t::Vector{Float64}, y::Vector{Float64})
    g = collect(floor(Int, minimum(t)):ceil(Int, maximum(t)))
    yi = similar(g, Float64)
    for (i, x) in enumerate(g)
        j = searchsortedfirst(t, x)
        if j <= 1
            yi[i] = y[1]
        elseif j > length(t)
            yi[i] = y[end]
        else
            t0, t1 = t[j-1], t[j]; w = (x - t0) / (t1 - t0)
            yi[i] = (1 - w) * y[j-1] + w * y[j]
        end
    end
    return g, yi
end

# Autocorrelation at lags 0..L, and the strongest peak in [MINLAG, MAXLAG].
function acf_peak(y::Vector{Float64})
    yc = y .- mean(y); n = length(yc)
    den = sum(abs2, yc)
    L = min(MAXLAG, n - 2)
    ac = [den == 0 ? 0.0 : sum(yc[1:n-k] .* yc[k+1:n]) / den for k in 0:L]
    lo = min(MINLAG, L)
    peak_lag, peak_val = 0, -Inf
    for k in lo:L
        if ac[k+1] > peak_val; peak_val = ac[k+1]; peak_lag = k; end
    end
    return ac, peak_lag, peak_val
end

df = CSV.read(joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv"),
              DataFrames.DataFrame; missingstring="")
expts = sort(unique(df.experiment))

rows = []
acf_plots = []
for e in expts
    sub = df[df.experiment .== e, :]
    ok = .!ismissing.(sub.algae) .& .!ismissing.(sub.rotifers)
    t  = Float64.(sub.day[ok])
    _, ag = to_grid(t, Float64.(sub.algae[ok]))
    _, rg = to_grid(t, Float64.(sub.rotifers[ok]))
    aca, la, va = acf_peak(ag)
    acr, lr, vr = acf_peak(rg)
    push!(rows, (; e, ndays=round(Int, maximum(t)),
                   algae_period=la, algae_strength=round(va, digits=3),
                   rotif_period=lr, rotif_strength=round(vr, digits=3),
                   score=round((va + vr) / 2, digits=3)))
    p = Plots.plot(0:length(aca)-1, aca, color=:seagreen, lw=1.5, label="algae",
                   title="C$e (score=$(round((va+vr)/2, digits=2)))", titlefontsize=8,
                   legend=false, grid=false, xlabel="lag (days)")
    Plots.plot!(p, 0:length(acr)-1, acr, color=:firebrick, lw=1.5)
    Plots.hline!(p, [0.0], color=:gray, ls=:dot)
    push!(acf_plots, p)
end

DataFrames.sort!(DataFrames.DataFrame(rows), :score, rev=true)
tbl = DataFrames.sort(DataFrames.DataFrame(rows), :score, rev=true)
println("\nExperiments ranked by cycle cleanliness (ACF secondary-peak strength):")
show(stdout, tbl; show_row_number=false, eltypes=false); println()

Plots.savefig(Plots.plot(acf_plots...; layout=(5, 2), size=(1100, 1400),
              plot_title="Autocorrelation per experiment — a clean cycle = strong positive secondary peak",
              plot_titlefontsize=11), joinpath(OUT, "acf_all.png"))
println("→ acf_all.png")
