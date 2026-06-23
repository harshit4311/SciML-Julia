#=
check_smoothing.jl — is there a slow predator–prey cycle hidden under the heavy
day-to-day noise? Overlays centred moving-average smooths (k points) on the raw
C1 points, faceted by time window. If a clean low-frequency cycle emerges, the
BNODE should be fit to the smoothed (or coarsely binned) series, not the raw one.

Run:  K=7 julia --project=../../.. check_smoothing.jl
=#

import CSV
import DataFrames
import Plots
include("plot_helpers.jl")

const EXPT = parse(Int, get(ENV, "EXPT", "1"))
const K    = parse(Int, get(ENV, "K",    "9"))     # moving-average window (points)
const NWIN = parse(Int, get(ENV, "NWIN", "5"))
const OUT  = joinpath(@__DIR__, "outputs", "data_explore")
mkpath(OUT)

# Centred moving average over K points (shrinks window at the edges).
function smooth(y::Vector{Float64}, k::Int)
    n = length(y); h = k ÷ 2
    [mean(@view y[max(1, i-h):min(n, i+h)]) for i in 1:n]
end
using Statistics: mean

df = CSV.read(joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv"),
              DataFrames.DataFrame; missingstring="")
df = df[df.experiment .== EXPT, :]
ok = .!ismissing.(df.algae) .& .!ismissing.(df.rotifers)
df = df[ok, :]; DataFrames.sort!(df, :day)

day = Float64.(df.day)
a   = Float64.(df.algae)
r   = Float64.(df.rotifers)
as  = smooth(a, K)
rs  = smooth(r, K)

# Draw the smoothed line as a zero-width "band" (lo=hi=mean) over the raw points.
faceted_grid(joinpath(OUT, "C$(EXPT)_smoothed.png"), day, [a, r];
             labels=["algae (raw + $(K)-pt smooth)", "rotifers (raw + $(K)-pt smooth)"],
             colors=[:seagreen, :firebrick], nwin=NWIN,
             bands=[(as, as, as), (rs, rs, rs)],
             title="C$EXPT — does a slow cycle survive a $(K)-point moving average?")
println("→ C$(EXPT)_smoothed.png  (K=$K)")
