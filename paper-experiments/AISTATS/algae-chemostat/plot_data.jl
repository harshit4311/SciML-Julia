#=
plot_data.jl — exploratory plots of the Blasius et al. (2020) rotifer–algae
chemostat dataset, BEFORE any model fitting. Just looks at the raw data:
algae (prey) vs rotifers (predator) per experiment, plus phase portraits.

Run:  julia --project=../../.. plot_data.jl
Outputs → outputs/data_explore/
=#

import CSV
import DataFrames
import Plots
include("plot_helpers.jl")

const DATA = joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv")
const OUT  = joinpath(@__DIR__, "outputs", "data_explore")
const NWIN = parse(Int, get(ENV, "NWIN", "5"))     # time-window rows in faceted views
mkpath(OUT)

df = CSV.read(DATA, DataFrames.DataFrame; missingstring="")
expts = sort(unique(df.experiment))
println("Loaded $(DataFrames.nrow(df)) rows, experiments: $expts")

# Keep only rows where the variable in question is present (NaN/missing dropped).
present(v) = .!ismissing.(v)

# ---- 1. Per-experiment time series (algae + rotifers on twin axes) -----------
function ts_panel(sub; title="")
    a_ok = present(sub.algae); r_ok = present(sub.rotifers)
    p = Plots.plot(title=title, titlefontsize=8, legend=false, grid=false,
                   xlabel="day", ylabel="algae (10⁶/ml)", ytickfontcolor=:seagreen,
                   yguidefontcolor=:seagreen, left_margin=3Plots.mm)
    Plots.plot!(p, sub.day[a_ok], collect(skipmissing(sub.algae)),
                color=:seagreen, lw=1.4, marker=:circle, ms=1.6, mc=:seagreen)
    # rotifers on a twin y-axis
    pr = Plots.twinx(p)
    Plots.plot!(pr, sub.day[r_ok], collect(skipmissing(sub.rotifers)),
                color=:firebrick, lw=1.4, marker=:circle, ms=1.6, mc=:firebrick,
                ylabel="rotifers (/ml)", legend=false, grid=false,
                ytickfontcolor=:firebrick, yguidefontcolor=:firebrick)
    return p
end

panels = [ts_panel(df[df.experiment .== e, :];
                   title="C$e  (n=$(sum(df.experiment .== e)), " *
                         "$(round(maximum(df.day[df.experiment .== e]), digits=0)) d)")
          for e in expts]
grid = Plots.plot(panels...; layout=(5, 2), size=(1300, 1500),
                  plot_title="Blasius 2020 — algae (green) vs rotifers (red), all 10 chemostats",
                  plot_titlefontsize=11)
Plots.savefig(grid, joinpath(OUT, "timeseries_all.png"))
println("→ timeseries_all.png")

# ---- 2. Phase portraits (algae vs rotifers), one panel per experiment --------
function phase_panel(sub; title="")
    ok = present(sub.algae) .& present(sub.rotifers)
    a = Float64.(sub.algae[ok]); r = Float64.(sub.rotifers[ok])
    p = Plots.plot(a, r, title=title, titlefontsize=8, legend=false, grid=false,
                   xlabel="algae (10⁶/ml)", ylabel="rotifers (/ml)",
                   lw=0.8, color=:gray, alpha=0.6, left_margin=3Plots.mm)
    # colour the points by time to show the cycling direction
    Plots.scatter!(p, a, r, marker_z=sub.day[ok], ms=2.4, c=:viridis,
                   colorbar=false, markerstrokewidth=0)
    return p
end
pp = [phase_panel(df[df.experiment .== e, :]; title="C$e") for e in expts]
pgrid = Plots.plot(pp...; layout=(5, 2), size=(1100, 1500),
                   plot_title="Phase portraits — colour = time (dark→light)",
                   plot_titlefontsize=11)
Plots.savefig(pgrid, joinpath(OUT, "phase_all.png"))
println("→ phase_all.png")

# ---- 3. Faceted zoom on C1 (the long flagship run) ---------------------------
# Split the ~374-day record into NWIN time-window rows × 2 channel columns so the
# individual cycles are legible instead of crammed into one axis.
for e in (1,)
    sub = df[df.experiment .== e, :]
    ok = present(sub.algae) .& present(sub.rotifers)
    day = Float64.(sub.day[ok])
    faceted_grid(joinpath(OUT, "C$(e)_faceted.png"), day,
                 [Float64.(sub.algae[ok]), Float64.(sub.rotifers[ok])];
                 labels=["algae (10⁶ cells/ml)", "rotifers (animals/ml)"],
                 colors=[:seagreen, :firebrick], nwin=NWIN,
                 title="C$e — rotifer–algae cycles, faceted by time window")
    println("→ C$(e)_faceted.png")
end

println("Done. Figures in $OUT")
