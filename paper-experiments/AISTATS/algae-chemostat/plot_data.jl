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

const DATA = joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv")
const OUT  = joinpath(@__DIR__, "outputs", "data_explore")
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

# ---- 3. Zoom on C1 (the long flagship run) -----------------------------------
c1 = df[df.experiment .== 1, :]
a_ok = present(c1.algae); r_ok = present(c1.rotifers)
pc1 = Plots.plot(size=(1300, 460), legend=:topright, grid=false,
                 xlabel="day", ylabel="algae (10⁶ cells/ml)",
                 title="C1 — long-run rotifer–algae cycles (~374 days)",
                 yguidefontcolor=:seagreen, ytickfontcolor=:seagreen)
Plots.plot!(pc1, c1.day[a_ok], collect(skipmissing(c1.algae)),
            color=:seagreen, lw=1.5, label="algae (prey)")
pr1 = Plots.twinx(pc1)
Plots.plot!(pr1, c1.day[r_ok], collect(skipmissing(c1.rotifers)),
            color=:firebrick, lw=1.5, label="rotifers (predator)",
            ylabel="rotifers (animals/ml)", legend=:topleft, grid=false,
            yguidefontcolor=:firebrick, ytickfontcolor=:firebrick)
Plots.savefig(pc1, joinpath(OUT, "C1_timeseries.png"))
println("→ C1_timeseries.png")

println("Done. Figures in $OUT")
