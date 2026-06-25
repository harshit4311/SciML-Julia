#=
residual_diag.jl — is the model's residual variance stationary?

Reads the MAP point-estimate dump written by ../algae-chemostat/algae_chemostat.jl
(outputs/algae_chemostat/map_prediction.csv) and analyses the residuals
r = actual − point_estimate for each channel:

  • residual vs time, with a centred rolling-std band and the train/forecast divider
  • train-window vs forecast-window residual std (ratio ≫ 1 ⇒ non-stationary)
  • a linear trend of the rolling std (slope > 0 ⇒ variance growing over time)

A well-specified, stationary-noise fit ⇒ flat residual std (ratio ≈ 1, slope ≈ 0).
A forced/misspecified fit (e.g. autonomous BNODE on forced C9) ⇒ residual variance
that grows into the forecast window and clusters at the Nᵢₙ switches.

Run:  julia --project=../../.. residual_diag.jl <EXPT>
=#

import CSV
import DataFrames
import Plots
using Statistics: mean, std
using Printf: @sprintf

const EXPT = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const W    = parse(Int, get(ENV, "ROLLWIN", "15"))     # rolling-std window (points)
const PRED = joinpath(@__DIR__, "..", "algae-chemostat", "outputs", "algae_chemostat", "map_prediction.csv")
const OUT  = joinpath(@__DIR__, "outputs"); mkpath(OUT)

isfile(PRED) || error("No prediction file at $PRED — run a MAP fit first:\n" *
                      "  MAP_ONLY=1 EXPT=$EXPT julia --project=../../.. ../algae-chemostat/algae_chemostat.jl")
df = CSV.read(PRED, DataFrames.DataFrame)
got = first(df.experiment)
got == EXPT || @warn "map_prediction.csv is for C$got, not C$EXPT — re-run the MAP fit with EXPT=$EXPT"

day    = Float64.(df.day)
n      = length(day)
ntrain = sum(df.split .== "train")
split_day = day[ntrain]

# Centred rolling std over W points.
rollstd(r) = [std(@view r[max(1, i - W ÷ 2):min(n, i + W ÷ 2)]) for i in 1:n]
# Slope of rolling std vs day (least squares) — a simple non-stationarity trend.
function slope(x, y)
    x̄ = mean(x); ȳ = mean(y)
    sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

channels = [("algae", df.algae_actual, df.algae_pred, :seagreen),
            ("rotifer", df.rotifer_actual, df.rotifer_pred, :firebrick)]

panels = []; summary = []
for (name, act, pred, col) in channels
    r  = Float64.(act) .- Float64.(pred)
    rs = rollstd(r)
    tr = view(r, 1:ntrain); fo = view(r, ntrain+1:n)
    s_tr, s_fo = std(tr), std(fo)
    ratio = s_fo / s_tr
    sl = slope(day, rs)
    push!(summary, (; experiment=EXPT, channel=name,
                      train_std=round(s_tr, digits=4), forecast_std=round(s_fo, digits=4),
                      ratio=round(ratio, digits=3), rollstd_slope=round(sl, digits=5)))

    p = Plots.plot(day, fill(0.0, n); color=:black, lw=1, ls=:dot, legend=:topleft, grid=false,
                   label="", title="C$EXPT $name residuals (actual − MAP)", titlefontsize=9,
                   xlabel="day", ylabel="residual")
    Plots.plot!(p, day, rs; ribbon=(rs, rs), fillalpha=0.18, color=col, lw=1.6,
                label="±rolling std (W=$W)")
    Plots.scatter!(p, day, r; ms=2.4, mc=col, msw=0, alpha=0.7, label="residual")
    Plots.vline!(p, [split_day]; color=:gray, ls=:dash, label="train/forecast")
    Plots.annotate!(p, day[end], maximum(rs),
        Plots.text(@sprintf("σ_train=%.3f  σ_fore=%.3f  (×%.2f)", s_tr, s_fo, ratio), 7, :right, :top))
    push!(panels, p)
end

Plots.savefig(Plots.plot(panels...; layout=(2, 1), size=(1100, 760),
              plot_title="C$EXPT residual-variance stationarity", plot_titlefontsize=11),
              joinpath(OUT, "residuals_C$(EXPT).png"))

# Append to the cross-experiment summary table.
sumcsv = joinpath(OUT, "residual_summary.csv")
sdf = DataFrames.DataFrame(summary)
isfile(sumcsv) && (sdf = vcat(CSV.read(sumcsv, DataFrames.DataFrame), sdf; cols=:union))
CSV.write(sumcsv, sdf)

println("→ outputs/residuals_C$(EXPT).png")
for s in summary
    println(@sprintf("  %-8s σ_train=%.4f  σ_fore=%.4f  ratio=%.2f  rollstd_slope=%+.5f%s",
                     s.channel, s.train_std, s.forecast_std, s.ratio, s.rollstd_slope,
                     s.ratio > 1.3 ? "   ← non-stationary (forecast variance↑)" : ""))
end
