#=
lv_gradient_score.jl — LV-ness via GRADIENT MATCHING (the well-posed metric).

Trajectory-fitting LV to this multi-cycle data is ill-posed: least squares has a
strong attractor at the flat/fixed-point solution (a mis-phased oscillation scores
worse than flat), so it kept returning NSE ≈ 0 regardless of the data. Gradient
matching sidesteps that entirely. On the (smoothed) data it asks, by closed-form
linear regression, whether the LOCAL RATES OF CHANGE have the Lotka–Volterra form:

    dx/dt = α·x − β·(x·y)        regress dx/dt on [x, xy]   → α, −β,  R²ₓ
    dy/dt = −δ·y + γ·(x·y)       regress dy/dt on [y, xy]   → −δ, γ,  R²ᵧ

LV-ness % = R² (variance of the derivatives explained by the LV form). 100% = the
vector field is exactly LV; 0% = the LV terms explain none of the dynamics. The
regression also yields the best-fit α, β, δ, γ directly (no optimisation).

Derivatives are estimated by central differences, so the data MUST be smoothed
first (SMOOTH_K) — raw finite differences are pure noise. The smoothing window is
the one knob; we sweep a few to show the score is stable.

Run:  EXPT=6 SMOOTH_K=9 julia --project=../../../.. lv_gradient_score.jl
=#

import CSV, DataFrames, Plots
using Statistics: mean
using Printf: @sprintf

const EXPT     = parse(Int, get(ENV, "EXPT", "6"))
const SMOOTH_K = parse(Int, get(ENV, "SMOOTH_K", "9"))
const OUT      = joinpath(@__DIR__, "outputs"); mkpath(OUT)
const DATA     = joinpath(@__DIR__, "..", "..", "algae-chemostat", "data", "blasius_rotifer_algae.csv")

df = CSV.read(DATA, DataFrames.DataFrame; missingstring="")
df = df[df.experiment .== EXPT, :]
keep = .!ismissing.(df.algae) .& .!ismissing.(df.rotifers)
df = df[keep, :]; DataFrames.sort!(df, :day)
day = Float64.(df.day); n0 = length(day)
smooth(y, k) = k <= 1 ? y : [mean(@view y[max(1, i - k ÷ 2):min(length(y), i + k ÷ 2)]) for i in 1:length(y)]

R2(t, p) = 1 - sum(abs2, t .- p) / sum(abs2, t .- mean(t))
pct(x) = max(0.0, x) * 100

function score(K)
    a = smooth(Float64.(df.algae), K); r = smooth(Float64.(df.rotifers), K)
    X = a ./ mean(a); Y = r ./ mean(r)                 # normalised states
    # central-difference derivatives (interior points)
    idx = 2:(n0 - 1)
    dX = [(X[i+1] - X[i-1]) / (day[i+1] - day[i-1]) for i in idx]
    dY = [(Y[i+1] - Y[i-1]) / (day[i+1] - day[i-1]) for i in idx]
    x = X[idx]; y = Y[idx]; xy = x .* y
    # OLS, pure LV form (no intercept)
    cx = hcat(x, xy) \ dX;  α = cx[1]; β = -cx[2]; px = hcat(x, xy) * cx
    cy = hcat(y, xy) \ dY;  δ = -cy[1]; γ = cy[2];  py = hcat(y, xy) * cy
    return (; K, α, β, δ, γ, R2x = R2(dX, px), R2y = R2(dY, py),
              dX, dY, px, py)
end

s = score(SMOOTH_K)
lvness = mean([pct(s.R2x), pct(s.R2y)])

println("\n===== C$EXPT Lotka–Volterra score (gradient matching, $(SMOOTH_K)-pt smooth) =====")
println(@sprintf("best-fit params (per-day, normalised states):"))
println(@sprintf("  α (prey growth)        = %+.4f", s.α))
println(@sprintf("  β (predation on prey)  = %+.4f%s", s.β, s.β < 0 ? "   ⚠ negative (anti-LV)" : ""))
println(@sprintf("  δ (predator death)     = %+.4f%s", s.δ, s.δ < 0 ? "   ⚠ negative (anti-LV)" : ""))
println(@sprintf("  γ (predator growth)    = %+.4f", s.γ))
println(@sprintf("LV-ness (R² of LV vector field):  algae-eq %.0f%%   rotifer-eq %.0f%%   →  %.0f%%",
                 pct(s.R2x), pct(s.R2y), lvness))

# sweep smoothing windows to show stability
println("\nSensitivity to smoothing window:")
for K in [5, 9, 15, 21]
    sk = score(K)
    println(@sprintf("  K=%2d : algae-eq %3.0f%%  rotifer-eq %3.0f%%  →  %3.0f%%   (α=%.2f β=%.2f δ=%.2f γ=%.2f)",
                     K, pct(sk.R2x), pct(sk.R2y), mean([pct(sk.R2x), pct(sk.R2y)]), sk.α, sk.β, sk.δ, sk.γ))
end

# diagnostic scatter: actual vs LV-predicted derivative
try
    p1 = Plots.scatter(s.dX, s.px; ms=2.5, msw=0, alpha=0.5, color=:seagreen, legend=false,
                       xlabel="actual dx/dt", ylabel="LV-predicted", title=@sprintf("algae eq: R²=%.0f%%", pct(s.R2x)))
    Plots.plot!(p1, [minimum(s.dX), maximum(s.dX)], [minimum(s.dX), maximum(s.dX)], color=:black, ls=:dash)
    p2 = Plots.scatter(s.dY, s.py; ms=2.5, msw=0, alpha=0.5, color=:firebrick, legend=false,
                       xlabel="actual dy/dt", ylabel="LV-predicted", title=@sprintf("rotifer eq: R²=%.0f%%", pct(s.R2y)))
    Plots.plot!(p2, [minimum(s.dY), maximum(s.dY)], [minimum(s.dY), maximum(s.dY)], color=:black, ls=:dash)
    Plots.savefig(Plots.plot(p1, p2; layout=(1, 2), size=(1000, 460),
                  plot_title="C$EXPT gradient-matching LV fit ($(SMOOTH_K)-pt smooth)"),
                  joinpath(OUT, "lv_gradmatch_C$(EXPT).png"))
    println("→ outputs/lv_gradmatch_C$(EXPT).png")
catch e; @warn "plot failed" exception=e; end

csv = joinpath(OUT, "lv_gradient_scores.csv")
row = (; experiment=EXPT, smooth_k=SMOOTH_K,
         alpha=round(s.α,digits=4), beta=round(s.β,digits=4), delta=round(s.δ,digits=4), gamma=round(s.γ,digits=4),
         lvness_algae_pct=round(pct(s.R2x),digits=1), lvness_rotifer_pct=round(pct(s.R2y),digits=1),
         lvness_pct=round(lvness,digits=1))
sdf = DataFrames.DataFrame([row])
isfile(csv) && (sdf = vcat(CSV.read(csv, DataFrames.DataFrame), sdf; cols=:union))
CSV.write(csv, sdf)
println("→ outputs/lv_gradient_scores.csv")
