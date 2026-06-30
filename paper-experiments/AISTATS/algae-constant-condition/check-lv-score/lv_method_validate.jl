#=
lv_method_validate.jl — calibrate the gradient-matching LV score.

C6 scored ~5% LV-ness. Is that REAL non-LV-ness, or does C6's heavy observation
noise defeat gradient matching even for a truly-LV system? This runs the IDENTICAL
score on SYNTHETIC LV data (a known 100%-LV signal) with increasing noise, so we
can read off what score a "true LV at C6's noise level" would earn.

If LV+C6-noise still scores high (say >50%) but C6 scores 5%, then C6 is genuinely
not LV. If LV+C6-noise also collapses to ~5%, the method can't see through the
noise and the C6 result is inconclusive.

Run:  julia --project=../../../.. lv_method_validate.jl
=#

include("../../map-tests/lv_bnode_common.jl")     # lotka_volterra!, DE
using Statistics: mean
using Printf: @sprintf
import Random

smooth(y, k) = k <= 1 ? y : [mean(@view y[max(1, i - k ÷ 2):min(length(y), i + k ÷ 2)]) for i in 1:length(y)]
R2(t, p) = 1 - sum(abs2, t .- p) / sum(abs2, t .- mean(t))
pct(x) = max(0.0, x) * 100

# gradient-matching score on (day, algae, rotifer) — identical to lv_gradient_score.jl
function gradmatch(day, a, r; K=9)
    X = smooth(a, K); X ./= mean(X); Y = smooth(r, K); Y ./= mean(Y)
    nn = length(day); idx = 2:(nn - 1)
    dX = [(X[i+1] - X[i-1]) / (day[i+1] - day[i-1]) for i in idx]
    dY = [(Y[i+1] - Y[i-1]) / (day[i+1] - day[i-1]) for i in idx]
    x = X[idx]; y = Y[idx]; xy = x .* y
    cx = hcat(x, xy) \ dX; cy = hcat(y, xy) \ dY
    (; R2x = R2(dX, hcat(x, xy) * cx), R2y = R2(dY, hcat(y, xy) * cy),
       α = cx[1], β = -cx[2], δ = -cy[1], γ = cy[2])
end

# --- synthetic LV: ~16-day period over 230 daily samples (like C6) ------------
p_true = [0.5, 0.5, 0.31, 0.5]                 # α,β,δ,γ → period ≈ 2π/√(αδ) ≈ 16 d
day = collect(0.0:1.0:230.0); N = length(day)
traj = Array(DE.solve(DE.ODEProblem(lotka_volterra!, [2.0, 0.5], (0.0, 230.0), p_true),
                      DE.Tsit5(); saveat=day, abstol=1e-9, reltol=1e-9))
A0 = traj[1, :] ./ mean(traj[1, :]); R0 = traj[2, :] ./ mean(traj[2, :])   # normalised clean signal
println(@sprintf("Synthetic LV: %d daily samples, amplitude algae≈%.1f rotifer≈%.1f (norm).",
                 N, maximum(A0) - minimum(A0), maximum(R0) - minimum(R0)))

println("\n  noise σ (norm) |  algae-eq  rotifer-eq  →  LV-ness  | recovered α,β,δ,γ")
for σ in [0.0, 0.1, 0.2, 0.4, 0.6]
    Random.seed!(7)
    a = A0 .+ σ .* randn(N); r = R0 .+ σ .* randn(N)
    g = gradmatch(day, a, r; K=9)
    println(@sprintf("  %8.2f       |  %5.0f%%    %5.0f%%    →  %5.0f%%   | %+.2f %+.2f %+.2f %+.2f",
                     σ, pct(g.R2x), pct(g.R2y), mean([pct(g.R2x), pct(g.R2y)]), g.α, g.β, g.δ, g.γ))
end
println("\n(C6 scored ~5% at σ̂≈0.4. Compare to the σ=0.4 row above: if that row is")
println(" much higher, C6 is genuinely non-LV; if it's also ~5%, the noise masks LV.)")
