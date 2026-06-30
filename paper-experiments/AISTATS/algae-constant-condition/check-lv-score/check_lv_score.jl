#=
check_lv_score.jl — how "Lotka–Volterra" is an experiment?

Fits the CLASSICAL 4-parameter Lotka–Volterra model
    dx/dt =  α x − β x y        (prey  = algae)
    dy/dt = −δ y + γ x y        (predator = rotifer)
to a chemostat experiment's full record and reports:
  • the best-fit (α, β, δ, γ),
  • an "LV-ness %" = Nash–Sutcliffe efficiency (100% = LV explains all the
    variance; 0% = no better than predicting the mean; clamped at 0).

Two scores are reported, mirroring how we fit the BNODE:
  • LOCAL (multiple-shooting, ~2-cycle horizon): does the data's *flow* look LV?
  • GLOBAL (single free-running orbit from the first point): does one LV orbit
    reproduce the whole time series? (Strict — penalises phase drift over cycles.)

These are BEST-FIT params for the closest LV model — the data is real biology,
not LV-generated, so the LV-ness % is exactly the "how close is it" measure.

Run:  EXPT=6 julia --project=../../../.. check_lv_score.jl
=#

include("../../map-tests/lv_bnode_common.jl")     # lotka_volterra!, DE/Optimization/Zygote imports
include("../../algae-chemostat/plot_helpers.jl")  # faceted_grid
using Statistics: mean
using Printf: @sprintf
import Random

const EXPT       = parse(Int, get(ENV, "EXPT", "6"))
const SEG_LEN    = parse(Int, get(ENV, "SEG_LEN", "36"))
const SEG_STRIDE = parse(Int, get(ENV, "SEG_STRIDE", "18"))
const N_SCREEN   = parse(Int, get(ENV, "N_SCREEN", "3000"))   # random LV params to screen (forward-only)
const TOPK       = parse(Int, get(ENV, "TOPK", "8"))          # best screened seeds to gradient-refine
const ITERS      = parse(Int, get(ENV, "ITERS", "1500"))
const NWIN       = parse(Int, get(ENV, "NWIN", "7"))
const SMOOTH_K   = parse(Int, get(ENV, "SMOOTH_K", "0"))   # centred moving-average window; 0 = raw data
const OUT        = joinpath(@__DIR__, "outputs"); mkpath(OUT)
const DATA       = joinpath(@__DIR__, "..", "..", "algae-chemostat", "data", "blasius_rotifer_algae.csv")

# --- load, optionally denoise, normalise the full record ---------------------
# Raw C6 is noise-dominated (pointwise NSE ≈ 0 for ANY smooth model — LV or BNODE).
# SMOOTH_K>1 measures LV-ness of the *cycle envelope* (the signal), separating the
# "is the dynamics LV-shaped?" question from the "how noisy is it?" question.
df = CSV.read(DATA, DataFrames.DataFrame; missingstring="")
df = df[df.experiment .== EXPT, :]
keep = .!ismissing.(df.algae) .& .!ismissing.(df.rotifers)
df = df[keep, :]; DataFrames.sort!(df, :day)
day = Float64.(df.day); n = length(day)
smooth(y, k) = k <= 1 ? y : [mean(@view y[max(1, i - k ÷ 2):min(length(y), i + k ÷ 2)]) for i in 1:length(y)]
a_in = smooth(Float64.(df.algae), SMOOTH_K); r_in = smooth(Float64.(df.rotifers), SMOOTH_K)
algae_scale = mean(a_in); rotif_scale = mean(r_in)
data = permutedims(hcat(a_in ./ algae_scale, r_in ./ rotif_scale))  # 2×n
t = day                                                # time in days → α..γ are per-day rates
println(@sprintf("C%d: %d usable days (0–%.0f). %s. Normalised (algae=%.3f, rotifer=%.3f).",
                 EXPT, n, day[end],
                 SMOOTH_K > 1 ? "Smoothed with $(SMOOTH_K)-pt moving average (cycle envelope)" : "Raw data",
                 algae_scale, rotif_scale))

# --- multiple-shooting segments (same protocol as the BNODE) -----------------
starts = collect(1:SEG_STRIDE:(n - 1))
segs = [(s, min(s + SEG_LEN - 1, n)) for s in starts]
last(segs)[2] < n && push!(segs, (max(1, n - SEG_LEN + 1), n))

solve_seg(p, s, e) = Array(DE.solve(
    DE.ODEProblem(lotka_volterra!, data[:, s], (t[s], t[e]), p),
    DE.Tsit5(); saveat=t[s:e], abstol=1e-6, reltol=1e-6, maxiters=Int(1e5)))

function ms_loss(logp)                                  # params kept positive via exp
    p = exp.(logp); L = 0.0
    for (s, e) in segs
        pred = solve_seg(p, s, e)
        all(isfinite, pred) || return Inf               # reject blow-ups
        L += sum(abs2, data[:, s:e] .- pred)
    end
    return L
end

# --- fit LV: random-search screen (explores PERIOD, which gradients fit badly)
# then gradient-refine the best seeds ----------------------------------------
println("Fitting LV ($(length(segs)) segments). Screening $N_SCREEN random params…")
Random.seed!(1)
screened = Vector{Tuple{Float64,Vector{Float64}}}()
for _ in 1:N_SCREEN
    logp = log.(0.05 .+ 3.0 .* rand(4))                 # each param ∈ [0.05, 3.05]
    o = try ms_loss(logp) catch; Inf end
    isfinite(o) && push!(screened, (o, logp))
end
sort!(screened, by = x -> x[1])
println(@sprintf("  best screened obj=%.4g (%d finite of %d)", screened[1][1], length(screened), N_SCREEN))

best = (obj = Inf, p = ones(4))
for j in 1:min(TOPK, length(screened))
    res = try
        Optimization.solve(
            Optimization.OptimizationProblem(
                Optimization.OptimizationFunction((x, _) -> ms_loss(x), Optimization.AutoZygote()), screened[j][2]),
            OptimizationOptimisers.Adam(0.02); maxiters=ITERS)
    catch e
        @warn "refine $j failed" exception=e; continue
    end
    println(@sprintf("  refine %2d  screen=%.4g → obj=%.4g", j, screened[j][1], res.objective))
    res.objective < best.obj && (global best = (obj = res.objective, p = exp.(res.u)))
end
α, β, δ, γ = best.p

# --- scores: NSE (= 1 − SS_res/SS_tot), per channel --------------------------
nse(d, p) = 1 - sum(abs2, d .- p) / sum(abs2, d .- mean(d))
pct(x) = max(0.0, x) * 100

# local (segmented) predictions
LD1 = Float64[]; LP1 = Float64[]; LD2 = Float64[]; LP2 = Float64[]
for (s, e) in segs
    pr = solve_seg(best.p, s, e)
    append!(LP1, pr[1, :]); append!(LD1, data[1, s:e])
    append!(LP2, pr[2, :]); append!(LD2, data[2, s:e])
end
nse_local = (nse(LD1, LP1), nse(LD2, LP2))

# global single free-running orbit from the first point
gpred = try
    Array(DE.solve(DE.ODEProblem(lotka_volterra!, data[:, 1], (t[1], t[end]), best.p),
                   DE.Tsit5(); saveat=t, abstol=1e-6, reltol=1e-6, maxiters=Int(1e5)))
catch; fill(NaN, 2, n) end
nse_global = size(gpred, 2) == n ?
    (nse(data[1, :], gpred[1, :]), nse(data[2, :], gpred[2, :])) : (NaN, NaN)

local_avg = mean(pct.(collect(nse_local)))
global_avg = mean(pct.(collect(nse_global)))

# --- plot the global LV orbit vs data (faceted) ------------------------------
try
    faceted_grid(joinpath(OUT, "lv_fit_C$(EXPT)$(SMOOTH_K > 1 ? "_s$(SMOOTH_K)" : "").png"), day,
                 [data[1, :], data[2, :]];
                 labels=["algae (norm) — LV α=$(round(α,digits=3))",
                         "rotifer (norm) — LV orbit"],
                 colors=[:seagreen, :firebrick], nwin=NWIN,
                 bands=[(gpred[1, :], gpred[1, :], gpred[1, :]),
                        (gpred[2, :], gpred[2, :], gpred[2, :])],
                 title="C$EXPT — best-fit classical LV orbit (line) vs data (points)")
    println("→ outputs/lv_fit_C$(EXPT).png")
catch e; @warn "plot failed" exception=e; end

# --- report + append to a cross-experiment table -----------------------------
println("\n===== C$EXPT Lotka–Volterra score =====")
println(@sprintf("best-fit params (per-day, normalised states):"))
println(@sprintf("  α (prey growth)        = %.4f", α))
println(@sprintf("  β (predation on prey)  = %.4f", β))
println(@sprintf("  δ (predator death)     = %.4f", δ))
println(@sprintf("  γ (predator growth)    = %.4f", γ))
println(@sprintf("LV-ness  LOCAL  (2-cycle MS):  algae %.0f%%  rotifer %.0f%%  →  %.0f%%",
                 pct(nse_local[1]), pct(nse_local[2]), local_avg))
println(@sprintf("LV-ness  GLOBAL (1 orbit)   :  algae %.0f%%  rotifer %.0f%%  →  %.0f%%",
                 pct(nse_global[1]), pct(nse_global[2]), global_avg))

row = (; experiment=EXPT, smooth_k=SMOOTH_K, alpha=round(α,digits=4), beta=round(β,digits=4),
         delta=round(δ,digits=4), gamma=round(γ,digits=4),
         lvness_local_pct=round(local_avg,digits=1), lvness_global_pct=round(global_avg,digits=1),
         algae_scale=round(algae_scale,digits=3), rotif_scale=round(rotif_scale,digits=3))
csv = joinpath(OUT, "lv_scores.csv")
sdf = DataFrames.DataFrame([row])
isfile(csv) && (sdf = vcat(CSV.read(csv, DataFrames.DataFrame), sdf; cols=:union))
CSV.write(csv, sdf)
println("→ outputs/lv_scores.csv")
