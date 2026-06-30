#=
lv_horizon.jl — discriminating LV-ness via fit-vs-prediction-horizon.

The whole-trajectory NSE pins to ~0% for ANY smooth orbit on this noisy,
phase-wandering data (LV and the BNODE both). To get a metric that actually
ranks experiments, we measure how well each model predicts h DAYS AHEAD from
every data point:
  • short horizon (sub-cycle) → local-flow match → "is the data locally LV?"
  • long horizon (many cycles) → phase drift dominates → → 0 for everyone.

We plot the classical LV (4 params, from check_lv_score.jl) against the fitted
BNODE (354 params, weights from the grow_horizon run) on the same axis. The
short-horizon LV % is the discriminating LV-ness; LV/BNODE is the fraction of
the *learnable* signal the mechanistic model already captures.

Prereqs: run check_lv_score.jl (writes outputs/lv_scores.csv) and the BNODE
grow_horizon for the same EXPT (writes ../../algae-chemostat/outputs/
algae_chemostat/map_params_C<EXPT>.csv).

Run:  EXPT=6 julia --project=../../../.. lv_horizon.jl
=#

include("../../map-tests/lv_bnode_common.jl")     # Lux, DE, ComponentArrays, lotka_volterra!
using Statistics: mean
using Printf: @sprintf
import Plots, Random

const EXPT     = parse(Int, get(ENV, "EXPT", "6"))
const TMAX_B   = parse(Float64, get(ENV, "TMAX_B", "30.53"))   # BNODE time-rescale used in its fit (≈ maxday/7.5)
const HIDDEN   = parse(Int, get(ENV, "HIDDEN", "16"))
const N_HIDDEN = parse(Int, get(ENV, "N_HIDDEN", "2"))
const HZ       = [1, 2, 4, 8, 16, 32]                          # prediction horizons (points ≈ days)
const OUT      = joinpath(@__DIR__, "outputs"); mkpath(OUT)
const DATA     = joinpath(@__DIR__, "..", "..", "algae-chemostat", "data", "blasius_rotifer_algae.csv")
const BPAR     = joinpath(@__DIR__, "..", "..", "algae-chemostat", "outputs", "algae_chemostat", "map_params_C$(EXPT).csv")
const LVCSV    = joinpath(OUT, "lv_scores.csv")

# --- data (drop missing) -----------------------------------------------------
df = CSV.read(DATA, DataFrames.DataFrame; missingstring="")
df = df[df.experiment .== EXPT, :]
keep = .!ismissing.(df.algae) .& .!ismissing.(df.rotifers)
df = df[keep, :]; DataFrames.sort!(df, :day)
day = Float64.(df.day); n = length(day)
ra = Float64.(df.algae); rr = Float64.(df.rotifers)

# --- LV: full-record-mean normalisation, time in days ------------------------
lv = CSV.read(LVCSV, DataFrames.DataFrame)
row = filter(r -> r.experiment == EXPT, lv)[end, :]
plv = Float64[row.alpha, row.beta, row.delta, row.gamma]
dataL = permutedims(hcat(ra ./ mean(ra), rr ./ mean(rr)))
tL = day
println(@sprintf("C%d LV params α=%.3f β=%.3f δ=%.3f γ=%.3f", EXPT, plv...))

# --- BNODE: train-window-mean normalisation, rescaled time [0,TMAX_B] --------
ntr = round(Int, 0.7 * n)
dataB = permutedims(hcat(ra ./ mean(ra[1:ntr]), rr ./ mean(rr[1:ntr])))
tB = (day .- day[1]) ./ (day[end] - day[1]) .* TMAX_B
layers = Any[Lux.Dense(2, HIDDEN, Lux.tanh)]
for _ in 2:N_HIDDEN; push!(layers, Lux.Dense(HIDDEN, HIDDEN, Lux.tanh)); end
push!(layers, Lux.Dense(HIDDEN, 2))
dudt2 = Lux.Chain(layers...)
Random.seed!(42); ps, st = Lux.setup(Random.default_rng(), dudt2)
p_struct = ComponentArrays.ComponentArray{Float64}(ps)
wb = Vector{Float64}(CSV.read(BPAR, DataFrames.DataFrame).w)
length(wb) == length(vec(collect(p_struct))) ||
    error("BNODE weights $(length(wb)) ≠ net $(length(vec(collect(p_struct)))); set HIDDEN/N_HIDDEN")
pb = ComponentArrays.ComponentVector(wb, ComponentArrays.getaxes(p_struct))
nfunc(u, p, t) = dudt2(u, p, st)[1]
println("BNODE weights loaded ($(length(wb))).")

# --- h-step-ahead endpoint NSE ----------------------------------------------
function endpt(f, u0, t0, t1, p)
    try
        v = Array(DE.solve(DE.ODEProblem(f, u0, (t0, t1), p), DE.Tsit5();
                           saveat=[t1], abstol=1e-6, reltol=1e-6, maxiters=Int(1e5)))[:, end]
        all(isfinite, v) ? v : nothing
    catch; nothing end
end
nse(d, p) = 1 - sum(abs2, d .- p) / sum(abs2, d .- mean(d))
pct(x) = max(0.0, x) * 100

function horizon_nse(f, dm, tm, p)
    map(HZ) do h
        D1 = Float64[]; P1 = Float64[]; D2 = Float64[]; P2 = Float64[]
        for i in 1:(n - h)
            pe = endpt(f, dm[:, i], tm[i], tm[i + h], p)
            pe === nothing && continue
            push!(P1, pe[1]); push!(D1, dm[1, i + h]); push!(P2, pe[2]); push!(D2, dm[2, i + h])
        end
        (nse(D1, P1), nse(D2, P2))
    end
end

lvr = horizon_nse(lotka_volterra!, dataL, tL, plv)
bnr = horizon_nse(nfunc, dataB, tB, pb)

# --- report + plot -----------------------------------------------------------
lv_avg = [mean(pct.(collect(x))) for x in lvr]
bn_avg = [mean(pct.(collect(x))) for x in bnr]
println("\n===== C$EXPT  fit vs prediction horizon (NSE %, variance explained) =====")
println("  horizon(d) |  LV alg  LV rot  LV avg |  BN alg  BN rot  BN avg |  LV/BN")
for (k, h) in enumerate(HZ)
    la, lo = pct(lvr[k][1]), pct(lvr[k][2]); ba, bo = pct(bnr[k][1]), pct(bnr[k][2])
    ratio = bn_avg[k] > 1 ? lv_avg[k] / bn_avg[k] : NaN
    println(@sprintf("  %7d    | %6.0f %6.0f %6.0f | %6.0f %6.0f %6.0f | %6.2f",
                     h, la, lo, lv_avg[k], ba, bo, bn_avg[k], ratio))
end
h4 = findfirst(==(4), HZ)
println(@sprintf("\nLV-ness @ 4-day horizon (local flow): %.0f%%   (BNODE %.0f%%  →  LV captures %.0f%% of learnable)",
                 lv_avg[h4], bn_avg[h4], bn_avg[h4] > 1 ? 100 * lv_avg[h4] / bn_avg[h4] : NaN))

try
    p = Plots.plot(HZ, lv_avg; marker=:circle, lw=2, color=:purple, label="classical LV (4 params)",
                   xlabel="prediction horizon (days)", ylabel="fit: % variance explained (NSE)",
                   title="C$EXPT — local LV-ness vs the BNODE", xscale=:log2, ylim=(0, 100), legend=:topright)
    Plots.plot!(p, HZ, bn_avg; marker=:square, lw=2, color=:black, label="BNODE (354 params)")
    Plots.savefig(p, joinpath(OUT, "lv_horizon_C$(EXPT).png"))
    println("→ outputs/lv_horizon_C$(EXPT).png")
catch e; @warn "plot failed" exception=e; end

CSV.write(joinpath(OUT, "lv_horizon_C$(EXPT).csv"), DataFrames.DataFrame(
    horizon_days = HZ,
    lv_algae = [pct(x[1]) for x in lvr], lv_rotifer = [pct(x[2]) for x in lvr], lv_avg = lv_avg,
    bnode_algae = [pct(x[1]) for x in bnr], bnode_rotifer = [pct(x[2]) for x in bnr], bnode_avg = bn_avg))
println("→ outputs/lv_horizon_C$(EXPT).csv")
