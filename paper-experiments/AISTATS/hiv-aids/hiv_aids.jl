#=
ACTG 315 HIV / ART case study — POOLED Bayesian Neural ODE (BNODE+MAP+HMC).

Companion to the Hudson Bay lynx-hare case study (../hudson-bay/). Same harness
lineage — the 2-32-32-32-2 tanh neural-ODE, the two-phase MAP scheduler, the NUTS
driver, and the diagnostics — all reused verbatim from map-tests/lv_bnode_common.jl.
The new machinery here is the POOLED likelihood.

Why pooled? Each of the 46 ACTG 315 patients has only ~8 (range 4–10) blood draws
over 28 weeks — far too sparse to fit a neural ODE per patient (the posterior would
be entirely prior-dominated, exactly the failure mode of fitting a BNODE to 8
lynx-hare points). Instead we fit ONE shared neural ODE f_θ to all 46 patients at
once. The population shares the same HIV-ART dynamics (same virus, same regimen,
same biology); what differs is each patient's initial condition:

    Shared:    dz/dt = f_θ(z),     z = [CD4, log10 viral load]      ← one network
    Patient i: z_i(0) = [CD4₀ⁱ, VL₀ⁱ]  → integrate → trajectory i

All 46 noisy trajectories (~360 observations) inform a single posterior p(θ | data).

Clinical question (treatment monitoring, not diagnosis):
  "Given a patient's early ART response, will their viral load suppress and CD4
   recover — or are they heading toward treatment failure / AIDS progression?"
The decision-relevant outputs are the posterior 5th-percentile CD4 trajectory
(precautionary AIDS-risk lower bound; AIDS threshold CD4 = 200 cells/µL) and the
95th-percentile viral-load trajectory (precautionary treatment-failure upper bound;
clinical suppression target ≈ 200 copies/mL, i.e. log10 ≈ 2.3).

Two train/test protocols (env var SPLIT):
  B (default, clinically realistic): hold out LATE time points per patient.
     Train on weeks 0–8 (day ≤ SPLIT_DAY) for all patients; forecast weeks 12–28.
     Maps directly onto "given 8 weeks of response, predict week-24 outcome."
  A (strongest generalisation test): hold out PATIENTS entirely.
     Train on patients 1..N_TRAIN_PAT; forecast the unseen patients from their
     baseline. Tests whether the posterior predictive is calibrated for a brand-new
     patient.

Limitation (state in the paper): this is a pooled model — shared population dynamics
with patient-specific initial conditions only. It does not model inter-patient
variation in *response rate* (genetics, adherence, immune function). The principled
extension is a hierarchical BNODE (population network + patient random effects);
that is future work and overkill for a 4-page workshop paper.

Run (DEV, minutes):
  julia --project=../../.. hiv_aids.jl

Run (PAPER, long — pooled solve is ~Npat× the cost of a single-trajectory run):
  NSAMP=250 NADPT=250 MAXDEPTH=8 DEV_TOL=1e-7 \
  MAP_PHASEA=4000 MAP_PHASEB=600 \
    julia --project=../../.. hiv_aids.jl

Use `caffeinate -i bash -c '…'` on macOS to prevent sleep during long runs.
=#

include("../map-tests/lv_bnode_common.jl")

# === Config — overrides via env vars =========================================
const NSAMP      = parse(Int,     get(ENV, "NSAMP",      "50"))
const NADPT      = parse(Int,     get(ENV, "NADPT",      "50"))
const MAXDEPTH   = parse(Int,     get(ENV, "MAXDEPTH",   "6"))
const DEV_TOL    = parse(Float64, get(ENV, "DEV_TOL",    "1e-6"))
const MAP_PHASEA = parse(Int,     get(ENV, "MAP_PHASEA", "2500"))
const MAP_PHASEB = parse(Int,     get(ENV, "MAP_PHASEB", "400"))
const INIT_SEED  = parse(Int,     get(ENV, "INIT_SEED",  "42"))
const NOISE_SEED = parse(Int,     get(ENV, "NOISE_SEED", "1"))   # PP noise draws
const SPLIT      = uppercase(get(ENV, "SPLIT", "B"))             # "B" (time) or "A" (patient)
const SPLIT_DAY  = parse(Float64, get(ENV, "SPLIT_DAY",  "56.0")) # week 8, option B
const N_TRAIN_PAT= parse(Int,     get(ENV, "N_TRAIN_PAT","36"))   # option A
const DAY_SCALE  = parse(Float64, get(ENV, "DAY_SCALE",  "28.0")) # days → [0, ~7] ODE time

# RNA channel weight in the Gaussian likelihood. Both channels are mean-normalised
# to O(1), so equal weighting (1.0) is the honest default; exposed for ablation.
const RNA_W = parse(Float64, get(ENV, "RNA_W", "1.0"))

# Network architecture. Default 2-16-16-16-2 (~626 params) — deliberately smaller
# than the synthetic-LV harness's 2-32-32-32-2 (2274 params), because 46 sparse
# patients (~275 training obs) over-constrain that bigger net. Both knobs are
# env-exposed so the capacity-vs-calibration ablation is a one-liner:
#   HIDDEN=16 N_HIDDEN=3 → 2-16-16-16-2   (≈626)   [default]
#   HIDDEN=16 N_HIDDEN=2 → 2-16-16-2      (≈354)
#   HIDDEN=8  N_HIDDEN=2 → 2-8-8-2        (≈122)
#   HIDDEN=32 N_HIDDEN=3 → 2-32-32-32-2   (2274)   [old default]
const HIDDEN   = parse(Int, get(ENV, "HIDDEN",   "16"))   # hidden-layer width
const N_HIDDEN = parse(Int, get(ENV, "N_HIDDEN", "3"))    # number of hidden layers
const ARCH_STR = "2-" * join(fill(string(HIDDEN), N_HIDDEN), "-") * "-2"

# Clinical thresholds (real units) used only for the decision-relevance figure.
const CD4_AIDS_THRESH = 200.0     # cells/µL — AIDS-defining boundary
const VL_FAIL_LOG10   = log10(200.0)  # copies/mL — clinical suppression target

# OUTDIR / RESULTS_CSV let validation runs write elsewhere so a quick smoke pass
# never clobbers a real run's figures or results row.
outdir = get(ENV, "OUTDIR", joinpath(@__DIR__, "outputs", "hiv_aids"))
mkpath(outdir)
csv_out = get(ENV, "RESULTS_CSV", joinpath(@__DIR__, "hiv_aids_results.csv"))

# === Load the ACTG 315 dataset ===============================================
data_csv = joinpath(@__DIR__, "data", "actg315.csv")
isfile(data_csv) || error("Missing $data_csv — run `julia --project=../../.. fetch_data.jl` first.")
println("Loading ACTG 315 (pooled HIV-ART dataset)…")
df = CSV.read(data_csv, DataFrames.DataFrame)
DataFrames.sort!(df, [:id, :day])
patient_ids = sort(unique(df.id))
n_pat = length(patient_ids)

# Channel order: row 1 = CD4 (cells/µL), row 2 = log10 viral load (log10 copies/mL).
# Normalise each channel by its training-window mean so the tanh network (sized for
# synthetic LV) stays in range. Time: days → t = day / DAY_SCALE ∈ [0, ~7].
train_mask_rows = SPLIT == "B" ? (df.day .<= SPLIT_DAY) :
                                 (in.(df.id, Ref(Set(patient_ids[1:min(N_TRAIN_PAT, n_pat)]))))
cd4_scale = mean(df.cd4[train_mask_rows])
rna_scale = mean(df.log10_rna[train_mask_rows])
println(@sprintf("  %d patients, %d observations. SPLIT=%s", n_pat, DataFrames.nrow(df), SPLIT))
println(@sprintf("  Normalisation: cd4_scale=%.2f cells/µL, rna_scale=%.3f log10-copies", cd4_scale, rna_scale))

# Per-patient bundles: each patient integrates from its own first observation.
struct Patient
    id::Int
    t::Vector{Float64}        # rescaled observation times (sorted)
    y::Matrix{Float64}        # 2 × n normalised observations [cd4_n; rna_n]
    u0::Vector{Float64}       # normalised IC = first observation
    train_idx::Vector{Int}    # indices into t used for the likelihood
    fore_idx::Vector{Int}     # held-out indices (forecast evaluation)
end

train_patient_set = Set(patient_ids[1:min(N_TRAIN_PAT, n_pat)])
patients = Patient[]
for pid in patient_ids
    sub = df[df.id .== pid, :]
    t   = Vector{Float64}(sub.day) ./ DAY_SCALE
    cd4n = Vector{Float64}(sub.cd4) ./ cd4_scale
    rnan = Vector{Float64}(sub.log10_rna) ./ rna_scale
    y   = permutedims(hcat(cd4n, rnan))          # 2 × n
    u0  = y[:, 1]
    n   = length(t)
    if SPLIT == "B"
        tr = findall(<=(SPLIT_DAY / DAY_SCALE), t)
        fo = findall(>(SPLIT_DAY / DAY_SCALE), t)
    else  # option A: whole patient is either train or forecast
        if pid in train_patient_set
            tr = collect(1:n); fo = Int[]
        else
            tr = Int[];        fo = collect(1:n)
        end
    end
    push!(patients, Patient(pid, t, y, u0, tr, fo))
end

train_patients = [p for p in patients if length(p.train_idx) >= 2]
fore_patients  = [p for p in patients if length(p.fore_idx) >= 1]
n_train_obs = sum(length(p.train_idx) for p in train_patients)
n_fore_obs  = sum(length(p.fore_idx)  for p in fore_patients)
println(@sprintf("  Train: %d patients / %d obs.  Forecast: %d patients / %d obs.",
                 length(train_patients), n_train_obs, length(fore_patients), n_fore_obs))

# === Network + initial params (configurable width/depth) ======================
function build_dudt(width::Int, nhidden::Int)
    layers = Any[Lux.Dense(2, width, Lux.tanh)]
    for _ in 2:nhidden
        push!(layers, Lux.Dense(width, width, Lux.tanh))
    end
    push!(layers, Lux.Dense(width, 2))
    return Lux.Chain(layers...)
end
dudt2 = build_dudt(HIDDEN, N_HIDDEN)
Random.seed!(INIT_SEED)
rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
p_struct  = ComponentArrays.ComponentArray{Float64}(p)
p_flat_nn = vec(collect(p_struct))
logσ_init = log(0.1)
p_flat_init = vcat(p_flat_nn, logσ_init)
println(@sprintf("  Architecture: %s → %d weights (+1 logσ)", ARCH_STR, length(p_flat_nn)))

# === Pooled fns (the only real departure from the single-trajectory harness) ==
unflatten_p(pf) = ComponentArrays.ComponentVector(pf, ComponentArrays.getaxes(p_struct))
neuralodefunc(u, p, t) = dudt2(u, p, st)[1]

# Solve one patient from its IC over the supplied (sorted) save-times.
function solve_patient(u0_i, ts, p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    prob_ = DE.ODEProblem(neuralodefunc, u0_i, (ts[1], ts[end]), p)
    Array(DE.solve(prob_, DE.Tsit5(), saveat=ts,
                   abstol=DEV_TOL, reltol=DEV_TOL, maxiters=Int(1e5)))
end

# Pooled Gaussian log-posterior: sum over all training patients, N(0,1) weight prior.
function l(θ_flat)
    θ_nn = unflatten_p(θ_flat[1:end-1])
    logσ = θ_flat[end]
    σ_cd4 = exp(logσ)
    σ_rna = exp(logσ) / sqrt(RNA_W)

    ll = sum(train_patients) do pt
        ts   = pt.t[pt.train_idx]
        pred = solve_patient(pt.u0, ts, θ_nn)
        ycd4 = pt.y[1, pt.train_idx]
        yrna = pt.y[2, pt.train_idx]
        r  = -0.5 * sum(((ycd4 .- pred[1, :]) ./ σ_cd4) .^ 2)
        r += -0.5 * sum(((yrna .- pred[2, :]) ./ σ_rna) .^ 2)
        r -= length(ts) * (log(σ_cd4) + log(σ_rna))
        r
    end
    lp  = -0.5 * sum(θ_flat[1:end-1] .^ 2)
    lp -= 0.5 * logσ^2
    return ll + lp
end

function dldθ(θ_flat)
    x, back = Zygote.pullback(l, θ_flat)
    return x, first(back(1))
end

# Forecast every held-out point from each patient's baseline; aggregate error.
function validation_metrics(p)
    p = p isa ComponentArrays.ComponentArray ? p : unflatten_p(p)
    sqerr = 0.0; ssd = 0.0; snorm = 0.0; npt = 0
    for pt in fore_patients
        ts_all = pt.t                                  # integrate from baseline …
        pred = solve_patient(pt.u0, ts_all, p)
        for j in pt.fore_idx                           # … score only held-out pts
            d1 = pt.y[1, j] - pred[1, j]; d2 = pt.y[2, j] - pred[2, j]
            sqerr += d1^2 + d2^2
            ssd   += pt.y[1, j]^2 + pt.y[2, j]^2
            npt   += 2
        end
    end
    mse = npt == 0 ? NaN : sqerr / npt
    rmse = sqrt(mse)
    rel_err = ssd == 0 ? NaN : sqrt(sqerr / ssd)
    return mse, rmse, rel_err
end

fns = (; unflatten_p, neuralodefunc, solve_patient, l, dldθ, validation_metrics)

# prob NamedTuple in the shape the shared run_map/run_nuts drivers expect.
prob = (; dudt2, st, p_struct, p_flat_init, logσ_init)

# Plot helpers + representative patients — defined here (before MAP) so the MAP
# figure can be written the instant MAP finishes, ahead of the long NUTS run.
to_cd4(x) = x .* cd4_scale                        # cells/µL
to_rna(x) = x .* rna_scale                        # log10 copies/mL
dense_grid(pt) = collect(range(pt.t[1], pt.t[end], length=120))
show_pat = fore_patients[unique(round.(Int, range(1, length(fore_patients), length=min(6, length(fore_patients)))))]

# === MAP pre-training (shared two-phase Adam scheduler) =======================
println("\n=== MAP pre-training (pooled) ===")
p_map, mm = run_map(prob, fns; phaseA_iters=MAP_PHASEA, phaseB_iters=MAP_PHASEB)

# --- map_fit.png : saved NOW, right after MAP (no need to wait for NUTS) -------
# Top row = CD4 (cells/µL, blue); bottom row = log10 viral load (red); one column
# per patient. Both channels on their own true scale — no rescaling.
try
    p_map_nn = unflatten_p(p_map[1:end-1])
    mp = show_pat[1:min(4, length(show_pat))]
    cd4_panels = Any[]; vl_panels = Any[]
    for (i, pt) in enumerate(mp)
        tg = dense_grid(pt); days = tg .* DAY_SCALE; od = pt.t .* DAY_SCALE
        traj = solve_patient(pt.u0, tg, p_map_nn)
        lg = i == 1
        pcd4 = Plots.plot(days, to_cd4(traj[1, :]), color=:blue, lw=2,
                          label=(lg ? "CD4 (MAP)" : ""), title="Patient $(pt.id)",
                          ylabel="CD4 (cells/µL)", legend=(lg ? :bottomright : false))
        Plots.scatter!(pcd4, od, to_cd4(pt.y[1, :]), color=:blue, alpha=0.6,
                       label=(lg ? "CD4 observed" : ""))
        Plots.hline!(pcd4, [CD4_AIDS_THRESH], color=:black, ls=:dot, label=(lg ? "AIDS=200" : ""))
        Plots.vline!(pcd4, [SPLIT_DAY], color=:gray, ls=:dash, label=(lg ? "train/forecast" : ""))

        pvl = Plots.plot(days, to_rna(traj[2, :]), color=:red, lw=2,
                         label=(lg ? "log10 VL (MAP)" : ""), xlabel="Day",
                         ylabel="log10 viral load", legend=(lg ? :topright : false))
        Plots.scatter!(pvl, od, to_rna(pt.y[2, :]), color=:red, alpha=0.6,
                       label=(lg ? "VL observed" : ""))
        Plots.hline!(pvl, [VL_FAIL_LOG10], color=:black, ls=:dot, label=(lg ? "suppression≈200cp" : ""))
        Plots.vline!(pvl, [SPLIT_DAY], color=:gray, ls=:dash, label="")
        push!(cd4_panels, pcd4); push!(vl_panels, pvl)
    end
    Plots.savefig(Plots.plot(cd4_panels..., vl_panels..., layout=(2, length(mp)), size=(1600, 760),
                             plot_title="ACTG 315 — MAP pre-training fit (point estimate, real units)"),
                  joinpath(outdir, "map_fit.png"))
    println("MAP figure → $(joinpath(outdir, "map_fit.png"))  (saved before NUTS)")
catch e
    @warn "map-fit plot failed" exception=e
end

# === NUTS (shared driver) =====================================================
println("\n=== NUTS sampling (pooled) ===")
samples, stats, nuts_rt = run_nuts(prob, fns, p_map;
                                   n_samples=NSAMP, n_adapts=NADPT, max_depth=MAXDEPTH)
diag = nuts_diagnostics(samples, stats)
n_post = size(samples, 2)

# === Posterior predictive over the forecast window ============================
# For every posterior draw, forecast each held-out patient from baseline and add
# observation noise; collect 90% predictive bands and check calibration.
# Wrapped in a function so the scalar accumulators get clean local scope (a
# top-level `for` loop would treat `hits += …` as an ambiguous soft-scope global).
σ_samples = exp.(samples[end, :])

function compute_pp_and_coverage()
    Random.seed!(NOISE_SEED)
    hits_cd4 = 0; hits_rna = 0; n_chk = 0
    pp_store = Dict{Int, Tuple{Vector{Float64}, Matrix{Float64}, Matrix{Float64}}}()
    for (pi, pt) in enumerate(fore_patients)
        nt = length(pt.t)
        pp_cd4 = fill(NaN, nt, n_post)
        pp_rna = fill(NaN, nt, n_post)
        for k in 1:n_post
            θ_nn = samples[1:end-1, k]
            σ_b = σ_samples[k]; σ_r = σ_b / sqrt(RNA_W)
            pred = try solve_patient(pt.u0, pt.t, θ_nn) catch; nothing end
            (pred === nothing || size(pred) != (2, nt) || !all(isfinite, pred)) && continue
            pp_cd4[:, k] .= pred[1, :] .+ randn(nt) .* σ_b
            pp_rna[:, k] .= pred[2, :] .+ randn(nt) .* σ_r
        end
        pp_store[pi] = (pt.t, pp_cd4, pp_rna)
        for j in pt.fore_idx
            lo_c = _finite_q(pp_cd4[j, :], 0.05); hi_c = _finite_q(pp_cd4[j, :], 0.95)
            lo_r = _finite_q(pp_rna[j, :], 0.05); hi_r = _finite_q(pp_rna[j, :], 0.95)
            if isfinite(lo_c) && isfinite(hi_c)
                hits_cd4 += (pt.y[1, j] >= lo_c && pt.y[1, j] <= hi_c); n_chk += 1
            end
            if isfinite(lo_r) && isfinite(hi_r)
                hits_rna += (pt.y[2, j] >= lo_r && pt.y[2, j] <= hi_r)
            end
        end
    end
    cov_cd4 = n_chk == 0 ? NaN : hits_cd4 / n_chk
    cov_rna = n_chk == 0 ? NaN : hits_rna / n_chk
    return pp_store, cov_cd4, cov_rna, n_chk
end

pp_store, coverage_cd4, coverage_rna, covg_n = compute_pp_and_coverage()
post = (; sigma_hat_mean = mean(σ_samples), sigma_hat_std = std(σ_samples))

# === Per-patient coverage breakdown ==========================================
# The misspecification test: does the pooled model miss specifically on the
# patients whose dynamics deviate most from the population average? We score each
# forecast patient's own coverage + signed CD4 forecast bias and pair it with an
# observable "response rate" (early CD4 / VL slope over the training window). If
# slow responders (flat early CD4) are over-predicted and fall out of the band,
# that is structural pooling error — the motivation for a hierarchical BNODE.
lin_slope(x, y) = (mx = mean(x); sx = sum((x .- mx) .^ 2);
                   sx == 0 ? NaN : sum((x .- mx) .* (y .- mean(y))) / sx)
function pearson(x, y)
    m = isfinite.(x) .& isfinite.(y)
    sum(m) < 3 && return NaN
    xx = x[m]; yy = y[m]; sx = std(xx); sy = std(yy)
    (sx == 0 || sy == 0) && return NaN
    mean((xx .- mean(xx)) .* (yy .- mean(yy))) / (sx * sy)
end

function per_patient_breakdown()
    rows = NamedTuple[]
    for (pi, pt) in enumerate(fore_patients)
        isempty(pt.fore_idx) && continue
        _, pp_cd4, pp_rna = pp_store[pi]
        hc = 0; nc = 0; hr = 0; nr = 0; resid = Float64[]
        for j in pt.fore_idx
            loc = _finite_q(pp_cd4[j, :], 0.05); hic = _finite_q(pp_cd4[j, :], 0.95)
            lor = _finite_q(pp_rna[j, :], 0.05); hir = _finite_q(pp_rna[j, :], 0.95)
            mc  = (f = filter(isfinite, pp_cd4[j, :]); isempty(f) ? NaN : mean(f))
            isfinite(loc) && isfinite(hic) && (hc += (pt.y[1, j] >= loc && pt.y[1, j] <= hic); nc += 1)
            isfinite(lor) && isfinite(hir) && (hr += (pt.y[2, j] >= lor && pt.y[2, j] <= hir); nr += 1)
            isfinite(mc) && push!(resid, (mc - pt.y[1, j]) * cd4_scale)   # predicted − observed, cells/µL
        end
        nc == 0 && continue
        trd = pt.t[pt.train_idx] .* DAY_SCALE
        push!(rows, (; id = pt.id,
            cd4_0      = pt.u0[1] * cd4_scale,
            rna_0      = pt.u0[2] * rna_scale,
            cd4_slope  = lin_slope(trd, pt.y[1, pt.train_idx] .* cd4_scale),  # cells/µL per day
            rna_slope  = lin_slope(trd, pt.y[2, pt.train_idx] .* rna_scale),  # log10 per day
            n_fore     = length(pt.fore_idx),
            cov_cd4    = hc / nc,
            cov_rna    = nr == 0 ? NaN : hr / nr,
            bias_cd4   = isempty(resid) ? NaN : mean(resid)))   # +ve ⇒ model over-predicts CD4
    end
    return rows
end

ppb = per_patient_breakdown()
if !isempty(ppb)
    CSV.write(joinpath(outdir, "per_patient_coverage.csv"), DataFrames.DataFrame(ppb))
end
slope_v   = [r.cd4_slope for r in ppb]
covcd4_v  = [r.cov_cd4   for r in ppb]
bias_v    = [r.bias_cd4  for r in ppb]
corr_slope_cov  = pearson(slope_v, covcd4_v)   # expect > 0: faster responders better covered
corr_slope_bias = pearson(slope_v, bias_v)     # expect < 0: slow responders over-predicted

# === Plots (posterior figures — map_fit.png already saved right after MAP) =====
# to_cd4/to_rna/dense_grid/show_pat are defined above (before MAP).

# --- posterior_predictive.png : 90% PP band on held-out points ----------------
try
    plts = []
    for pt in show_pat
        pi = findfirst(==(pt), fore_patients)
        _, pp_cd4, pp_rna = pp_store[pi]
        nt = length(pt.t); days = pt.t .* DAY_SCALE
        loc = to_cd4([_finite_q(pp_cd4[i, :], 0.05) for i in 1:nt])
        hic = to_cd4([_finite_q(pp_cd4[i, :], 0.95) for i in 1:nt])
        mec = to_cd4([(f=filter(isfinite, pp_cd4[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:nt])
        pc = Plots.plot(days, mec, ribbon=(mec .- loc, hic .- mec), fillalpha=0.25,
                        color=:blue, lw=2, label="CD4 90% PP", xlabel="Day", ylabel="CD4 (cells/µL)",
                        title="Patient $(pt.id)", legend=:topleft)
        Plots.scatter!(pc, days[pt.train_idx], to_cd4(pt.y[1, pt.train_idx]), color=:blue,
                       alpha=0.5, label="CD4 train")
        Plots.scatter!(pc, days[pt.fore_idx], to_cd4(pt.y[1, pt.fore_idx]), color=:darkblue,
                       marker=:star5, ms=7, label="CD4 held-out")
        Plots.hline!(pc, [CD4_AIDS_THRESH], color=:black, ls=:dot, label="AIDS=200")
        Plots.vline!(pc, [SPLIT_DAY], color=:gray, ls=:dash, label="")
        push!(plts, pc)
    end
    Plots.savefig(Plots.plot(plts..., layout=(2, 3), size=(1400, 700)),
                  joinpath(outdir, "posterior_predictive.png"))
catch e
    @warn "posterior-predictive plot failed" exception=e
end

# --- phase_space.png : (CD4, log10 VL) posterior trajectories ------------------
try
    ps = Plots.plot(xlabel="CD4 (cells/µL)", ylabel="log10 viral load",
                    title="HIV-ART posterior dynamics (CD4 ↑, VL ↓ = suppression)")
    for pt in show_pat
        tg = dense_grid(pt)
        for _ in 1:min(60, n_post)
            θ_nn = samples[1:end-1, rand(1:n_post)]
            r = try solve_patient(pt.u0, tg, θ_nn) catch; continue end
            size(r, 2) == length(tg) || continue
            Plots.plot!(ps, to_cd4(r[1, :]), to_rna(r[2, :]), alpha=0.05, color=:purple, label="")
        end
        Plots.scatter!(ps, to_cd4(pt.y[1, :]), to_rna(pt.y[2, :]), alpha=0.6, label="")
    end
    Plots.hline!(ps, [VL_FAIL_LOG10], color=:red, ls=:dot, label="suppression target")
    Plots.vline!(ps, [CD4_AIDS_THRESH], color=:black, ls=:dot, label="AIDS threshold")
    Plots.savefig(ps, joinpath(outdir, "phase_space.png"))
catch e
    @warn "phase-space plot failed" exception=e
end

# --- decision_relevance.png : precautionary clinical bounds -------------------
# The figure that makes this a clinical case study: posterior 5th-percentile CD4
# (worst-plausible immune recovery) and 95th-percentile viral load (worst-plausible
# treatment failure), each against its clinical threshold.
try
    pt = show_pat[end]                                 # one illustrative patient
    pi = findfirst(==(pt), fore_patients)
    _, pp_cd4, pp_rna = pp_store[pi]
    nt = length(pt.t); days = pt.t .* DAY_SCALE
    q05_cd4 = to_cd4([_finite_q(pp_cd4[i, :], 0.05) for i in 1:nt])
    me_cd4  = to_cd4([(f=filter(isfinite, pp_cd4[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:nt])
    q95_rna = to_rna([_finite_q(pp_rna[i, :], 0.95) for i in 1:nt])
    me_rna  = to_rna([(f=filter(isfinite, pp_rna[i, :]); isempty(f) ? NaN : mean(f)) for i in 1:nt])

    p1 = Plots.plot(days, me_cd4, color=:blue, lw=2, label="CD4 posterior mean",
                    xlabel="Day", ylabel="CD4 (cells/µL)", title="AIDS-risk lower bound (patient $(pt.id))")
    Plots.plot!(p1, days, q05_cd4, color=:blue, lw=2, ls=:dash, label="CD4 5th pct (precautionary)")
    Plots.hline!(p1, [CD4_AIDS_THRESH], color=:black, ls=:dot, label="AIDS threshold = 200")
    Plots.scatter!(p1, days, to_cd4(pt.y[1, :]), color=:blue, alpha=0.5, label="observed")

    p2 = Plots.plot(days, me_rna, color=:red, lw=2, label="VL posterior mean",
                    xlabel="Day", ylabel="log10 viral load", title="Treatment-failure upper bound")
    Plots.plot!(p2, days, q95_rna, color=:red, lw=2, ls=:dash, label="VL 95th pct (precautionary)")
    Plots.hline!(p2, [VL_FAIL_LOG10], color=:black, ls=:dot, label="suppression target ≈ 200 copies")
    Plots.scatter!(p2, days, to_rna(pt.y[2, :]), color=:red, alpha=0.5, label="observed")

    Plots.savefig(Plots.plot(p1, p2, layout=(1, 2), size=(1300, 500)),
                  joinpath(outdir, "decision_relevance.png"))
catch e
    @warn "decision-relevance plot failed" exception=e
end

# --- coverage_breakdown.png : the misspecification diagnostic ------------------
# Left:  per-patient CD4 forecast coverage vs early CD4 response rate.
# Right: per-patient signed CD4 forecast bias vs early CD4 response rate.
# Slow responders (low/flat early slope) clustering at low coverage / positive
# bias = the pooled model over-predicts recovery for them → structural pooling error.
try
    if !isempty(ppb)
        ms = 4 .+ 1.5 .* [r.n_fore for r in ppb]
        pa = Plots.scatter(slope_v, covcd4_v, markersize=ms, color=:blue, alpha=0.6,
                           xlabel="early CD4 slope, wk 0–8 (cells/µL per day)",
                           ylabel="CD4 forecast coverage", legend=:bottomright,
                           label="patient (size ∝ #held-out)",
                           title=@sprintf("Coverage vs response rate (r=%.2f)", corr_slope_cov))
        Plots.hline!(pa, [0.90], color=:black, ls=:dot, label="nominal 0.90")
        Plots.vline!(pa, [0.0],  color=:gray,  ls=:dash, label="no CD4 recovery")

        pb = Plots.scatter(slope_v, bias_v, markersize=ms, color=:purple, alpha=0.6,
                           xlabel="early CD4 slope, wk 0–8 (cells/µL per day)",
                           ylabel="CD4 forecast bias: predicted − observed (cells/µL)",
                           legend=:topright, label="patient",
                           title=@sprintf("Over-prediction vs response rate (r=%.2f)", corr_slope_bias))
        Plots.hline!(pb, [0.0], color=:black, ls=:dot, label="unbiased")
        Plots.savefig(Plots.plot(pa, pb, layout=(1, 2), size=(1300, 520)),
                      joinpath(outdir, "coverage_breakdown.png"))
    end
catch e
    @warn "coverage-breakdown plot failed" exception=e
end

# --- NUTS diagnostics (shared) ------------------------------------------------
try
    n_show = min(5, size(samples, 1))
    chn = MCMCChains.Chains(reshape(permutedims(samples[1:n_show, :]), size(samples, 2), n_show, 1))
    Plots.savefig(Plots.plot(chn),               joinpath(outdir, "chain_trace.png"))
    Plots.savefig(MCMCChains.autocorplot(chn),   joinpath(outdir, "autocor.png"))
catch e
    @warn "trace/autocorrelation plotting skipped" exception=e
end

# === Console summary =========================================================
println("\n----- ACTG 315 pooled BNODE summary -----")
println(@sprintf("Split          : %s  (train %d obs / forecast %d obs)",
                 SPLIT, n_train_obs, n_fore_obs))
println(@sprintf("MAP forecast   : rmse=%.4f  rel_err=%.3f%%", mm.map_rmse, 100*mm.map_rel_err))
println(@sprintf("Posterior σ̂    : mean=%.4f  std=%.4f (normalised units)",
                 post.sigma_hat_mean, post.sigma_hat_std))
println(@sprintf("Forecast covg  : CD4=%.3f  VL=%.3f  (target ≈0.90, n=%d pts)",
                 coverage_cd4, coverage_rna, covg_n))
println(@sprintf("Misspec test   : corr(early-slope, CD4-coverage)=%+.2f  corr(early-slope, CD4-bias)=%+.2f",
                 corr_slope_cov, corr_slope_bias))
println("                 (expect >0 and <0 respectively if slow responders are the misses → pooling error)")
println(@sprintf("NUTS diag      : accept=%.3f  EBFMI=%.3f  treedepth=%.2f  divergences=%d",
                 diag.accept, diag.ebfmi, diag.treedepth, diag.ndiverge))
println(@sprintf("NUTS quality   : ESS_min=%.2f  R̂_max=%.3f  runtime=%.1fs",
                 diag.ess_min, diag.rhat_max, nuts_rt))

# === Write results row =======================================================
append_result!(csv_out, (;
    dataset = "ACTG 315 HIV-ART (pooled, 46 patients)",
    split = SPLIT,
    n_patients = n_pat,
    n_train_obs, n_fore_obs,
    arch = ARCH_STR,
    n_weights = length(p_flat_nn),
    config = "MAP=$MAP_PHASEA/$MAP_PHASEB, NUTS=$NSAMP/$NADPT, " *
             "max_depth=$MAXDEPTH, tol=$DEV_TOL, split=$SPLIT, split_day=$SPLIT_DAY, arch=$ARCH_STR",
    init_seed = INIT_SEED,
    cd4_scale, rna_scale,
    map_rmse       = mm.map_rmse,
    map_rel_err    = mm.map_rel_err,
    sigma_hat_mean = post.sigma_hat_mean,
    sigma_hat_std  = post.sigma_hat_std,
    coverage_forecast_cd4 = coverage_cd4,
    coverage_forecast_rna = coverage_rna,
    corr_slope_cov_cd4  = corr_slope_cov,
    corr_slope_bias_cd4 = corr_slope_bias,
    accept    = diag.accept,
    ebfmi     = diag.ebfmi,
    treedepth = diag.treedepth,
    ndiverge  = diag.ndiverge,
    ess_min   = diag.ess_min,
    rhat_max  = diag.rhat_max,
    runtime_s = nuts_rt,
))

println("\nCSV   → $csv_out")
println("Plots → $outdir/")
