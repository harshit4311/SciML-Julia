#=
fetch_data.jl — download + tidy the Blasius et al. (2020) rotifer–algae
chemostat predator–prey time series.

Source: Blasius B, Rudolf L, Weithoff G, Gaedke U, Fussmann G.F.
"Long-term cyclic persistence in an experimental predator–prey system."
Nature 577, 226–230 (2020).  https://doi.org/10.1038/s41586-019-1857-0

Data deposited on figshare (CC BY 4.0):
  https://doi.org/10.6084/m9.figshare.10045976.v1
  "Time series of long-term experimental predator–prey cycles"

Ten chemostat experiments (C1…C10) of the planktonic predator–prey system:
green alga *Monoraphidium minutum* (prey) grazed by the rotifer
*Brachionus calyciflorus* (predator), run under constant temperature/light
with nitrogen-limited inflow. The ten files correspond to the 10 experiments
in Extended Data Table 1 of the paper. Each row is one measurement day.

Raw columns (whitespace after the comma in the header):
  time (days), algae (10^6 cells/ml), rotifers (animals/ml), egg-ratio,
  eggs (per ml), dead animals (per ml), external medium (mu mol N / l)

Missing measurements are coded `NaN` in the source and carried through as
`missing` in the tidy file. We concatenate all ten experiments, tag each with
an integer `experiment` id (1…10), and keep the measured fields under clean
names. The two Lotka–Volterra state variables for the BNODE are `algae`
(prey) and `rotifers` (predator); the remaining columns (predator life-stage
characteristics + inflow nitrogen) are retained for reference.

Result is written to data/blasius_rotifer_algae.csv (committed so the
experiment runs offline/reproducibly).

Run:  julia --project=../../.. fetch_data.jl
=#

import Downloads
import CSV
import DataFrames

# figshare file ids for the ten experiments (article 10045976, v1).
const FILE_IDS = [
    (1,  18105239),
    (2,  18105233),
    (3,  18105230),
    (4,  18105227),
    (5,  18105224),
    (6,  18105221),
    (7,  18105218),
    (8,  18105212),
    (9,  18105215),
    (10, 18105236),
]
const BASE = "https://ndownloader.figshare.com/files/"
const OUT  = joinpath(@__DIR__, "data", "blasius_rotifer_algae.csv")
mkpath(dirname(OUT))

# NaN → missing so the tidy CSV has empty cells for absent measurements.
nan2miss(x) = (x isa Real && isnan(x)) ? missing : x

parts = DataFrames.DataFrame[]
for (expt, fid) in FILE_IDS
    println("Downloading experiment C$expt (figshare file $fid)…")
    raw = Downloads.download(BASE * string(fid))
    df = CSV.read(raw, DataFrames.DataFrame)        # comma-delimited, 7 columns

    # Source columns are positional; rename by position for robustness against
    # the spaces/units embedded in the header text.
    n = names(df)
    tidy = DataFrames.DataFrame(
        experiment = fill(expt, DataFrames.nrow(df)),
        day        = round.(Float64.(df[!, n[1]]), digits=4),
        algae      = nan2miss.(round.(Float64.(df[!, n[2]]), digits=4)),  # 10^6 cells/ml (prey)
        rotifers   = nan2miss.(round.(Float64.(df[!, n[3]]), digits=4)),  # animals/ml (predator)
        egg_ratio  = nan2miss.(round.(Float64.(df[!, n[4]]), digits=4)),
        eggs       = nan2miss.(round.(Float64.(df[!, n[5]]), digits=4)),  # per ml
        dead       = nan2miss.(round.(Float64.(df[!, n[6]]), digits=4)),  # dead animals per ml
        medium_N   = nan2miss.(round.(Float64.(df[!, n[7]]), digits=4)),  # external medium, µmol N / l
    )
    push!(parts, tidy)
end

all = reduce(vcat, parts)
DataFrames.sort!(all, [:experiment, :day])
CSV.write(OUT, all)

nexpt = length(unique(all.experiment))
println("Wrote $(DataFrames.nrow(all)) measurement-days from $nexpt experiments → $OUT")
