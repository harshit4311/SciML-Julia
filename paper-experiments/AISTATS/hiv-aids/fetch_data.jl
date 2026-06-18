#=
fetch_data.jl — download + tidy the ACTG 315 longitudinal HIV dataset.

Source: AIDS Clinical Trials Group protocol 315 (ACTG 315), distributed by
Dr. Hulin Wu (UTHealth Houston, Dept. of Biostatistics):

  https://sph.uth.edu/dept/bads/faculty-home/hulinwu/datasets/actg315longitudinaldataviralload

46 chronically HIV-infected adults followed for up to 28 weeks after starting
combination ART (zidovudine + lamivudine + ritonavir). Each blood draw records
HIV-1 viral load (log10 RNA copies/mL; values below the 100-copy assay limit are
imputed at 50 copies → log10 = 1.699) and CD4+ T-cell count (cells/µL).

The raw file columns are:  Obs.No  Patid  Day  log10(RNA)  CD4
We keep the four meaningful fields and rename for clarity. The result is written
to data/actg315.csv (committed so the experiment runs offline/reproducibly).

Run:  julia --project=../../.. fetch_data.jl
=#

import Downloads
import CSV
import DataFrames

const URL = "https://sph.uth.edu/dept/bads/faculty-home/hulinwu/datasets/data/ACTG315LongitudinalDataViralLoadData.txt"
const OUT = joinpath(@__DIR__, "data", "actg315.csv")
mkpath(dirname(OUT))

println("Downloading ACTG 315 from Hulin Wu's UTHealth dataset page…")
raw = Downloads.download(URL)

# The file is whitespace-delimited with a header line:
#   Obs.No Patid Day log10(RNA) CD4
df = CSV.read(raw, DataFrames.DataFrame; delim=' ', ignorerepeated=true)
DataFrames.rename!(df, names(df)[2] => :id, names(df)[3] => :day,
                       names(df)[4] => :log10_rna, names(df)[5] => :cd4)

tidy = DataFrames.DataFrame(
    id        = Int.(round.(df.id)),
    day       = Int.(round.(df.day)),
    log10_rna = round.(Float64.(df.log10_rna), digits=4),
    cd4       = round.(Float64.(df.cd4),       digits=4),
)
DataFrames.sort!(tidy, [:id, :day])
CSV.write(OUT, tidy)

npat = length(unique(tidy.id))
println("Wrote $(DataFrames.nrow(tidy)) observations from $npat patients → $OUT")
