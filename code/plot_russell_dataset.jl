# ----------------------------------------------------
# Import necessary libraries
# ----------------------------------------------------
import CSV
import DataFrames
import Plots
using Dates

# ----------------------------------------------------
# Define the path to the new dataset
# ----------------------------------------------------
file_path = "/Users/harshit/Downloads/Research-Commons-Quant/SciML-Julia/russell-datasets/detrended_200_russell_indices.csv"

# ----------------------------------------------------
# Load the dataset into a DataFrame
# ----------------------------------------------------
df = CSV.read(file_path, DataFrames.DataFrame)

# Ensure the date column is of Date type by extracting the first 10 characters
df.date = Date.(first.(string.(df.date), 10))

# ----------------------------------------------------
# Create the time-series plot with improved x-axis ticks
# ----------------------------------------------------

# Determine the range of years for clear ticks
start_year = year(minimum(df.date))
end_year = year(maximum(df.date))
xticks = [Date(y) for y in start_year:end_year+1]

p = Plots.plot(
    df.date,
    df.Growth_detrended,
    label = "Growth (Detrended)",
    xlabel = "Date",
    ylabel = "Detrended Value",
    title = "Detrended Growth vs Value Indices (Russell 200)",
    legend = :topleft,
    grid = true,
    size = (900, 500),
    xticks = xticks,
    xrotation = 45
)

# Add Value series to the same plot
Plots.plot!(
    p,
    df.date,
    df.Value_detrended,
    label = "Value (Detrended)"
)

# ----------------------------------------------------
# Save the plot
# ----------------------------------------------------
Plots.savefig(p, "detrended_russell_indices_plot.png")

println("✅ Styled plot with improved x-axis saved to detrended_russell_indices_plot.png")
