# Import necessary libraries
import CSV
import DataFrames
import Plots

# Define the path to the dataset
file_path = "/Users/harshit/Downloads/Research-Commons-Quant/SciML-Julia/russell-datasets/detrended_20_russell_growth_value_predator_prey.csv"

# Load the dataset into a DataFrame
df = CSV.read(file_path, DataFrames.DataFrame)

# Create the time-series plot
# The row number is used as the implicit x-axis (time)
p = Plots.plot(
    df.Growth_Population,
    label="Growth Population",
    xlabel="Time Step",
    ylabel="Detrended Value",
    title="Detrended Russell Growth vs. Value Time Series",
    legend=:outertopright
)

Plots.plot!(
    p,
    df.Value_Population,
    label="Value Population"
)

# Save the plot to a file
Plots.savefig(p, "detrended_dataset_plot.png")

println("Plot saved to detrended_dataset_plot.png")

# To run this file, open the Julia REPL in your project directory and execute:
# include("code/plot_dataset.jl")