# Import necessary libraries
import CSV
import DataFrames
import Plots

# Define the path to the dataset
file_path = "/Users/harshit/Downloads/Research-Commons-Quant/SciML-Julia/russell-datasets/growth_value_cumulative_returns_corrected.csv"

# Load the dataset into a DataFrame
df = CSV.read(file_path, DataFrames.DataFrame)

# Create the time-series plot
# The row number is used as the implicit x-axis (time)
p = Plots.plot(
    df.growth_cumulative_ret,
    label="Growth Cumulative Returns",
    xlabel="Time Step",
    ylabel="Cumulative Returns",
    title="Russell Growth vs. Value Cumulative Returns",
    legend=:outertopright
)

Plots.plot!(
    p,
    df.value_cumulative_ret,
    label="Value Cumulative Returns"
)

# Save the plot to a file
Plots.savefig(p, "cumulative_returns_dataset_plot.png")

println("Plot saved to cumulative_returns_dataset_plot.png")

# To run this file, open the Julia REPL in your project directory and execute:
# include("code/plot_russell_dataset.jl")