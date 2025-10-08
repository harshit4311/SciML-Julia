# Import necessary libraries
import CSV
import DataFrames
import Plots

# Set the plot theme to dark for a professional look
# Plots.theme(:dark)

# Define the path to the dataset
file_path = "/Users/harshit/Downloads/Research-Commons-Quant/SciML-Julia/russell-datasets/growth_value_cumulative_returns_corrected.csv"

# Load the dataset into a DataFrame
df = CSV.read(file_path, DataFrames.DataFrame)

# Create the time-series plot using the Date column for the x-axis
p = Plots.plot(
    df.Date,
    df.growth_cumulative_ret,
    label="Growth Cumulative Returns",
    xlabel="Date",
    ylabel="Cumulative Returns",
    title="Cumulative Returns of Growth and Value ETFs",
    legend=:topleft
)

Plots.plot!(
    p,
    df.Date,
    df.value_cumulative_ret,
    label="Value Cumulative Returns"
)

# Add a grid
Plots.plot!(grid=true)

# Save the plot to a file
Plots.savefig(p, "cumulative_returns_dataset_plot_styled.png")

println("Styled plot saved to cumulative_returns_dataset_plot_styled.png")

# To run this file, open the Julia REPL in your project directory and execute:
# include("code/plot_russell_dataset.jl")