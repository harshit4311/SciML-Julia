# Setup Guide

This repository contains Julia code for running **Bayesian Neural ODEs** for **Lotka–Volterra (LV) modeling**, using:

- **Julia 1.11.6**
- **SciML / DifferentialEquations.jl**
- **Lux.jl** for neural networks
- **AdvancedHMC.jl** for NUTS/HMC sampling
- Reproducible environment via `Project.toml` + `Manifest.toml`

Follow this guide to reproduce the environment and run the code.

---

## 1. Install Julia (v1.11.6)

Download Julia 1.11.6:

https://julialang.org/downloads/

Verify installation:

```bash
julia --version
```

Expected:
```bash
julia version 1.11.6
```

---

## 2. Clone the Repository 
```bash
git clone https://github.com/Research-Commons/julia-SciML.git
cd julia-SciML
```

---

## 3. Activate the Project Environment
```bash
julia --project=.
```

Inside Julia, run:
```bash
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
This installs all package versions exactly as recorded in the project and manifest files.

---

## 4. Run the Code from the root (example, to run Experiment-1)
```bash
cd SciML-Julia
julia --project=. synthetic-dataset-experiments/exp_1.jl
```