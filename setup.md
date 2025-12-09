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