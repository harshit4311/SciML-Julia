#!/usr/bin/env julia

using Pkg

@info "Activating local project environment"
Pkg.activate(@__DIR__)

@info "Instantiating dependencies from Manifest.toml"
Pkg.instantiate()

@info "Precompiling dependencies (this may take a while)"
Pkg.precompile()

@info "Dependency setup complete"
