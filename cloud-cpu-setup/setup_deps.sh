#!/usr/bin/env bash
set -e

ROOT="$HOME/cloud-cpu-setup"

cd $ROOT

julia --project=. setup_deps.jl
