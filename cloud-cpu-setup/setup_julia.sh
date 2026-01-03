#!/usr/bin/env bash
set -e

# ===== CONFIG =====
JULIA_VERSION="1.11.6"
JULIA_MAJOR_MINOR="1.11"
INSTALL_DIR="$HOME/julia"
PROFILE_FILE="$HOME/.bashrc"
# ==================

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    wget \
    tar \
    git \
    build-essential \
    htop \
    tmux

echo "Downloading Julia $JULIA_VERSION..."
cd /tmp
wget https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz

echo "Installing Julia..."
mkdir -p $INSTALL_DIR
tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C $INSTALL_DIR --strip-components=1

echo "Adding Julia to PATH..."
if ! grep -q "julia/bin" $PROFILE_FILE; then
    echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> $PROFILE_FILE
fi

source $PROFILE_FILE

echo "Julia installed successfully:"
which julia
julia --version
