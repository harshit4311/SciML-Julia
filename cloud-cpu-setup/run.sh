#!/usr/bin/env bash
set -e

SESSION="exp20_run"
ROOT="$HOME/cloud-cpu-setup"

cd "$ROOT"

# Create dirs if they don't exist
mkdir -p logs checkpoints

# Prevent accidental duplicate runs
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

# Start detached tmux session
tmux new-session -d -s "$SESSION" "
    julia --project=. exp_20.jl | tee logs/exp_20.log
"

echo "✅ exp_20 started successfully"
echo "📟 tmux session : $SESSION"
echo "🔌 Attach with  : tmux attach -t $SESSION"
