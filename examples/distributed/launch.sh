#!/usr/bin/env bash
# Launch the distributed torch-sla examples.
#
# Single node:
#   ./launch.sh                 # run every example with 4 procs
#   ./launch.sh matvec 2        # one example, 2 procs
#   ./launch.sh all 8           # everything, 8 procs
#
# Multiple nodes: set RDZV_ENDPOINT (the head node's IP:PORT) on EVERY
# node and run the SAME command on each. NNODES / RDZV_ID are honored if
# set (defaults: NNODES from the host count you pass, RDZV_ID=sla).
#   # on each node:
#   RDZV_ENDPOINT=HEAD_NODE_IP:29500 NNODES=2 ./launch.sh all 4
#
# The example scripts auto-select NCCL+CUDA when a GPU is visible (one
# GPU per LOCAL_RANK) and fall back to gloo+CPU otherwise.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE=${1:-all}
NPROC=${2:-4}

# Multi-node rendezvous: if RDZV_ENDPOINT is set, use c10d rendezvous
# across NNODES nodes; otherwise run a single-node standalone job.
RDZV_ENDPOINT=${RDZV_ENDPOINT:-}
NNODES=${NNODES:-1}
RDZV_ID=${RDZV_ID:-sla}

if [[ -n "$RDZV_ENDPOINT" ]]; then
    LAUNCH=(torchrun
        --nnodes="$NNODES" --nproc_per_node="$NPROC"
        --rdzv-id="$RDZV_ID" --rdzv-backend=c10d
        --rdzv-endpoint="$RDZV_ENDPOINT")
    MODE="multi-node (nnodes=$NNODES, rdzv=$RDZV_ENDPOINT)"
else
    LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC")
    MODE="single-node (standalone)"
fi

run() {
    echo
    echo "========== $1 ($MODE, nproc=$NPROC) =========="
    "${LAUNCH[@]}" "$SCRIPT_DIR/$2"
}

case "$EXAMPLE" in
    matvec)              run "Distributed Matvec"               distributed_matvec.py ;;
    solve)               run "Distributed CG Solve"             distributed_solve.py ;;
    eigsh)               run "Distributed LOBPCG eigsh"         distributed_eigsh.py ;;
    persistence)         run "Distributed Persistence"          distributed_persistence.py ;;
    connected_components) run "Distributed connected_components" distributed_connected_components.py ;;
    nonlinear_solve)     run "Distributed nonlinear_solve"      distributed_nonlinear_solve.py ;;
    all)
        run "Distributed Matvec"               distributed_matvec.py
        run "Distributed CG Solve"             distributed_solve.py
        run "Distributed LOBPCG eigsh"         distributed_eigsh.py
        run "Distributed Persistence"          distributed_persistence.py
        run "Distributed connected_components" distributed_connected_components.py
        run "Distributed nonlinear_solve"      distributed_nonlinear_solve.py
        ;;
    *)
        echo "Usage: $0 {matvec|solve|eigsh|persistence|connected_components|nonlinear_solve|all} [nproc]"
        echo "Multi-node: set RDZV_ENDPOINT=HEAD_IP:PORT NNODES=<n> (run on every node)."
        exit 1
        ;;
esac

echo
echo "All examples completed."
