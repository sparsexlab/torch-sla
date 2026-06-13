#!/usr/bin/env bash
# Launch the distributed torch-sla examples.
#
#   ./launch.sh              # run every example with 4 procs
#   ./launch.sh matvec 2     # one example, 2 procs
#   ./launch.sh all 8        # everything, 8 procs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE=${1:-all}
NPROC=${2:-4}

run() {
    echo
    echo "========== $1 (nproc=$NPROC) =========="
    torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/$2"
}

case "$EXAMPLE" in
    matvec)      run "Distributed Matvec"       distributed_matvec.py ;;
    solve)       run "Distributed CG Solve"     distributed_solve.py ;;
    eigsh)       run "Distributed LOBPCG eigsh" distributed_eigsh.py ;;
    persistence) run "Distributed Persistence"  distributed_persistence.py ;;
    all)
        run "Distributed Matvec"       distributed_matvec.py
        run "Distributed CG Solve"     distributed_solve.py
        run "Distributed LOBPCG eigsh" distributed_eigsh.py
        run "Distributed Persistence"  distributed_persistence.py
        ;;
    *)
        echo "Usage: $0 {matvec|solve|eigsh|persistence|all} [nproc]"
        exit 1
        ;;
esac

echo
echo "All examples completed."
