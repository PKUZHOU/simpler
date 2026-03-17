#!/bin/bash
# End-to-end test: distributed TREDUCE on a2a3sim platform (no hardware required).
#
# Verifies the full sim pipeline:
#   1. Compile runtime + orchestration + kernel for a2a3sim
#   2. Build distributed_worker with DISTRIBUTED_PLATFORM=sim
#   3. Launch 2 worker processes with POSIX shared-memory RDMA window mock
#   4. Verify golden output
#
# Usage:
#   cd 3rd/simpler
#   bash distributed/tests/test_sim_e2e.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIMPLER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DISTRIBUTED_ROOT="$SIMPLER_ROOT/distributed"

echo "=== Distributed TREDUCE Sim E2E Test ==="
echo "SIMPLER_ROOT: $SIMPLER_ROOT"
echo ""

# Use the host_build_graph treduce_distributed example with a2a3sim platform
KERNELS_DIR="$SIMPLER_ROOT/examples/a2a3/host_build_graph/treduce_distributed/kernels"
GOLDEN="$SIMPLER_ROOT/examples/a2a3/host_build_graph/treduce_distributed/golden.py"

if [ ! -f "$KERNELS_DIR/kernel_config.py" ]; then
    echo "ERROR: kernel_config.py not found at $KERNELS_DIR"
    exit 1
fi

cd "$SIMPLER_ROOT"

python3 examples/scripts/run_example.py \
    -k "$KERNELS_DIR" \
    -g "$GOLDEN" \
    -p a2a3sim \
    --distributed \
    --nranks 2

echo ""
echo "=== Sim E2E Test PASSED ==="
