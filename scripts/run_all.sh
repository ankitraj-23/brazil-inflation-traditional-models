#!/usr/bin/env bash
#
# Run the full Phase 2 traditional-models pipeline, in order.
#
#   1. prepare_data.py          parse raw CEIC exports -> clean dataset
#   2. run_baselines.py         RW, RW_AO, AR1
#   3. run_var.py               VAR1 (+ combined so far)
#   4. run_phillips_backward.py PC_BACKWARD (+ full combined file)
#
# Raw CEIC CSVs must already be in data/raw/ (see README.md). This script never
# prints, copies, or commits raw data — it only invokes the pipeline stages.

set -euo pipefail

# Run from the repository root regardless of the caller's working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

section() {
    echo ""
    echo "========================================================================"
    echo ">>> $1"
    echo "========================================================================"
}

section "1/4  Preparing clean dataset (src/prepare_data.py)"
python src/prepare_data.py

section "2/4  Baseline forecasts: RW, RW_AO, AR1 (src/run_baselines.py)"
python src/run_baselines.py

section "3/4  VAR(1) forecasts (src/run_var.py)"
python src/run_var.py

section "4/4  Backward Phillips Curve forecasts (src/run_phillips_backward.py)"
python src/run_phillips_backward.py

section "DONE  All Phase 2 stages completed successfully."
