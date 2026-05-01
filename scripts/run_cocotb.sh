#!/usr/bin/env bash
# =============================================================================
# run_cocotb.sh - Execute every CocoTB testbench in the repo.
# =============================================================================
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/tb/cocotb"

for mk in Makefile.pe Makefile.dotproduct Makefile.systolic ; do
  echo "=========================================================="
  echo "  $mk"
  echo "=========================================================="
  rm -rf sim_build results.xml
  make -f "$mk"
done

echo
echo "All CocoTB testbenches finished."
