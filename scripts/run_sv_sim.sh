#!/usr/bin/env bash
# =============================================================================
# run_sv_sim.sh - Compile and run all SystemVerilog testbenches with iverilog.
# =============================================================================
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RTL=$ROOT/rtl
TB=$ROOT/tb/sv
SIM=$ROOT/sim_results
mkdir -p "$SIM"

run_tb () {
  local top=$1
  local extra_rtl=$2
  local out=$SIM/$top.log
  echo "=========================================================="
  echo "  Compiling and running $top"
  echo "=========================================================="
  iverilog -g2012 -o "$SIM/$top.vvp" \
    "$RTL/mmul_pkg.sv" $extra_rtl "$TB/${top}.sv"
  ( cd "$SIM" && vvp "$top.vvp" ) | tee "$out"
}

run_tb tb_mmul_pe                "$RTL/mmul_pe.sv"
run_tb tb_mmul_systolic_os       "$RTL/mmul_pe.sv $RTL/mmul_systolic_os.sv"
run_tb tb_mmul_dotproduct        "$RTL/mmul_dotproduct.sv"
run_tb tb_mmul_weight_stationary "$RTL/mmul_weight_stationary.sv"

echo
echo "All SV testbenches finished.  Logs in $SIM"
