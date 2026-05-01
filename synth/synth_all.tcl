# synth_all.tcl — Batch synthesis for AI_MMUL_Unit modules
# Target: Artix-7 xc7a35tcpg236-1 | Clock: 125 MHz (8 ns) | Vivado 2025.2
# Brendan Lynskey 2026

set_param general.maxThreads 2

set part     "xc7a35tcpg236-1"
set clk_ns   8.0
set log_dir  "synthesis_logs"
file mkdir $log_dir

# {top_module {source files}}
set modules {
    {mmul_pe                  {rtl/mmul_pkg.sv rtl/mmul_pe.sv}}
    {mmul_dotproduct          {rtl/mmul_pkg.sv rtl/mmul_dotproduct.sv}}
    {mmul_systolic_os         {rtl/mmul_pkg.sv rtl/mmul_pe.sv rtl/mmul_systolic_os.sv}}
    {mmul_weight_stationary   {rtl/mmul_pkg.sv rtl/mmul_weight_stationary.sv}}
}

set summary {}

foreach mod $modules {
    set top   [lindex $mod 0]
    set srcs  [lindex $mod 1]

    puts "=========================================="
    puts "Synthesising: $top"
    puts "=========================================="

    create_project -in_memory -part $part

    foreach src $srcs {
        read_verilog -sv $src
    }

    read_xdc synth/clock.xdc

    # -mode out_of_context: treat as internal block (no IO buffers).
    # Required because these modules expose hundreds of bits of packed-array
    # ports — far more than the cpg236 package's 106 user IO pins.  This
    # measures the array's intrinsic timing/area, not the cost of bringing
    # signals to physical pins (which an integrator handles at chip top).
    synth_design -top $top -part $part -mode out_of_context
    opt_design
    place_design
    route_design

    report_utilization     -file "$log_dir/${top}_utilization.rpt"
    report_timing_summary  -file "$log_dir/${top}_timing.rpt"

    set util_rpt   [report_utilization -return_string]
    set timing_rpt [report_timing_summary -return_string]

    set luts "?"
    if {[regexp {Slice LUTs\s*\|\s*(\d+)} $util_rpt -> val]} { set luts $val }
    set ffs "?"
    if {[regexp {Slice Registers\s*\|\s*(\d+)} $util_rpt -> val]} { set ffs $val }
    set bram "0"
    if {[regexp {Block RAM Tile\s*\|\s*(\S+)} $util_rpt -> val]} { set bram $val }
    set dsp "0"
    if {[regexp {DSPs\s*\|\s*(\d+)} $util_rpt -> val]} { set dsp $val }

    set fmax "?"
    if {[regexp {\n\s+(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\s+\d+\s+\d+\s+} $timing_rpt -> wns_val tns_val]} {
        set wns [expr {double($wns_val)}]
        set fmax [format "%.1f" [expr {1000.0 / ($clk_ns - $wns)}]]
    }

    lappend summary [list $top $luts $ffs $bram $dsp $fmax]

    puts "  LUTs=$luts  FFs=$ffs  BRAM=$bram  DSP=$dsp  Fmax=$fmax MHz"

    close_project
}

puts ""
puts "=========================================="
puts "AI_MMUL_Unit Synthesis Summary"
puts "=========================================="
puts [format "%-25s %6s %6s %6s %6s %10s" "Module" "LUTs" "FFs" "BRAM" "DSP" "Fmax(MHz)"]
puts [string repeat "-" 65]
foreach entry $summary {
    puts [format "%-25s %6s %6s %6s %6s %10s" {*}$entry]
}
puts ""
puts "Target: $part | Clock: [expr {1000.0 / $clk_ns}] MHz ($clk_ns ns)"
