# Clock constraint — 125 MHz (8.0 ns), matches CRC / LFSR convention.
create_clock -period 8.0 -name clk [get_ports clk]
