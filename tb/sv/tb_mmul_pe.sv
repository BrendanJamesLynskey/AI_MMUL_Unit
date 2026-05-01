// =============================================================================
// tb_mmul_pe.sv - Self-checking testbench for mmul_pe
// =============================================================================
`timescale 1ns/1ps

module tb_mmul_pe;
  import mmul_pkg::*;

  logic   clk = 0;
  logic   rst_n = 0;
  logic   clear = 0;
  logic   enable = 0;
  data_t  a_in = 0;
  data_t  w_in = 0;
  data_t  a_out;
  data_t  w_out;
  acc_t   acc_out;

  always #5 clk = ~clk;

  mmul_pe dut (.*);

  int passed = 0;
  int failed = 0;

  task automatic check(string name, longint exp, longint got);
    if (exp === got) begin
      $display("  [PASS] %-40s exp=%0d got=%0d", name, exp, got);
      passed++;
    end else begin
      $display("  [FAIL] %-40s exp=%0d got=%0d", name, exp, got);
      failed++;
    end
  endtask

  initial begin
    $display("================================================");
    $display(" tb_mmul_pe - Processing-Element selfcheck");
    $display("================================================");

    // ---- reset ----
    rst_n  = 0; clear = 0; enable = 0; a_in = 0; w_in = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // ---- Test 1: reset clears acc ----
    check("reset zeros acc",            0, acc_out);

    // ---- Test 2: simple Q8.8 MAC: 2.0 * 3.0 = 6.0 ----
    // 2.0 -> 0x0200, 3.0 -> 0x0300; product = 0x60000 (6.0 in Q16.16)
    enable = 1;
    a_in = 16'sh0200;
    w_in = 16'sh0300;
    @(posedge clk);
    enable = 0; a_in = 0; w_in = 0;
    @(posedge clk);
    check("Q8.8 2.0*3.0 -> 6.0 (raw 0x60000)", 32'h60000, acc_out);

    // ---- Test 3: accumulate three more terms ----
    // +1.0 * 4.0 = +4.0
    enable = 1;
    a_in = 16'sh0100; w_in = 16'sh0400;
    @(posedge clk);
    a_in = 16'sh0200; w_in = 16'sh0500;  // +2.0*5.0 = +10.0
    @(posedge clk);
    a_in = 16'sh0100; w_in = 16'sh0100;  // +1.0*1.0 = +1.0
    @(posedge clk);
    enable = 0; a_in = 0; w_in = 0;
    @(posedge clk);
    // 6 + 4 + 10 + 1 = 21.0  -> raw = 21 * 65536 = 0x150000
    check("accumulator sums (6+4+10+1)*65536", 32'h150000, acc_out);

    // ---- Test 4: clear ----
    clear = 1;
    @(posedge clk);
    clear = 0;
    @(posedge clk);
    check("clear zeros acc",          0, acc_out);

    // ---- Test 5: signed negative product ----
    // -1.0 * +2.0 = -2.0  raw = -0x20000
    enable = 1;
    a_in = -16'sh0100;     // -1.0 in Q8.8
    w_in =  16'sh0200;     // +2.0
    @(posedge clk);
    enable = 0; a_in = 0; w_in = 0;
    @(posedge clk);
    check("signed -1.0 * +2.0 = -2.0",  -32'sh20000, $signed(acc_out));

    // ---- Test 6: a_out / w_out skew ----
    clear = 1; @(posedge clk); clear = 0;
    enable = 1;
    a_in = 16'sh0AAA; w_in = 16'sh1234;
    @(posedge clk);
    check("a_out forwards a_in",  16'sh0AAA, $signed(a_out));
    check("w_out forwards w_in",  16'sh1234, $signed(w_out));
    enable = 0;

    // ---- Summary ----
    $display("");
    $display("================================================");
    $display(" tb_mmul_pe : %0d PASSED, %0d FAILED", passed, failed);
    $display("================================================");

    if (failed) $fatal(1, "tb_mmul_pe failed");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "tb_mmul_pe TIMEOUT");
  end

endmodule
