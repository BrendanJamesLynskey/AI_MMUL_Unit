// =============================================================================
// tb_mmul_systolic_os.sv - 4x4 output-stationary systolic-array testbench
// =============================================================================
// (Icarus-compatible: avoids passing unpacked arrays through subroutine ports
// by using module-scope arrays directly inside tasks.)
// =============================================================================

`timescale 1ns/1ps

module tb_mmul_systolic_os;
  import mmul_pkg::*;

  localparam int ROWS = 4;
  localparam int COLS = 4;
  localparam int K    = 4;

  logic  clk = 0;
  logic  rst_n = 0;
  logic  clear = 0;
  logic  enable = 0;
  logic signed [ROWS-1:0][DATA_WIDTH-1:0] a_in;
  logic signed [COLS-1:0][DATA_WIDTH-1:0] b_in;
  logic signed [ROWS*COLS-1:0][ACC_WIDTH-1:0] result;

  always #5 clk = ~clk;

  mmul_systolic_os #(.ROWS(ROWS), .COLS(COLS)) dut (.*);

  int passed = 0;
  int failed = 0;

  // Module-scope tile operands and reference
  data_t tile_A [ROWS][K];
  data_t tile_B [K][COLS];
  acc_t  tile_ref [ROWS*COLS];

  function automatic data_t q88(input real x);
    return data_t'($rtoi(x * 256.0));
  endfunction

  task automatic compute_reference;
    for (int r = 0; r < ROWS; r++) begin
      for (int c = 0; c < COLS; c++) begin
        acc_t s = '0;
        for (int kk = 0; kk < K; kk++) s += acc_t'(tile_A[r][kk] * tile_B[kk][c]);
        tile_ref[r*COLS + c] = s;
      end
    end
  endtask

  task automatic run_tile;
    int total_cycles;
    total_cycles = K + ROWS + COLS - 2 + 4;
    clear = 1; @(posedge clk); clear = 0; @(posedge clk);
    enable = 1;
    for (int t = 0; t < total_cycles; t++) begin
      for (int r = 0; r < ROWS; r++) begin
        int kk;
        kk = t - r;
        a_in[r] = (kk >= 0 && kk < K) ? tile_A[r][kk] : '0;
      end
      for (int c = 0; c < COLS; c++) begin
        int kk;
        kk = t - c;
        b_in[c] = (kk >= 0 && kk < K) ? tile_B[kk][c] : '0;
      end
      @(posedge clk);
    end
    enable = 0;
    for (int r = 0; r < ROWS; r++) a_in[r] = '0;
    for (int c = 0; c < COLS; c++) b_in[c] = '0;
    @(posedge clk);
  endtask

  task automatic check_tile(input string name);
    int local_pass = 0;
    int local_fail = 0;
    compute_reference;
    run_tile;
    for (int r = 0; r < ROWS; r++) begin
      for (int c = 0; c < COLS; c++) begin
        int idx = r*COLS + c;
        if (result[idx] === tile_ref[idx]) local_pass++;
        else begin
          $display("  [FAIL] %s C[%0d][%0d] exp=%0d got=%0d",
                   name, r, c, $signed(tile_ref[idx]), $signed(result[idx]));
          local_fail++;
        end
      end
    end
    if (local_fail == 0)
      $display("  [PASS] %-30s : all %0d elements match", name, ROWS*COLS);
    passed += local_pass;
    failed += local_fail;
  endtask

  task automatic clear_tile;
    for (int r = 0; r < ROWS; r++)
      for (int kk = 0; kk < K; kk++) tile_A[r][kk] = '0;
    for (int kk = 0; kk < K; kk++)
      for (int c = 0; c < COLS; c++) tile_B[kk][c] = '0;
  endtask

  initial begin
    $display("====================================================");
    $display(" tb_mmul_systolic_os : %0dx%0d output-stationary array", ROWS, COLS);
    $display("====================================================");

    rst_n = 0; enable = 0; clear = 0;
    for (int r = 0; r < ROWS; r++) a_in[r] = '0;
    for (int c = 0; c < COLS; c++) b_in[c] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // ---- Test 1: identity * range ----
    clear_tile;
    for (int r = 0; r < ROWS; r++)
      tile_A[r][r] = q88(1.0);
    for (int kk = 0; kk < K; kk++)
      for (int c = 0; c < COLS; c++)
        tile_B[kk][c] = q88($itor(kk*COLS + c + 1));
    check_tile("identity * range(1..16)");

    // ---- Test 2: 2x2 worked example padded ----
    clear_tile;
    tile_A[0][0] = q88(1.0); tile_A[0][1] = q88(2.0);
    tile_A[1][0] = q88(3.0); tile_A[1][1] = q88(4.0);
    tile_B[0][0] = q88(5.0); tile_B[0][1] = q88(6.0);
    tile_B[1][0] = q88(7.0); tile_B[1][1] = q88(8.0);
    check_tile("2x2 worked example");

    // ---- Test 3: random ----
    for (int trial = 0; trial < 5; trial++) begin
      for (int r = 0; r < ROWS; r++)
        for (int kk = 0; kk < K; kk++)
          tile_A[r][kk] = data_t'($random % 1024);
      for (int kk = 0; kk < K; kk++)
        for (int c = 0; c < COLS; c++)
          tile_B[kk][c] = data_t'($random % 1024);
      check_tile($sformatf("random trial #%0d", trial));
    end

    $display("");
    $display("====================================================");
    $display(" tb_mmul_systolic_os : %0d PASSED, %0d FAILED", passed, failed);
    $display("====================================================");

    if (failed) $fatal(1, "tb_mmul_systolic_os: failures");
    $finish;
  end

  initial begin
    #200000;
    $fatal(1, "tb_mmul_systolic_os TIMEOUT");
  end

endmodule
