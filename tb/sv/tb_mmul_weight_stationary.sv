// =============================================================================
// tb_mmul_weight_stationary.sv - Weight-stationary MMUL testbench
// =============================================================================
`timescale 1ns/1ps

module tb_mmul_weight_stationary;
  import mmul_pkg::*;

  localparam int M = 4;
  localparam int N = 4;
  localparam int K = 4;

  logic  clk = 0;
  logic  rst_n = 0;
  logic  clear = 0;
  logic  load_w = 0;
  logic [$clog2(K)-1:0] load_w_row = 0;
  logic signed [N-1:0][DATA_WIDTH-1:0] w_row;
  logic  compute_en = 0;
  logic [$clog2(K)-1:0] k_index = 0;
  logic signed [M-1:0][DATA_WIDTH-1:0] a_col;
  logic signed [M*N-1:0][ACC_WIDTH-1:0] acc;

  always #5 clk = ~clk;

  mmul_weight_stationary #(.M(M), .N(N), .K(K)) dut (.*);

  int passed = 0;
  int failed = 0;

  // Module-scope operand storage
  data_t tile_A [M][K];
  data_t tile_B [K][N];

  function automatic data_t q88(input real x);
    return data_t'($rtoi(x * 256.0));
  endfunction

  task automatic load_weights;
    load_w = 1;
    for (int kk = 0; kk < K; kk++) begin
      load_w_row = kk[$clog2(K)-1:0];
      for (int nn = 0; nn < N; nn++) w_row[nn] = tile_B[kk][nn];
      @(posedge clk);
    end
    load_w = 0;
    for (int nn = 0; nn < N; nn++) w_row[nn] = '0;
    @(posedge clk);
  endtask

  task automatic stream_activations;
    compute_en = 1;
    for (int kk = 0; kk < K; kk++) begin
      k_index = kk[$clog2(K)-1:0];
      for (int r = 0; r < M; r++) a_col[r] = tile_A[r][kk];
      @(posedge clk);
    end
    compute_en = 0;
    for (int r = 0; r < M; r++) a_col[r] = '0;
    @(posedge clk);
  endtask

  task automatic check_against_reference(input string name);
    int lp = 0;
    int lf = 0;
    for (int r = 0; r < M; r++) begin
      for (int c = 0; c < N; c++) begin
        acc_t s = '0;
        for (int kk = 0; kk < K; kk++) s += acc_t'(tile_A[r][kk] * tile_B[kk][c]);
        if (acc[r*N + c] === s) lp++;
        else begin
          $display("  [FAIL] %s C[%0d][%0d] exp=%0d got=%0d",
                   name, r, c, $signed(s), $signed(acc[r*N + c]));
          lf++;
        end
      end
    end
    if (lf == 0) $display("  [PASS] %-30s : all %0d elements match", name, M*N);
    passed += lp;
    failed += lf;
  endtask

  task automatic clear_tile;
    for (int r = 0; r < M; r++) for (int kk = 0; kk < K; kk++) tile_A[r][kk] = '0;
    for (int kk = 0; kk < K; kk++) for (int c = 0; c < N; c++) tile_B[kk][c] = '0;
  endtask

  initial begin
    $display("====================================================");
    $display(" tb_mmul_weight_stationary : M=%0d N=%0d K=%0d", M, N, K);
    $display("====================================================");

    rst_n = 0;
    for (int n = 0; n < N; n++) w_row[n] = '0;
    for (int r = 0; r < M; r++) a_col[r] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // ---- Test 1: 2x2 worked example padded ----
    clear_tile;
    tile_A[0][0] = q88(1.0); tile_A[0][1] = q88(2.0);
    tile_A[1][0] = q88(3.0); tile_A[1][1] = q88(4.0);
    tile_B[0][0] = q88(5.0); tile_B[0][1] = q88(6.0);
    tile_B[1][0] = q88(7.0); tile_B[1][1] = q88(8.0);
    clear = 1; @(posedge clk); clear = 0; @(posedge clk);
    load_weights;
    stream_activations;
    check_against_reference("2x2 padded");

    // ---- Test 2: random ----
    for (int trial = 0; trial < 5; trial++) begin
      for (int r = 0; r < M; r++) for (int kk = 0; kk < K; kk++) tile_A[r][kk] = data_t'($random % 512);
      for (int kk = 0; kk < K; kk++) for (int c = 0; c < N; c++) tile_B[kk][c] = data_t'($random % 512);
      clear = 1; @(posedge clk); clear = 0; @(posedge clk);
      load_weights;
      stream_activations;
      check_against_reference($sformatf("random #%0d", trial));
    end

    $display("");
    $display("====================================================");
    $display(" tb_mmul_weight_stationary : %0d PASSED, %0d FAILED", passed, failed);
    $display("====================================================");
    if (failed) $fatal(1, "tb_mmul_weight_stationary failed");
    $finish;
  end

  initial begin
    #200000;
    $fatal(1, "tb_mmul_weight_stationary TIMEOUT");
  end

endmodule
