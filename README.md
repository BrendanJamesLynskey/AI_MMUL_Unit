# AI Matrix Multiplier Units (MMUL)

A deep-dive study of the **hardware Matrix Multiplier Unit** — the kernel of every modern AI accelerator. The repo contains an interactive presentation, a structured technical brief, four parameterised RTL implementations, a Python golden model, and a comprehensive verification suite (SystemVerilog + CocoTB).

## ▶ [Open the interactive presentation](https://brendanjameslynskey.github.io/AI_MMUL_Unit/)

> **Setup:** GitHub Pages serves `index.html` directly from the main branch. Open locally with any browser if needed — no build step.

---

## What's inside

| Path | Contents |
|------|----------|
| `index.html` | Interactive 24-slide presentation: history, architectures, number formats, memory/cache, performance, power & thermals |
| `docs/research_brief.md` | Dense 4,200-word reference — TPU/Tensor Core/Cerebras/Groq/Dojo/Sohu specs, citations |
| `rtl/` | Four parameterised SystemVerilog MMUL implementations |
| `python/` | Cycle-accurate Python golden model + number-format quantisers (BF16, FP16, FP8, MX, NVFP4, INT8) |
| `tb/sv/` | Self-checking SystemVerilog testbenches (4 modules) |
| `tb/cocotb/` | CocoTB testbenches driven by the Python golden model |
| `scripts/` | Master runners: `run_sv_sim.sh`, `run_cocotb.sh` |

---

## RTL implementations

| Module | Style | Lat. | Throughput | Used by |
|--------|-------|-----:|-----------:|---------|
| `mmul_pe.sv` | Single MAC processing element with skew flops | 1 cyc | 1 MAC/cyc | Building block |
| `mmul_systolic_os.sv` | Output-stationary systolic array (TPU-style) | K + R + C − 2 | R·C MACs/cyc | TPU v1, AWS Trainium, Trillium |
| `mmul_weight_stationary.sv` | Weight-stationary, broadcast | K (post-load) | M·N MACs/cyc | NVDLA, Apple ANE, Hailo, Ethos-N |
| `mmul_dotproduct.sv` | K-wide combinational dot product / reduction tree | 1 cyc (comb) | K MACs/cyc | NVIDIA Tensor Core lane (conceptual) |

All RTL is parameterised in `rtl/mmul_pkg.sv` (DATA_WIDTH, FRAC_BITS, ACC_WIDTH, ROWS, COLS, K_DIM) and uses **Q8.8 signed fixed-point** by default with 32-bit accumulators. The Python model in `python/mmul_models.py` additionally implements BF16, FP16, FP8 (E4M3 / E5M2), MX block-scaled (32-element block, E8M0 scale), and NVFP4 quantisers.

---

## Verification

| Tier | Tool | Tests | Status |
|------|------|------:|--------|
| Python golden model | `python3 test_mmul_models.py` | 7 | ✅ PASS |
| SystemVerilog testbenches | `iverilog` via `scripts/run_sv_sim.sh` | 248 | ✅ PASS |
| CocoTB — driven by Python golden | `scripts/run_cocotb.sh` | 10 | ✅ PASS |

### Run everything

```bash
# Python golden model
python3 python/test_mmul_models.py

# SystemVerilog testbenches (Icarus Verilog 12+)
bash scripts/run_sv_sim.sh

# CocoTB testbenches (cocotb 2.0+, Icarus Verilog)
bash scripts/run_cocotb.sh
```

The CocoTB tests import the very same Python model used to develop the algorithms — so the RTL is checked against an executable specification, not a hand-rolled SV reference (although there is also one of those in `tb/sv/`).

---

## Worked example: 2×2 matmul on the systolic array

```
A = [[1, 2],     B = [[5, 6],     C = A·B = [[19, 22],
     [3, 4]]          [7, 8]]                [43, 50]]
```

In the output-stationary 4×4 array (with 2×2 padded to 4×4), inputs are skewed:

| t | a_in[0] | a_in[1] | b_in[0] | b_in[1] |
|--:|--------:|--------:|--------:|--------:|
| 0 |     1.0 |     0.0 |     5.0 |     0.0 |
| 1 |     2.0 |     3.0 |     7.0 |     6.0 |
| 2 |     0.0 |     4.0 |     0.0 |     8.0 |
| 3 |     0.0 |     0.0 |     0.0 |     0.0 |

After `K + ROWS + COLS − 2 = 8` cycles, PE(0,0).acc = 1·5 + 2·7 = 19, PE(0,1).acc = 1·6 + 2·8 = 22, etc. Verified bit-exact in `tb_mmul_systolic_os.sv` and `test_mmul_systolic_os.py`.

---

## Topics covered in the presentation

1. **Why a dedicated MMUL?** — operational intensity, the Horowitz energy equation, the 200× DRAM-vs-MAC gap, spatial reuse.
2. **History** — Kung & Leiserson 1978/1982, iWarp, GAPP, TPU v1, V100 Tensor Cores.
3. **Architectures** — output-/weight-/input-/row-stationary, broadcast trees, TMA / TMEM.
4. **What runs on an MMUL inside a Transformer** — Q/K/V projections, scores, output proj, FFN up/down, LM head; what does *not* (RMSNorm, softmax, RoPE, residual add, embedding gather); the prefill-vs-decode regimes.
5. **Number formats** — FP32 → TF32 → FP16/BF16 → FP8 (E4M3/E5M2) → FP6 → FP4, INT8/4, OCP MX block-scaled, NVFP4.
6. **Real systems** — TPU v1 → Trillium → Ironwood, NVIDIA V100 → B200, Cerebras WSE-3, Groq LPU, Tesla Dojo, AWS Trainium2, Apple ANE, Intel AMX, Arm SME, Etched Sohu, Tenstorrent Wormhole/Blackhole.
7. **Memory hierarchy & cache utility** — HBM, on-die SRAM, weight & activation buffers, why caches are *not* a great fit and what replaces them.
8. **Performance** — peak vs sustained TOPS/TFLOPS, utilisation, roofline.
9. **Power & thermals** — energy per MAC, package power, voltage delivery, hotspot density, liquid cooling.
10. **Tradeoffs** — array size vs utilisation, format vs accuracy, fill bubbles, reconfiguration.
11. **Future directions** — block-scaled microformats, FP4 training, optical I/O, transformer-only ASICs.

---

## Repository conventions

- SV style follows the rest of [BrendanJamesLynskey/Hardware](https://github.com/BrendanJamesLynskey/Hardware) — packed multi-dim ports, packages, snake_case modules, Q8.8 fixed-point default.
- Python golden models have **no third-party dependencies** — pure stdlib for portability.
- Every TB self-checks with PASS/FAIL counters; CI-friendly exit codes.

---

## License & provenance

Educational content; cite primary sources (TPU papers, NVIDIA whitepapers, OCP MX spec) for any quantitative claim. References live in `docs/research_brief.md`.
