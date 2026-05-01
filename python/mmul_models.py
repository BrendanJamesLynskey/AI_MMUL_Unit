"""
mmul_models.py - Reference Python models for hardware Matrix Multiplier Units.

Each model is a *cycle-level functional* description of one MMUL dataflow.
The same C = A * B^T result is produced; the difference is *how* operands
are streamed, where partial sums live, and what is reused.

Used as the golden model for the SystemVerilog and CocoTB testbenches in
the parent repo.

Dataflows implemented:
  - naive_matmul         : O(N^3) reference
  - output_stationary    : TPU-style systolic array (A flows W, B flows S,
                           partial sum accumulates in PE)
  - weight_stationary    : NVDLA-style (B/W pre-loaded, A streams)
  - input_stationary     : A pre-loaded, B/W streams
  - dot_product_tree     : NVIDIA Tensor-Core-style broadcast + tree

Number-format helpers:
  - q_fixed              : signed Q(I.F) fixed-point
  - bf16, fp16           : packed bfloat16, IEEE binary16
  - fp8_e4m3, fp8_e5m2   : Hopper FP8 variants
  - mxfp8, mxfp4         : OCP MX block-scaled formats (32-elem block, E8M0 scale)
  - int8                 : symmetric per-tensor INT8

This file has no third-party deps — only struct/math from the stdlib.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Q-format fixed-point  (default Q8.8 to match the SV implementation)
# ----------------------------------------------------------------------------

class QFixed:
    """Signed Q(I.F) fixed-point, two's-complement, total = I + F + 1 bits."""

    def __init__(self, int_bits: int = 8, frac_bits: int = 8):
        self.I = int_bits
        self.F = frac_bits
        self.W = int_bits + frac_bits  # signed: 1 sign + I-1 int + F frac.
        self.scale = 1 << frac_bits
        self.max = (1 << (self.W - 1)) - 1
        self.min = -(1 << (self.W - 1))

    def to_int(self, x: float) -> int:
        raw = int(round(x * self.scale))
        if raw > self.max:
            raw = self.max
        elif raw < self.min:
            raw = self.min
        return raw & ((1 << self.W) - 1)

    def to_float(self, raw: int) -> float:
        if raw & (1 << (self.W - 1)):
            raw -= 1 << self.W
        return raw / self.scale

    def mul_acc(self, a_raw: int, w_raw: int, acc: int, acc_w: int = 32) -> int:
        """signed mul-accumulate, returning acc_w-bit two's complement int."""
        a = a_raw if a_raw < (1 << (self.W - 1)) else a_raw - (1 << self.W)
        w = w_raw if w_raw < (1 << (self.W - 1)) else w_raw - (1 << self.W)
        prod = a * w  # 2W-bit signed
        new_acc = acc + prod
        mask = (1 << acc_w) - 1
        # Wrap into acc_w bits (2's complement)
        return new_acc & mask


# ----------------------------------------------------------------------------
# Floating-point helpers — packed-int representations
# ----------------------------------------------------------------------------

def fp32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_fp32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def to_bf16(x: float) -> int:
    """Round-to-nearest-even truncate of FP32 to BF16 (16-bit)."""
    bits = fp32_to_bits(x)
    # round-to-nearest-even on bottom 16 bits
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bits = (bits + rounding_bias) >> 16
    return bits & 0xFFFF


def from_bf16(b: int) -> float:
    return bits_to_fp32((b & 0xFFFF) << 16)


def to_fp16(x: float) -> int:
    """IEEE-754 binary16 (FP16) round-to-nearest-even."""
    f = struct.unpack("<I", struct.pack("<f", x))[0]
    sign = (f >> 16) & 0x8000
    exp32 = (f >> 23) & 0xFF
    mant = f & 0x7FFFFF
    if exp32 == 0xFF:  # Inf/NaN
        return sign | 0x7C00 | (0x200 if mant else 0)
    e = exp32 - 127 + 15
    if e >= 31:
        return sign | 0x7C00  # +/-inf
    if e <= 0:
        if e < -10:
            return sign  # too small -> 0
        mant |= 0x800000
        shift = 14 - e
        round_bit = (mant >> (shift - 1)) & 1
        result = mant >> shift
        if round_bit and (mant & ((1 << (shift - 1)) - 1) or (result & 1)):
            result += 1
        return sign | result
    rounded = (mant + 0x1000) >> 13
    if rounded >= 0x400:
        rounded = 0
        e += 1
        if e >= 31:
            return sign | 0x7C00
    return sign | (e << 10) | (rounded & 0x3FF)


def from_fp16(b: int) -> float:
    sign = (b >> 15) & 1
    exp = (b >> 10) & 0x1F
    mant = b & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        # subnormal
        return ((-1) ** sign) * mant * (2 ** -24)
    if exp == 31:
        return float("nan") if mant else ((-1) ** sign) * float("inf")
    return ((-1) ** sign) * (1 + mant / 1024.0) * (2.0 ** (exp - 15))


# FP8 variants — return an 8-bit unsigned int.
def to_fp8(x: float, e_bits: int, m_bits: int) -> int:
    """Generic FP8 quantizer. e4m3 -> (4,3), e5m2 -> (5,2)."""
    if x == 0:
        return 0
    sign = 1 if x < 0 else 0
    x = abs(x)
    bias = (1 << (e_bits - 1)) - 1
    e_max = (1 << e_bits) - 1
    if math.isinf(x) or math.isnan(x):
        return (sign << 7) | (e_max << m_bits) | ((1 << m_bits) - 1 if math.isnan(x) else 0)
    e = math.floor(math.log2(x)) if x > 0 else 0
    biased = e + bias
    if biased >= e_max:
        biased = e_max - 1
        m_int = (1 << m_bits) - 1
    elif biased < 1:
        # subnormal-ish: clamp to smallest normal for simplicity
        biased = 0
        m_int = 0
    else:
        mant = x / (2 ** e) - 1.0
        m_int = int(round(mant * (1 << m_bits)))
        if m_int >= (1 << m_bits):
            m_int = 0
            biased += 1
            if biased >= e_max:
                biased = e_max - 1
                m_int = (1 << m_bits) - 1
    return (sign << 7) | (biased << m_bits) | m_int


def from_fp8(b: int, e_bits: int, m_bits: int) -> float:
    sign = (b >> 7) & 1
    bias = (1 << (e_bits - 1)) - 1
    exp = (b >> m_bits) & ((1 << e_bits) - 1)
    mant = b & ((1 << m_bits) - 1)
    if exp == 0:
        return ((-1) ** sign) * mant * (2.0 ** (1 - bias - m_bits))
    return ((-1) ** sign) * (1 + mant / (1 << m_bits)) * (2.0 ** (exp - bias))


# OCP MX block-scaled — block of K=32 elements share one E8M0 (8-bit unsigned)
# power-of-two scale.  Element type is FP4/FP6/FP8 etc.
@dataclass
class MXBlock:
    scale_e8m0: int       # 0..255, value = 2^(scale_e8m0 - 127); 255 => NaN
    elements: list[int]   # encoded element ints

    K = 32  # OCP MX block size


def quantize_mx(values: list[float], elem_to_int, elem_max_repr: float) -> MXBlock:
    """Quantise K=32 floats into an MX block sharing one E8M0 scale."""
    assert len(values) == MXBlock.K
    amax = max(abs(v) for v in values)
    if amax == 0:
        return MXBlock(scale_e8m0=0, elements=[elem_to_int(0.0)] * len(values))
    # Choose scale so that max |value| / scale fits in elem_max_repr.
    e = math.ceil(math.log2(amax / elem_max_repr)) if amax > 0 else 0
    scale = 2.0 ** e
    encoded = [elem_to_int(v / scale) for v in values]
    return MXBlock(scale_e8m0=(e + 127) & 0xFF, elements=encoded)


# ----------------------------------------------------------------------------
# Reference matrix multiplications  --- C = A x B  (M x K)*(K x N) = (M x N)
# ----------------------------------------------------------------------------

def naive_matmul(A: list[list[int]],
                 B: list[list[int]],
                 q: QFixed) -> list[list[int]]:
    """O(N^3) matmul in fixed-point — golden model."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    assert len(B) == K
    C = [[0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc = q.mul_acc(A[i][k], B[k][j], acc)
            C[i][j] = acc & 0xFFFFFFFF
    return C


# ----------------------------------------------------------------------------
# Output-stationary systolic array  (TPU-style, A flows ->, B flows v)
# ----------------------------------------------------------------------------

def output_stationary(A: list[list[int]],
                      B: list[list[int]],
                      q: QFixed,
                      rows: int = None,
                      cols: int = None) -> list[list[int]]:
    """Cycle-accurate model of an MxN output-stationary systolic array.

    On each cycle:
      - row r of array sees A[r][k]    (skewed: arrives at PE (r,c) at cycle r+c+k)
      - col c of array sees B[k][c]    (same skew)
      - PE(r,c) accumulates A[r][k]*B[k][c] for all k.
    """
    M = len(A); K = len(A[0]); N = len(B[0])
    rows = rows or M
    cols = cols or N
    assert M == rows and N == cols, "Tile-matrix-size must match array shape"
    # acc[r][c] == final partial sum at PE(r,c)
    acc = [[0] * cols for _ in range(rows)]

    # Skewed inputs:  a_skew[t][r] = A[r][t-r] if 0 <= t-r < K else 0
    #                 b_skew[t][c] = B[t-c][c] if 0 <= t-c < K else 0
    T = K + rows + cols  # plenty of cycles for the wave to drain
    for t in range(T):
        # Inputs for this cycle (skewed entry)
        for r in range(rows):
            for c in range(cols):
                k = t - r - c       # which inner-product term arrives now
                if 0 <= k < K:
                    a_raw = A[r][k]
                    b_raw = B[k][c]
                    acc[r][c] = q.mul_acc(a_raw, b_raw, acc[r][c])
    return [[v & 0xFFFFFFFF for v in row] for row in acc]


# ----------------------------------------------------------------------------
# Weight-stationary  (B is pre-loaded, A streams)
# ----------------------------------------------------------------------------

def weight_stationary(A: list[list[int]],
                      B: list[list[int]],
                      q: QFixed) -> list[list[int]]:
    """Functional WS model: B held in PE grid, activations broadcast over K."""
    M = len(A); K = len(A[0]); N = len(B[0])
    out = [[0] * N for _ in range(M)]
    for k in range(K):                      # one K-step per cycle
        for r in range(M):
            for c in range(N):
                out[r][c] = q.mul_acc(A[r][k], B[k][c], out[r][c])
    return [[v & 0xFFFFFFFF for v in row] for row in out]


# ----------------------------------------------------------------------------
# Input-stationary
# ----------------------------------------------------------------------------

def input_stationary(A: list[list[int]],
                     B: list[list[int]],
                     q: QFixed) -> list[list[int]]:
    M = len(A); K = len(A[0]); N = len(B[0])
    out = [[0] * N for _ in range(M)]
    # A is held in registers; B streams.  Functionally equivalent.
    for r in range(M):
        for k in range(K):
            for c in range(N):
                out[r][c] = q.mul_acc(A[r][k], B[k][c], out[r][c])
    return [[v & 0xFFFFFFFF for v in row] for row in out]


# ----------------------------------------------------------------------------
# Dot-product tree (broadcast)  - models NVIDIA Tensor Core conceptually
# ----------------------------------------------------------------------------

def dot_product_tree(A: list[list[int]],
                     B: list[list[int]],
                     q: QFixed) -> list[list[int]]:
    """K parallel multipliers feeding an adder tree, one (M,N) tile per group."""
    M = len(A); K = len(A[0]); N = len(B[0])
    out = [[0] * N for _ in range(M)]
    for r in range(M):
        for c in range(N):
            # Compute K products in parallel, then sum via tree.
            prods = []
            a_raw = [A[r][k] for k in range(K)]
            b_raw = [B[k][c] for k in range(K)]
            for k in range(K):
                a = a_raw[k] if a_raw[k] < (1 << (q.W - 1)) else a_raw[k] - (1 << q.W)
                b = b_raw[k] if b_raw[k] < (1 << (q.W - 1)) else b_raw[k] - (1 << q.W)
                prods.append(a * b)
            # Adder tree (associative for integers; for FP would matter)
            while len(prods) > 1:
                nxt = []
                for i in range(0, len(prods), 2):
                    nxt.append(prods[i] + (prods[i + 1] if i + 1 < len(prods) else 0))
                prods = nxt
            out[r][c] = prods[0] & 0xFFFFFFFF
    return out


# ----------------------------------------------------------------------------
# Convenience: random-fill matrices in float, return both raw+float for tests
# ----------------------------------------------------------------------------

def quantise_matrix(M_float: list[list[float]], q: QFixed) -> list[list[int]]:
    return [[q.to_int(v) for v in row] for row in M_float]


def dequantise_matrix(M_raw: list[list[int]], q: QFixed,
                      acc_bits: int = 32) -> list[list[float]]:
    """Convert MAC accumulator values (raw int) back to float."""
    out = []
    for row in M_raw:
        out_row = []
        for v in row:
            if v & (1 << (acc_bits - 1)):
                v -= 1 << acc_bits
            out_row.append(v / (q.scale * q.scale))  # two scale factors from a*b
        out.append(out_row)
    return out
