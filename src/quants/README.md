# Quantized Weight Kernels

This directory provides code for consuming quantized weights. Three layers are
tightly coupled:
a host-side repack from standard weight formats into device-friendly blocked formats,
a device-side unpack of the custom blocks, and
a matrix multiplication with bfloat16 activation vectors.

## Performance

Single-core 256x64 `mmul.py` benchmarks. Repack and transport use 256x16
tiles; each tile is padded to a 64-byte boundary.

| format | bits/w | % PM [1] | est w/s [2] | limit |
| :-- | --: | --: | --: | :-- |
| Q4_0 | 4.5 | 24.7% | 94B | compute at 1.5, memory 98B at 1.8 GHz |
| Q4_0_packed16 | 4.5 | 23.2% | 98B | near-exact, memory-bound at 1.8 GHz |
| Q4_0_hilo | 5.0 | 22.2% | 88B | memory, +11% size |
| Q6_K | 6.625 | 30.5% | 66B | memory-bound, 66B |
| Q6_K_int8 | 8.625 | 26.7% | 51B | memory, +31% size |
| Q8_0 | 8.5 | 43.8% | 52B | memory |

[1]: Kernel static instruction size, as a percentage of
the 16 KiB AIE2P Compute Tile program memory.

[2]: Theoretical Billions of weights/second processed by the full NPU:
the minimum of the compute limit (`weights/cycle * 1.5 GHz * 32`) and
the memory limit (`55 GB/s / bytes-per-weight`). The fused mmul is the
production path; standalone unpack cycle counts are diagnostic only.

Measured 2026-11-18 on Linux 7.3.5 with the current compiler, 256x64 shape
(fused totals, then per tile): Q8_0 7707 cycles (240.8 cycles/tile, 2.126
weights/cycle, exact canonical fp16 scales); Q4_0 7883 cycles (246.3, 2.078
weights/cycle, max abs error 1.53e-4); Q4_0_packed16 7397 cycles (231.16,
2.215 weights/cycle, max abs error 1.8125 under its named 5%-of-column
policy).  The exact and packed16 Q4_0 variants use the same 9216 packed bytes
for this shape, so the only possible win is cycles: currently 6.2% (7397
against 7883).

The program variant was removed and the format stays in `src/quants` with its
tests; packed16 is near-exact rather than bit-exact, so it cannot serve as the
unprefixed Q4_0.  Details in `src/program/NOTES.md`.

## Package layout

```text
<format>/layout.h       packed transport-tile constants and pointer helpers
<format>/repack.h       host-side canonical-to-tile conversion
<format>/unpack.aie.h   device-side dequantization (exact except q4_0_packed16)
<format>/mmul.aie.h     device-side fused dequantization and matrix multiplication
common/                 shared tile shape, scale, repack, and mmul helpers
tests/                  host repack shims, NPU harnesses, NumPy references
```

Format names are canonical GGUF types; suffixes denote packed variants
(`q4_0_packed16`, `q4_0_hilo`, `q6_k_int8`).

Loop and compiler-optimization findings, including the pointer-increment and
unroll-pragma results, are in `NOTES.md` under "Loop optimization and Peano
idioms".
