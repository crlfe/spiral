# Quantized Weight Kernels

This directory provides code for consuming quantized weights. Three layers are
tightly coupled:
a host-side repack from standard weight formats into device-friendly blocked formats,
a device-side unpack of the custom blocks, and
a matrix multiplication with bfloat16 activation vectors.

## Performance

Single-core 256x64 `mmul.py` benchmarks:

| format | bits/w | % PM [1] | est w/s [2] | limit |
| :-- | --: | --: | --: | :-- |
| Q4_0 | 4.5 | 16.8% | 90B | memory, 98B at 1.8 GHz |
| Q4_0_hilo | 5.0 | 22.1% | 88B | memory, +11% size |
| Q6_K | 6.5625 | 13.0% | 41B | compute, 50B at 1.8 GHz |
| Q6_K_int8 | 8.75 | 29.0% | 50B | memory, +33% size |
| Q8_0 | 8.5 | 16.1% | 52B | memory |

[1]: Kernel static instruction size, as a percentage of
the 16 KiB AIE2P Compute Tile program memory.

[2]: Theoretical Billions of weights/second processed by the full NPU:
the minimum of the compute limit (`weights/cycle * 1.5 GHz * 32`) and
the memory limit (`55 GB/s / bytes-per-weight`).

## Package layout

```text
<format>/layout.h       packed record/chunk constants and pointer helpers
<format>/repack.h       host-side canonical-to-packed conversion
<format>/unpack.aie.h   device-side exact dequantization
<format>/mmul.aie.h     device-side fused dequantization and matrix multiplication
common/                 shared tile shape, scale, repack, and mmul helpers
tests/                  host repack shims, NPU harnesses, NumPy references
```

Format names are canonical GGUF types; suffixes denote packed variants
(`q4_0_hilo`, `q6_k_int8`).
