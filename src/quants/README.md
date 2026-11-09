# Quantized Weight Kernels

This directory provides code for consuming quantized weights. Three layers are
tightly coupled:
a host-side repack from standard weight formats into device-friendly blocked formats,
a device-side unpack of the custom blocks, and
a matrix multiplication with bfloat16 activation vectors.

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

## Status

Single-core 256x64 `mmul.py` results:

| format | cycles | packed size (bits/weight) | vs canonical | weights/cycle | estimated weights/s (32 cores) |
| :-- | --: | --: | --: | --: | --: |
| Q4_0 | 9872 | 4.5 | 0% | 1.660 | 96B |
| Q4_0_hilo | 7267 | 5.0 | +11.1% | 2.255 | 88B |
| Q6_K | 25957 | 6.5625 | 0% | 0.631 | 36B |
| Q6_K_int8 | 10254 | 8.75 | +33.3% | 1.598 | 50B |
| Q8_0 | 8245 | 8.5 | 0% | 1.987 | 52B |

All benchmark shapes use 256x64 tiles, and the test harness requires
`weights_rows` to be a multiple of 256. Formats may use smaller internal
records; Q4_0, for example, still packs 64-row records.

The weights/s are a simple theoretical ballpark, assuming linear scaling from
the single-core test to 32 cores, limited by either compute or memory:

- compute speed limit = `weights/cycle * 1.8 GHz * 32`
- memory bandwidth limit = `55 GB/s / bytes-per-weight`

The recommended variants (`q4_0`, `q4_0_hilo`, `q6_k_int8`, and `q8_0`) are
memory-bound or near-balanced at 55 GB/s. Native `q6_k` is compute-bound by
its vectorized bit-plane decode; it is retained as a size and optimization
baseline.

## Reproduce

```sh
source .../mlir-aie/iron_env.sh

python src/quants/tests/mmul.py --format q4_0
python src/quants/tests/mmul.py --format q4_0_hilo
python src/quants/tests/mmul.py --format q6_k
python src/quants/tests/mmul.py --format q6_k_int8
python src/quants/tests/mmul.py --format q8_0
```
