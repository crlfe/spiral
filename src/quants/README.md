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
<format>/unpack.aie.h   device-side dequantization test
<format>/mmul.aie.h     device-side fused dequantization and matrix multiplication
common/                 shared tile shape, scale, repack, and mmul helpers
tests/                  host repack shims, NPU harnesses, NumPy references
```

## Status

Single-core 64-column `mmul.py` results:

| format | tile | cycles | packed size (bits/weight) | vs canonical | weights/cycle | estimated weights/s (32 cores) |
| :-- | :-- | --: | --: | --: | --: | --: |
| Q4_0 | 64x64 | 1880 | 5.0 | +11.1% | 2.179 | 88B |
| Q6_K | 256x64 | 10254 | 8.75 | +33.3% | 1.598 | 50B |
| Q8_0 | 256x64 | 8245 | 8.5 | 0% | 1.987 | 52B |

The weights/s are a simple theoretical ballpark, assuming linear scaling from
the single-core test to 32 cores, limited by either compute or memory:

- compute speed limit = `weights/cycle * 1.5 GHz * 32`
- memory bandwidth limit = `55 GB/s / bytes-per-weight`

All three quantizations are currently limited by memory bandwidth.

## Reproduce

```sh
source .../mlir-aie/iron_env.sh

python src/quants/tests/mmul.py --format q4_0
python src/quants/tests/mmul.py --format q6_k
python src/quants/tests/mmul.py --format q8_0
```
