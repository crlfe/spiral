> Role: orientation. Current as of 2026-11-24. History: `git log -- src/quants/README.md`.

# Quantized weight kernels

Header-only code for consuming quantized weights. Three layers are tightly
coupled: a host-side repack from canonical weight formats into device-friendly
blocked tiles, a device-side unpack, and a matrix multiply with bfloat16
activation vectors.

The production path for a model is the fixed program in
[`../program`](../program), which calls the fused kernels here. The standalone
unpack loops exist only for format conformance testing.

## Package layout

```text
<format>/layout.h       packed transport-tile constants and pointer helpers
<format>/repack.h       host-side canonical-to-tile conversion
<format>/unpack.aie.h   device-side dequantization (test-only entry points)
<format>/mmul.aie.h     device-side fused dequantization and mmul
common/                 shared tile shape, scale, repack and mmul helpers
tests/                  host repack shims, NPU harnesses, NumPy references
```

Formats: `q4_0`, `q4_0_packed16` (near-exact), `q4_0_hilo` (host-expanded
scales), `q6_k`, `q6_k_int8`, `q8_0`. Names are canonical GGUF types; suffixes
denote packed variants. The unprefixed names are exact and keep the canonical
tile size; everything else is a named variant with its own verification policy.

## Running the tests

```sh
source tools/env.sh
python src/quants/tests/unpack.py --format q4_0 --cols 64
python src/quants/tests/mmul.py   --format q4_0 --cols 64
```

`unpack.py` is the exact format/layout conformance test; `mmul.py` is the
end-to-end fused correctness and cycle benchmark. Q6_K needs `--cols 64` or
`128` in the single-tile harness. `QUANTS_KERNEL_FLAGS` appends Peano flags.

## Where to go next

- Techniques, kernel structure, per-format layouts, and the tile/repack/unpack
  API: [`DESIGN.md`](DESIGN.md).
- Cycles, weights/cycle, `.text`, and full-NPU estimates:
  [`../../docs/measurements.md`](../../docs/measurements.md).
- Why a previous approach lost: [`../../docs/rejected.md`](../../docs/rejected.md).
- Open formats and multi-core work: [`TODO.md`](TODO.md).
