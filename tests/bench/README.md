# Bench Experiments

Small standalone experiments that answer one question at a time about the AMD
XDNA2 NPU. This is a staging area for understanding and optimization, not a
test suite:

- nothing under `src/` imports these files;
- nothing here is gated by CI;
- a proven helper moves to `src/quants/common/`;
- a confirmed finding is recorded in `src/quants/NOTES.md`.

Detailed results, history, and rejected experiments live in
[`NOTES.md`](NOTES.md). This README is the map and the current recommendation.

## Experiments

| file | question |
| :-- | :-- |
| `float16_support.py` | Which float16 operations does the compiler and XDNA2 actually accept? |
| `float16_convert.py` | How fast is raw float16 bits to f32? |
| `float16_scale.py` | How do we split and apply float16-derived scales with two or three bf16 terms? |

Each experiment is a Python driver plus a `.aie.cc` kernel shim. The shared
compile/run/trace code is in `bench_common.py`; the shared device helpers are
in `float16_bench.aie.h`. Every probe reports cycles, values per iteration, a
correctness number against a NumPy reference, and the compiler diagnostic for
anything that does not build.

## Running

Activate the MLIR-AIE environment, then run from this directory:

```sh
source ~/Projects/mlir-aie/ironenv/bin/activate
python float16_scale.py
```

Useful flags:

```sh
python float16_scale.py --list            # probe names
python float16_scale.py --compile-only    # no device or tracing
python float16_scale.py --probe float16_shift_split_bf16
python float16_scale.py --seed 7
```

`--compile-only` uses the same Peano invocation as the JIT path, so it needs
the AIE environment but not the NPU device. The trace ring occasionally drops
the closing `event1()`; the driver retries that run up to three times.

## Best choices

| condition | choice |
| :-- | :-- |
| Canonical two-byte float16 scales, exact, denorms required | exponent-add denorm, 32-wide: `float16_expadd_split_bf16_denorm` |
| Canonical two-byte float16 scales, exact, denorms can be flushed or host pre-biased | exponent-add flush, 32-wide: `float16_expadd_split_bf16_flush` |
| Host may re-encode the scale bytes, ~0.1% scale error is acceptable | packed 16-bit encoding, 32-wide: `float16_packed16_split_bf16` |
| Exact host-precomputed pair, and an extra two bytes per scale is affordable | host bf16 hi/lo planes, as used by `q4_0_hilo` |
| Three-term Q6_K scale preparation | rounded hardware split with the 32-lane `q6k_3term_prep`; the integer two-term path is slower |
| Two-term int8 scale application | `int8_scale_bf16_2term`; the integer path is slower |

The packed 16-bit format is the fastest two-term path: 28.2 cycles per 32
values, or about 7.05 per eight, and it handles denormal inputs without a
device branch, but it is not bit-exact. For canonical float16 bytes, the
exponent-add denorm path is exact for all finite values and costs 65.3 cycles
per 32; the flush path costs 43.3 per 32 but drops denormal scales. The
default probes are the 32-wide batches because 32 bf16 lanes is the native
512-bit width; `_x8` and `_x16` names remain for history and format-shaped
cases. Host-side pre-biasing alone does not reduce the device
cost of the old float16-to-f32 split. See `NOTES.md` for the full tables.

Detailed timings, the encoding definition, and the rejected alternatives are in
`NOTES.md`.
