# Bench Experiments

Small standalone experiments that answer one question at a time about the AMD
XDNA2 NPU. This is a staging area for understanding and optimization, not a
test suite:

- nothing under `src/` imports these files;
- nothing here is gated by CI;
- a proven helper moves to `src/quants/common/`;
- a confirmed finding is recorded in `src/quants/NOTES.md`.

Each experiment is flat and prefixed by its name: one Python driver, one kernel
shim, and (when useful) a short `.md` note. The driver reports cycles, a
correctness number against a NumPy reference, and the compiler diagnostics for
anything that does not build.

## Running

Activate the MLIR-AIE environment, then run from this directory:

```sh
source ~/Projects/mlir-aie/ironenv/bin/activate
python fp16_ops.py
```

Useful flags:

```sh
python fp16_ops.py --list                 # probe names
python fp16_ops.py --compile-only         # no device or tracing
python fp16_ops.py --probe f16_mul        # one probe
python fp16_ops.py --seed 7
```

`--compile-only` uses the same Peano invocation as the JIT path, so it still
needs the AIE environment but not the NPU device.

The trace ring occasionally drops the closing `event1()` record. The driver
retries that run up to three times; the kernel result is deterministic, so a
retry only replaces the timing sample.

## Experiments

| file | question | status |
| :-- | :-- | :-- |
| `fp16_ops.py` | Which float16 operations do the compiler and XDNA2 actually support? | see result below |

### `fp16_ops`

XDNA2 is arch 21 (`aie2p`). The AIE API headers define `float16` for arch 22
(`aie2ps`) only; on arch 21 it is a placeholder type. This experiment checks
that on the device and measures the bf16 fallbacks the quant kernels rely on.

Result (2026-11-10, Ryzen AI 9 HX 475):

- no `aie::vector<float16>` and no fp16 load, store, convert, arithmetic, or
  mmul;
- `sizeof(float16)` is 1 on XDNA2, so it is not usable as compressed storage;
- an fp16 scale is decoded from raw bits to f32 (`load_fp16x8`), then split into
  a bf16 hi/lo pair (`split_bf16_hi_lo`);
- the bf16 and emulated-f32 scale paths compile and are the usable baselines.

| probe | compile | cycles | cyc/iter | max err |
| :-- | :--: | --: | --: | --: |
| `u16_load_store` | yes | 353 | 11.0 | 0 |
| `bf16_load_store` | yes | 353 | 11.0 | 0 |
| `fp16_bits_f32` | yes | 1195 | 37.3 | 0 |
| `fp16_shift_f32` | yes | 1064 | 33.2 | 0 |
| `fp16_bits_split_bf16` | yes | 1515 | 47.3 | 0 |
| `fp16_shift_split_bf16` | yes | 1384 | 43.2 | 0 |
| `fp16_direct_split_bf16` | yes | 1964 | 61.4 | 0 |
| `int8_scale_f32` | yes | 3496 | 109.2 | 0 |
| `int8_scale_bf16` | yes | 866 | 27.1 | 0 |
| `int8_scale_bf16_2term` | yes | 964 | 30.1 | 0 |
| `int8_scale_bf16_accum` | yes | 964 | 30.1 | 0 |
| `int8_scale_int_mul` | yes | 642 | 20.1 | 0 |
| `int8_scale_int_round` | yes | 1355 | 42.3 | 0 |
| `int8_scale_int_split` | yes | 1649 | 51.5 | 0 |
| `q6k_3term_apply` | yes | 1063 | 33.2 | 0 |
| `q6k_3term_full` | yes | 3885 | 121.4 | 0 |
| `q6k_int_apply` | yes | 2054 | 64.2 | 0 |

The last group tests the alternate decoder: represent each fp16 scale as an
int16 mantissa and a bf16 power-of-two (`fp16 = M * 2^E`). `q * M` is exact in
int32, and rounding to bf16 commutes with the power-of-two factor, so
`int8_scale_int_split` is bit-exact against `int8_scale_bf16_2term`. It is
slower, though: the int32 -> bf16 conversion costs about 22 cycles per 64
weights, more than the second bf16 multiply it replaces. The breakdown is in
`int8_scale_int_mul` (20.1), `int8_scale_int_round` (+22.2), and the final
bf16 scale (+9.2).

The Q6_K probes apply an 18-bit effective scale (`d * int8` subgroup scale).
The current path needs three bf16 terms; the integer path needs two int16
terms. Apply-only is 33.2 vs 64.2 cycles per 64 weights, so the integer path is
slower even with free host-precomputed terms. The three-term preparation costs
about 88 cycles per 8-column group (121.4 full vs 33.2 apply) and is shared by
two tiles in production, so the integer path would need preparation under about
26 cycles per group to break even. `q6k_int_full` currently fails to compile:
the device-side 32 -> 16-bit narrowing hits a Peano legalizer crash on `G_OR`.

`aie::accumulate` can express the two-term bf16 scale only when the scale is
uniform across all lanes. It lowers to the same `vmul.f` + `vmac.f` pair as
`aie::mul` + `aie::mac` and measures identically (30.1). It cannot apply the
per-column scales used by Q4_0/Q6_K/Q8_0 because its coefficients are
scalar-per-data-vector, not per-lane.

`aie::sliding_mul_ch` is structurally the right shape for per-channel scales
(`Channels = 8`, `Outputs = rows`, one coefficient per column), but it is not
usable for the current kernels: XDNA2 only implements it for 8-bit and 16-bit
integers (acc32/acc64), and a bf16 instantiation fails at
`sliding_mul_ch_bits_impl<..., 16, 16, bfloat16, bfloat16>`. It also requires
`CoeffStep = 1`, so the same scale cannot be reused across `Points`; `Points = 1`
wastes the native 4/8-point instruction, while `Points = 4/8` requires
duplicating the eight scales into the coefficient vector. It may be useful for
a future integer-scale path, not for bf16 scale application.

For the scale helpers themselves: `fp16_bits_f32` (37.3) is `load_fp16x8`,
while `fp16_bits_split_bf16` (47.3) adds `split_bf16_hi_lo`. The split costs
about 10 cycles per 8 scales and is four vector ops plus a second store;
`load_fp16x8` is the larger decode cost. The shift-based decode
(`fp16_shift_f32`) is 33.2 cycles, about 11% faster than the mask/rebias form,
and `fp16_shift_split_bf16` is 43.2. Both are exact for normal fp16 and flush
zero/subnormal like the existing decode.

The direct fp16 -> 2xbf16 route (`fp16_direct_split_bf16`) is exact but slower:
61.4 cycles. The blocker is normalizing the leftover 3-bit fraction. A
compare/select version came in at 69.3 cycles; the arithmetic version converts
the 3-bit `r` to bf16 and multiplies by the bf16 power-of-two, which costs more
than the f32 round-trip it avoids. A hardware `parallel_lookup` table could
remove the normalization, but for eight scales per block the 4-way interleaved
LUT setup and minimum 16-lane fetch are unlikely to pay off.

The nine fp16 probes that remain are unsupported operations; `q6k_int_full` is
the compiler bug above. The driver prints the first compiler error for each.
The durable write-up lives in `src/quants/NOTES.md` under "float16 on XDNA2".
