# Bench Notes

Detailed results and history for the standalone float16 experiments. The
current map and recommendations are in [`README.md`](README.md).

Measurements are from a Ryzen AI 9 HX 475 (XDNA2, `aie2p`, arch 21), one
core, 32 iterations, unless noted otherwise. `vals/iter` is the number of
float16 values or int8 weights processed per iteration; probes with different
values per iteration are not directly comparable by `cyc/iter`. The two-term
table was re-measured on 2026-11-18 after a kernel reboot with unlocked clocks;
use per-value ratios from the same session rather than absolute cycles across
boots.

## float16_support

XDNA2 is arch 21 (`aie2p`). The AIE API headers define `float16` for arch 22
(`aie2ps`) only; on arch 21 it is a placeholder type.

Result (2026-11-10):

- no `aie::vector<float16>` and no float16 load, store, convert, arithmetic, or
  mmul;
- `sizeof(float16)` is 1 on XDNA2, so it is not usable as compressed storage;
- a float16 scale must be decoded from raw bits to f32, then split into a bf16
  hi/lo pair;
- the bf16 and emulated-f32 scale paths compile and are the usable baselines.

| probe | compile | cycles | vals/iter | cyc/iter | max err |
| :-- | :--: | --: | --: | --: | --: |
| `u16_load_store` | yes | 353 | 64 | 11.0 | 0 |
| `bf16_load_store` | yes | 353 | 64 | 11.0 | 0 |
| `float16_scalar_add` | no | - | - | - | - |
| `float16_vector_decl` | no | - | - | - | - |
| `float16_vector_load_store` | no | - | - | - | - |
| `float16_size` | no | - | - | - | - |
| `float16_int8_convert` | no | - | - | - | - |
| `float16_acc_convert` | no | - | - | - | - |
| `float16_add` | no | - | - | - | - |
| `float16_mul` | no | - | - | - | - |
| `float16_mmul` | no | - | - | - | - |

The driver prints the first compiler error for each unsupported probe.

## float16_convert

Raw float16 bits to f32.

| probe | cycles | vals/iter | cyc/iter | max err |
| :-- | --: | --: | --: | --: |
| `float16_bits_f32_x8` | 1195 | 8 | 37.3 | 0 |
| `float16_shift_f32_x8` | 1064 | 8 | 33.2 | 0 |

`float16_bits_f32_x8` is `load_float16x8`: extract the sign, exponent, and
mantissa,
re-bias the exponent, and select zero for zero/subnormal inputs. The
shift-based decode (`float16_shift_f32_x8`) is about 11% faster: zero-extend the
pattern, `(u << 17) >> 4` to form `(h & 0x7fff) << 13`, OR in the sign, and add
`0x38000000` for the f32 bias. It is exact for normal float16 and flushes
zero/subnormal inputs.

Host pre-biasing by `2^10` and decoding with a `+102` bias removes the
subnormal case but does not reduce the device work: zero still needs an
exponent select, and the f32 field placement still needs two shifts/masks plus
the bias add. `cast_to<float>` and the `accum<accfloat>` constructor are
bit-level moves, so building directly into an accfloat accumulator does not
remove instructions.

## float16_scale

### Two-term splits

| probe | cycles | vals/iter | cyc/iter | max err |
| :-- | --: | --: | --: | --: |
| `float16_bits_split_bf16_x8` | 1515 | 8 | 47.3 | 0 |
| `float16_shift_split_bf16_x8` | 1384 | 8 | 43.2 | 0 |
| `float16_biased_shift_split_bf16_x8` | 1384 | 8 | 43.2 | 0 |
| `float16_direct_split_bf16_x8` | 1964 | 8 | 61.4 | 0 |
| `float16_packed16_split_bf16_x16` | 904 | 16 | 28.2 | 9.7e-4 rel |
| `float16_expadd_split_bf16_flush_x16` | 1385 | 16 | 43.3 | 5.9e-5 abs |
| `float16_expadd_split_bf16_denorm_x16` | 1929 | 16 | 60.3 | 0 |
| `float16_bits_split_bf16` | 1740 | 32 | 54.4 | 0 |
| `float16_shift_split_bf16` | 1450 | 32 | 45.3 | 0 |
| `float16_biased_shift_split_bf16` | 1450 | 32 | 45.3 | 0 |
| `float16_direct_split_bf16` | 2252 | 32 | 70.4 | 0 |
| `float16_packed16_split_bf16` | 904 | 32 | 28.2 | 9.8e-4 rel |
| `float16_expadd_split_bf16_flush` | 1387 | 32 | 43.3 | 6.0e-5 abs |
| `float16_expadd_split_bf16_denorm` | 2089 | 32 | 65.3 | 0 |

The unsuffixed rows are the default 32-wide probes: one native 512-bit bf16
vector for each term. The `_x8` and `_x16` rows are kept as history and
format-shaped cases until we confirm no consumer needs those widths.

`float16_bits_split_bf16_x8` is `load_float16x8` followed by
`split_bf16_hi_lo`; `float16_shift_split_bf16_x8` uses the faster shift
decode. The split costs about 10 cycles per eight scales and is four vector ops plus a second store; the
decode dominates.

The direct float16 to two-bf16 route (`float16_direct_split_bf16_x8`) is
exact but slower: 61.4 cycles. The blocker is normalizing the leftover three-bit
fraction. A compare/select version came in at 69.3 cycles; the arithmetic
version converts the three-bit `r` to bf16 and multiplies by the bf16
power-of-two, which costs more than the f32 round-trip it avoids. A hardware
`parallel_lookup` table could remove the normalization, but the four-way
interleaved LUT setup and minimum 16-lane fetch did not look worthwhile; a
byte-split LUT prototype with a separate low-byte conversion and bf16 multiply
was also slower than the bit-op path and was not kept.

#### Host-packed 16-bit encoding

`float16_packed16_split_bf16` keeps the two bf16 terms in the same two bytes as
the canonical float16 scale. The host normalizes the value to a bf16 exponent
and a nine-bit significand fraction, then packs:

```text
bit 15      sign
bits 14:9   bf16 exponent minus 102 (0 marks zero)
bits 8:2    top seven fraction bits (the hi term)
bits 1:0    residual code {0, 1.5, 2, 3} in 1/512 units (the lo term)
```

The two-bit code already carries the residual's exponent delta and mantissa
bit, so the device produces hi and lo with integer shifts, masks, adds, and two
selects. There is no float16 decode, float conversion, or bf16 multiply.
`hi + lo` reproduces the host-packed nine-bit fraction with at most 0.5 ulp of
error, about 0.1% relative; zero is exact. Normal and subnormal inputs share
the same path because the host normalizes both; most denormal inputs are exact,
the rest are within the same 0.1% bound.

`float16_packed16_split_bf16_x16` is 28.2 cycles per 16, or about 14.1 per
eight. The default 32-wide probe is still 28.2 cycles, or about 7.05 per eight,
because one 512-bit code vector now feeds both native bf16 outputs. That is the
fastest two-term result at any width. The cost is one dropped mantissa bit (nine-bit fraction
instead of the full eleven-bit float16 significand), and `inf`/`nan` codes are
not represented, so host repack must map them.

#### Exponent-add split

`float16_expadd_split_bf16_flush` and `..._denorm` (with `_x16` variants)
start from canonical float16 bytes. `filter_even` / `filter_odd` split the low and high bytes, the
low byte goes through the native `to_float<bfloat16>` conversion, and
`(E - 25) << 7` is added to its exponent field. The high byte supplies `hi`
with ordinary bit ops, so there is no table, no bf16 multiply, and no 32-bit
unpack.

The flush variant maps every `E == 0` value to zero: the `_x16` row is 43.3
cycles per 16, or 21.6 per eight; the default 32-wide row is 43.3 cycles, or
10.8 per eight. The denorm
variant builds `hi = top2 * 2^-16` for denorms and uses a `-24` exponent delta
for `lo`; it is bit-exact for finite float16 and costs 60.3 cycles per 16
(30.2 per eight), or 65.3 cycles per 32 (16.3 per eight). Supporting denorms
therefore costs about 17 cycles per 16, or 22 cycles per 32, in this path.

When host re-encoding is allowed, the packed 16-bit format already covers
denorms at its normal 0.1% bound and is faster. The exponent-add path is for
canonical float16 bytes with exact denorms.

### Two-term application to int8

| probe | cycles | vals/iter | cyc/iter | max err |
| :-- | --: | --: | --: | --: |
| `int8_scale_f32` | 3496 | 64 | 109.2 | 0 |
| `int8_scale_bf16` | 866 | 64 | 27.1 | 0 |
| `int8_scale_bf16_2term` | 964 | 64 | 30.1 | 0 |
| `int8_scale_bf16_accum` | 964 | 64 | 30.1 | 0 |
| `int8_scale_int_mul` | 642 | 64 | 20.1 | 0 |
| `int8_scale_int_round` | 1355 | 64 | 42.3 | 0 |
| `int8_scale_int_split` | 1649 | 64 | 51.5 | 0 |

The last group tests the alternate decoder: represent each float16 scale as
an int16 mantissa and a bf16 power-of-two (`float16 = M * 2^E`). `q * M` is
exact in int32, and rounding to bf16 commutes with the power-of-two factor, so
`int8_scale_int_split` is bit-exact against `int8_scale_bf16_2term`. It is
slower: the int32 -> bf16 conversion costs about 22 cycles per 64 weights, more
than the second bf16 multiply it replaces. The breakdown is
`int8_scale_int_mul` (20.1), `int8_scale_int_round` (+22.2), and the final bf16
scale (+9.2).

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
`CoeffStep = 1`, so the same scale cannot be reused across `Points`; `Points =
1` wastes the native 4/8-point instruction, while `Points = 4/8` requires
duplicating the eight scales into the coefficient vector.

### Three-term / Q6_K scales

| probe | cycles | vals/iter | cyc/iter | max err |
| :-- | --: | --: | --: | --: |
| `q6k_3term_prep` | 3434 | 32 | 107.3 | 0 |
| `q6k_3term_prep_x8` | 3499 | 8 | 109.3 | 0 |
| `q6k_3term_apply_x8` | 1063 | 64 | 33.2 | 0 |
| `q6k_3term_full_x8` | 3885 | 64 | 121.4 | 0 |
| `q6k_int_apply` | 2054 | 64 | 64.2 | 0 |
| `q6k_int_full` | no | - | - | - |

The Q6_K probes apply an 18-bit effective scale (`d * int8` subgroup scale).
The current path needs three bf16 terms; the integer path needs two int16
terms. Apply-only is 33.2 vs 64.2 cycles per 64 weights, so the integer path is
slower even with free host-precomputed terms. Three-term preparation costs
about 88 cycles per 8-column group (121.4 full vs 33.2 apply) and is shared by
two tiles in production, so the integer path would need preparation under
about 26 cycles per group to break even.

`q6k_3term_prep` is the 32-lane preparation demo: it broadcasts eight `d`
values across four subgroups, multiplies by 32 int8 subgroup scales, and splits
the 32 products into three bf16 terms in one native-width batch. It measures
107.3 cycles for 32 terms, or about 26.8 cycles per eight-column group.
`q6k_3term_prep_x8` measures 109.3 cycles for one eight-column group, so the
32-wide batch is roughly four times cheaper per group and is close to the
26-cycle preparation budget where the integer two-term path would break even.
The application probes keep the `_x8` suffix because they still prepare and
apply one eight-column group at a time; a full 4-subgroup apply demo remains.

`q6k_int_full` currently fails to compile: the device-side 32 -> 16-bit
narrowing hits a Peano legalizer crash on `G_OR`.

The Q6_K kernel-level experiments kept the paired tile decode, the exact bf16
`d_hi/d_lo` scale product, the hoisted `d`, the `broadcast_vector` scale
expansion, and the redundant-mask removal: 256x64 fused mmul 24453 -> 18967
cycles and standalone unpack 19922 -> 16434. Two-column fusion reached 18803
cycles (~1% better) but grew program memory from 2128 to 3392 bytes and the
stack frame from 576 to 2816 bytes, so it was rejected. For native Q6_K,
per-tile scale expansion was also not kept; the paired decode already retains
the 32-lane scale cache and expands once per pair.

The Q6_K_int8 kernel picked up the same exact bf16 scale product plus per-tile
scale expansion: fused mmul 10191 -> 9055 cycles, standalone unpack
6708 -> 5364, program memory 5008 -> 4752 bytes, and stack frame
2880 -> 2624 bytes. Column-sequential ordering (+1.0%) and forcing the input
tile into a register (+1.1%) were slower and were not kept.

The bitwise three-term scale split was also tried. It is exact and produces
bit-identical bf16 weights, but it is slower than the hardware conversion
round-trip (measured before the `broadcast_vector` change): Q6_K unpack
16594 -> 16914 with `filter_odd` and 17650 with shift+pack; Q6_K_int8 unpack
5524 -> 5972 and 6708 respectively. The existing rounded split was kept.

## Vector width

AIE2P native vector registers are 512 bits:

| element type | native lanes |
| :-- | --: |
| `int8` / `uint8` | 64 |
| `int16` / `uint16` / `bfloat16` / float16 bits | 32 |
| `float` / `int32` | 16 |
| `int4` | 128 |

The `_x8` probes exercise a quarter of a native bf16 vector and the `_x16`
packed16/expadd probes exercise half. The default 32-wide scale probes fill one
native register for the input codes and one for each bf16 output; they confirm
the ordering: packed16 remains fastest per value, the exponent-add
flush path is next, and the f32-based and direct splits remain slower even at
32-wide. The f32 conversion probes naturally use 16 values on the output side.
Eight-value cases remain relevant where the packed format itself is eight-wide,
such as a Q4_0 block's eight column scales; the fused Q4_0 path can batch two
records to reach 32.

## Durable write-up

The production-facing findings live in `src/quants/NOTES.md` under "float16 on
XDNA2". These bench notes hold the experiment history and the alternatives
that were rejected.
