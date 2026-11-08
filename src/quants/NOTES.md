# NPU kernel notes

Findings from optimizing the `src/quants` Q4_0, Q6_K, and Q8_0 kernels on
XDNA2 / AIE2P using Peano and IRON. The material is organized as reusable AIE2
techniques, shared kernel patterns, and then format-specific results. Current
status and future plans live in `TODO.md`; the quick summary lives in
`README.md`.

## Benchmarks and conventions

- Measure with `event0()` / `event1()` and the trace stack.
- Use **one outer event pair per kernel**. Per-chunk or per-tile events distort
  the measurement and cost 150-200 cycles by themselves.
- Keep the trace buffer large enough for long kernels. The harness now uses
  64 KiB; earlier 4 KiB and 16 KiB buffers overflowed on wider or longer
  kernels.
- Cycle counts are deterministic enough that single runs are useful, but repeat
  after code-layout changes.
- Report logical weights/cycle for one core:
  `weights/cycle = weights_rows * weights_cols / cycles`.
- Use the physical packed size, not the canonical format size, for bandwidth:
  - Q4_0: 4.5 -> 5.0 bits/weight (0.625 bytes/weight)
  - Q6_K: 6.5625 -> 8.75 bits/weight (1.09375 bytes/weight)
  - Q8_0: 8.5 bits/weight (1.0625 bytes/weight)
- Full-device estimate:
  `min(weights/cycle * 1.5 GHz * 32, 55 GB/s / bytes-per-weight)`.
  The first term is the compute cap, the second the packed-weight memory cap.
- 60 GB/s requires about 2.00 weights/core-cycle at the current 5.0-bit Q4_0
  layout, or 2.22 at canonical 4.5 bits.
- One AIE2 VLIW bundle is 16 bytes.

## Reusable AIE2 techniques

These are the API and code-generation findings worth carrying into future
quantized operations. They were checked against the mlir-aie docs, the
programming guide, `third_party/aie_api/include/aie_api/aie.hpp`, and the
`aie_api` examples.

### int4 vectors and conversions

- `aie::vector<int4, N>` / `vector<uint4, N>` are native on XDNA2 for
  `N = 32/64/128/256`; `N = 128` is one 512-bit register.
- `vector<uint8, 32>.cast_to<int4>()` is a reinterpret, not a conversion. It
  turns 32 bytes into 64 signed nibbles in byte order (low nibble first), so
  Q4_0 needs no nibble extraction or sign extension. `cast_to` only changes
  lane type/count; use it only when the packed bit order already matches the
  desired lane order.
- `to_float` has a `shift` argument. On XDNA2 the vector int-to-bf16 path uses
  magic bf16 constants, `0x4b01 - 128*shift` (plus `0x5301 - 128*shift` for
  16/32-bit inputs), so a compile-time shift costs no extra instruction.
  `to_float(v, s) = v / 2^s`, so a left shift by `k` is `to_float(v, -k)`.
  This can fold a power-of-two scale or replace an explicit `upshift`.
- `to_fixed<int4/uint4>` and `to_fixed_sign` exist on Gen2 for bf16/float
  vectors and use `conv_even`; they are useful for repacking or requantizing,
  not the hot dequant path.
- `aie::op_sign(v, bool)` gives a runtime signed/unsigned view without changing
  the vector type. It is useful for IQ4 indices or mixed signed/unsigned
  payloads without separate kernels.
- For sub-byte payloads, pack bit planes so lane order can be reconstructed
  with shifts, masks, and in-order `concat`. The abandoned native Q6_K layout
  used low-nibble/high-nibble and 2-bit-field planes this way; it worked, but
  the device bit-plane decode was still the throughput bottleneck.

### Bitwise and shift operations

- On XDNA2, `bit_and` / `bit_or` / `bit_xor` / `bit_not` for 8/16/32-bit are
  native `band` / `bor` / `bxor` / `bneg`. The "scalar operation on AIE" note
  in the Doxygen comments applies to AIE1, not Gen2.
- `bit_and(x, -1)` and `bit_or(x, 0)` fold at compile time.
- For 4-bit vectors, the API implements bitwise ops by unpacking to 8-bit,
  operating, and packing back. Mask the raw `uint8` payload first and cast to
  `int4` only for the final conversion.
- `upshift`, `downshift`, and `logical_downshift` are not raw shifts: each goes
  through an accumulator ups/srs pair, int4 shifts unpack/repack, and
  `max_shift` is 59. Both downshift variants also set
  `rounding_mode::floor` through `scoped_mode` per call. Avoid them in the
  innermost loop when a `to_float` shift can absorb the same scaling.
- Use `logical_downshift` for unsigned nibble extraction and `downshift` for
  arithmetic sign-extending shifts.

### Lane expansion and permutations

- `broadcast_vector<N>(v, index)` broadcasts one N-element chunk to a larger
  vector. When `N * type_bits >= 128` it is implemented as
  `v.extract<N>(index).grow_replicate<Elems>()`. Q8_0's
  `broadcast_vector<8, bfloat16, 64>` is therefore already the
  `grow_replicate` path.
- `vector::grow_replicate<ElemsOut>()` replicates the whole short vector. For
  a 128-bit source it lowers through `broadcast_elem_128`, then may expand to a
  broadcast plus a two-register concat.
- `aie::broadcast<T, Elems>({...})` accepts a small repeating array on Gen2
  when `type_bits * N <= 64` and N is a power of two. It packs the pattern into
  a scalar and broadcasts it; useful for repeating byte masks such as
  `{0x0F, 0xF0}` or 2-bit plane masks.
- `concat()` has a tuple overload, so `concat(interleave_zip(even, odd, 1))`
  joins the two halves of a deinterleave. `interleave_unzip` is the inverse.
- `filter_even` / `filter_odd(v, chunk_size)` return a half-size vector by
  selecting alternating chunks in one permute-style op. For 8-bit data this is
  byte granularity, so it fits separating ql/qh planes, not nibbles.
  `filter_mode`, `interleave_zip_mode`, and `interleave_unzip_mode` allow
  runtime-selected modes.
- `transpose(v, Row, Col)` is a single permute for small matrices. Push layout
  transforms into host repack when possible.
- `interleave_custom_static<S0..S31>` / `interleave_custom` implement arbitrary
  two-source lane selects and can replace chains of `shuffle_up`/`shuffle_down`.
- `shuffle_up_replicate`, `shuffle_down_fill`, `reverse`, and
  `select(v1, v2, mask)` are single ops. `mask<Elems>::from_uint64` /
  `from_uint32` builds compile-time masks.

### Codebook lookup tables

- `aie::parallel_lookup<T, aie::lut<4, Value>>` gives four parallel LUT accesses
  per instruction on Gen2 and is the most promising path for IQ4_NL/IQ4_XS
  codebook gathers.
- Index types must be at least 8-bit, so unpack 4-bit indices to `int8` first.
  `fetch` accepts at most 32 indices per call and returns at most 32 8/16-bit
  values (16 for 32-bit values), so a 64-lane tile needs two fetches.
- 8-bit values are stored as 16-bit LUT entries. The `aie::lut<4, OffsetType,
  SlopeType>` wrapper hides the required 4-way interleaved layout; see
  `third_party/aie_api/examples/lookup_table.cpp`. Construct the
  `parallel_lookup` object once, outside the loop. Out-of-range policy is
  `saturate` by default, `truncate` optional.
- Store the codebook in the LUT output type. For IQ4_NL, storing the small
  signed `kvalues_iq4nl` entries as bf16 avoids a post-lookup conversion at 2x
  LUT storage, which is negligible for 16 entries.

### mmul modes and emulation

- XDNA2 dense mmul modes: int8 x int4 has a native `4x16x16`; int8 x int8 has
  `4x8x8` and `8x8x8`; bf16 x bf16 has `8x8x4`, `4x8x8`, `4x8x4`, `8x1x8`,
  and `8x8x8` only through bfp16 emulation. There is no bf16 x int4 mode, so
  bf16-activation dequantization must still produce bf16 weights. `4x4x8` bf16
  is unavailable.
- `mmul<..., int8, int4>` consumes packed int4 B directly on XDNA2. This is
  only a drop-in win for a future W4A8 path with int8 activations.
- `mmul<8, 8, 8, bfloat16, bfp16ebs8>` lets bf16 A consume a block-float8 B
  directly. It is a relaxed-accuracy path, not an exact Q4_0/Q6_K path.
  `mx4`/`mx6`/`mx9` block vectors are AIE-MLv2 only.
- The current `4x8x8` bf16 mmul is itself emulated from
  `mul_elem_32`/`mac_elem_32`; the manual bf16 scale splits sit on top of that.
- `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1` switches `4x8x8`/`8x8x8` bf16
  mmul to bfp16 hardware. It was about 25% faster on one core (Q4_0 1880 ->
  1407, Q6_K 10339 -> 7755 cycles) but lost exactness (max relative error
  0.86% / 0.56%). Treat it as a separate relaxed mode.
- Element-wise float `aie::mul` is emulated via `__AIE_API_FP32_EMULATION__`
  and is much slower than the manual bf16 splits: Q4_0 unpack 707 -> 5540 and
  fused mmul 1880 -> 5889 cycles; Q6_K unpack 6676 -> 26068 cycles and the
  fused variant exceeded program memory. It is exact, but not a shortcut.
- The `AIE2_FP32_EMULATION_ACCURACY_FAST` / `_LOW` flags did not change those
  element-wise 32/64-lane timings. In the XDNA2 headers they only remap
  `mul_elem_16` and `mul_4x8_8x4`; the AIE API uses `mul_elem_32_conf` /
  `mul_elem_64_conf`. The unprefixed `AIE_FP32_...` names are the AIE-MLv2
  variants.
- A default-constructed `aie::mmul` is zero-qualified: the first `mac` behaves
  as `mul`. `aie::op_zero(acc, first_iteration)` can fold a runtime zero into
  the first `mac`, and `accum_ownership::ref` lets the accumulator alias an
  output buffer.
- `aie::op_neg(v, bool)` may fold into a subsequent mul/mmul. It does not fold
  `q - 32` centering or a `-dmin*min` bias.

### Loads, alignment, and record layout

- `load_v<Elems>` requires full vector alignment on XDNA2: 16 bytes for
  128-bit, 32 for 256-bit, 64 for 512-bit. Use
  `aie::vector_ldst_align_v<T, Elems>` / `vector_ldst_align_elems_v` in
  `static_assert`s.
- `load_unaligned_v<Elems>(p, aligned_elems)` is correct when the base is
  aligned to a known smaller boundary. `load_floor_bytes_v` / `load_floor_v`
  deliberately round the pointer down and are only safe when the truncated
  address is intended. A 64-byte load from a 32-byte-aligned address silently
  returns the wrong tile.
- `vector<int4, 128>` is one 512-bit register and valid for `load_v<128>`, so a
  naturally aligned 64-byte nibble tile can be loaded directly as int4. For
  the current 32-byte tiles, `load_v<64>(const int4*)` is the equivalent
  256-bit load.
- Put payloads first when a tile load needs a larger alignment than the record
  header; pad the record so the next record keeps that alignment. Payload-first
  layouts removed per-chunk padding and fixed the Q8_0/Q6_K alignment bugs.
- Host repack must stay plain bytes; host-compiled headers must not include
  `aie::vector`.
- Device-only layout views are useful for static assertions, but hot-path
  struct member access can cost more than vector-pointer access.

## Shared kernel patterns

### Two-column fusion

The best fused structure found so far:

- one `unpack_mmul<weights_rows, weights_cols>` implementation per format;
- process two independent column tiles at a time;
- share the input tile between both columns;
- use two mmul accumulators;
- fully unroll the inner tile loop;
- dequantize both columns, then issue both mmuls;
- store outputs once per column tile.

What helped:

- inlining unpack directly into the fused kernel instead of cursor objects;
- aligning tile payloads so `load_v` is valid;
- `aie::vector` pointers for input, output, and payload tiles;
- sharing scale/dequant helpers between the standalone unpack test and the
  fused kernel;
- precomputing exact bf16 scale terms in host repack when the precision fits;
- `broadcast_vector` for 8-to-64 scale expansion.

What did not help:

- partial unrolling (`unroll_count(4)`) or removing the unroll pragma;
- `noinline` on `next()` or the mmul helper;
- four columns processed concurrently;
- hoisting accumulators across records for Q4_0 (it already scales well);
- multiple accumulators per column;
- pointer increments instead of indexed addresses;
- `__restrict` on the fused kernel pointers;
- `-O3` / `-Os` instead of the default optimization;
- copying the input tile before both mmuls;
- dequantizing all tiles before issuing mmuls.

### Scale handling

- Match the number of bf16 terms to the exactness requirement:
  - fp16 `d` alone: two bf16 terms are exact (Q8_0).
  - fp16 `d * int8_scale`: up to 18 significand bits, so three bf16 terms are
    required (Q6_K).
- Canonical Q4_0/Q6_K/Q8_0 scales are assumed to be normal fp16 values.
  Host repack flushes subnormal scales to zero and emits one warning per
  repack call; zero scales need no special case because the two-term split
  cancels exactly. Q6_K and Q8_0 still convert fp16 on the device, so this is
  the only place that can catch the subnormal case.
- Q4_0 now has the host repack the exact bf16 `hi + lo` pair of each fp16
  scale. The record grows from 288 to 320 bytes, but the device avoids the
  fp16 decode and split, and the larger format is still faster at 55 GB/s.
- Expand the eight scale lanes to 64 once per block, not per tile.
  `broadcast_vector<8, bfloat16, 64>` is the preferred expansion.
- The 64-lane active scale cache is the best balance found. 32-lane scales
  plus per-tile expansion, 8-lane scales plus per-tile expansion, and
  reload/split per tile were all slower.
- Q6_K's `d` is hoisted to once per record; each chunk's 32 int8 scales are
  expanded to 32 lanes, multiplied by `d`, and split into hi/mid/lo once.
- Rejected scale approaches:
  - scale multiplication after the mmul: slower and outside exact tolerance;
  - two-term bf16 split for Q6_K: only ~3% faster and unsafe for canonical
    scales up to 127;
  - element-wise float `aie::mul`: exact but 3-8x slower;
  - Q6_K host-precomputed hi/mid/lo terms: 8239 cycles on one core but
    1.375 bytes/weight, worse than the exact int8 layout at 32 cores.
- A single-bf16 Q4_0 fast mode is about 11-12% faster on the old layout but
  has max abs error around 2.0. It belongs behind a named relaxed mode.

### Register pressure and scheduling

- Two-column fusion is faster despite spilling more. Do not give up its ILP to
  reduce spills.
- Per-column `noinline` helpers reduce pressure but lose too much ILP.
- Struct-based record views regressed the fused Q4_0 kernel by about 4.6%;
  vector-pointer access is faster in the hot path.
- The hot loop keeps two accumulators, two scale vectors, and two payload
  streams live. Four columns remains too much register pressure.

### Mmul shape

- The kernels use `aie::mmul<4, 8, 8>` with bf16 inputs and weights.
- `aie::mmul<4, 4, 8>` is not defined for bf16.
- `aie::mmul<4, 8, 4>` is supported but doubles mmul operations and requires
  output-column splitting/interleaving; it was not faster than the two-column
  path.
- The `vextbcst` instructions in the fused loop are inherent to the current
  `4x8x8` formulation.

### Profiling method

Replace the dequantizer with a constant to separate mmul/loop overhead from
decode/scale work. For example, Q6_K's mmul and loop alone are 3211 cycles
(31%); payload load plus int8-to-bf16 reaches 4297 (42%); the exact scale split
and three multiplies account for the remaining 6042 cycles (58%). Q8_0's
constant-dequantizer profile is about 3786 cycles (46% of total).

## Format-specific results

### Q4_0

Current exact layout: one 64-row x 8-column record is 320 bytes (5.0
bits/weight).

```text
 64 bytes  2 x 32-byte pre-split bf16 hi/lo scale blocks
256 bytes  8 x 32-byte int4 tile payloads
```

The host repack writes the exact bf16 `hi + lo` split of each fp16 scale.
Scales come first, payloads second; every 32-byte tile is naturally aligned.

Results, 1.5 GHz:

| shape | cycles | cycles/tile | weights/cycle | implied 32-core GB/s |
| --: | --: | --: | --: | --: |
| 64x64 | 1880 | 235.000 | 2.179 | 65.4 |
| 64x128 | 3740 | 233.750 | 2.190 | 65.7 |
| 64x256 | 7236 | 226.125 | 2.264 | 67.9 |

The old fp16-scale layout measured 2400 / 4764 / 9268 cycles, so host
pre-splitting is about 21-22% faster on one core. The implied GB/s uses the
current 5.0-bit packed size; the measured aggregate limit is 55-57 GB/s, so
the full device becomes memory-bound around 1.83-1.90 weights/core-cycle.
Long-K checks hold about 2.25 weights/cycle at 768 and 1536 rows.

Standalone exact unpack: 707 cycles (was 1499). The fast single-bf16 mode
measured 2119 cycles at 64 columns on the old layout but is not exact.

### Q6_K

Canonical Q6_K is 6.5625 bits/weight. The native 6-bit payload was compute
bound at 30160 cycles for 256x64 (0.543 weights/cycle); the device bit-plane
decode dominated.

The production layout expands every 6-bit weight to signed int8 (`q - 32`) in
the host repack. The per-16-row scale structure is unchanged, so the expansion
is lossless. Device decode is a 64-byte load plus `to_float<bfloat16>`.

Record layout: 256 rows x 8 columns, 2240 bytes (8.75 bits/weight).

```text
2048 bytes  4 x 512-byte int8 payload chunks (8 x 64-byte 8x8 tiles each)
 128 bytes  per-chunk int8 subgroup scales (4 chunks x 32 bytes)
  16 bytes  per-column fp16 d
  48 bytes  padding so every record is 64-byte aligned
```

Results, 1.5 GHz:

| shape | cycles | cycles/tile | weights/cycle | bytes/weight |
| --: | --: | --: | --: | --: |
| 256x64 native 6-bit baseline | 30160 | 942.500 | 0.543 | 0.828 |
| 256x128 native 6-bit baseline | 60276 | 941.812 | 0.544 | 0.828 |
| 256x64 int8 payload | 10254 | 320.438 | 1.598 | 1.094 |
| 256x128 int8 payload | 20425 | 319.141 | 1.604 | 1.094 |
| standalone unpack 256x64 | 6708 | 209.625 | 2.442 | - |
| 256x64 host scale terms | 8239 | 257.469 | 1.989 | 1.375 |
| Q8_0 256x64 reference | 8245 | 257.656 | 1.987 | 1.0625 |

The int8 payload is an exact 2.94x speedup on one core. At 55 GB/s it is
memory-bound at about 1.05 weights/core-cycle, roughly 1.93x the native 6-bit
format. Long-K checks hold 1.55-1.58 weights/cycle at 1024/1536 rows.

Rejected or alternate approaches:

- host-precomputing all three bf16 scale terms: 8239 cycles on one core but
  2816-byte records (1.375 bytes/weight), worse at the 32-core memory budget;
- applying the exact scale after the mmul: 14258 cycles and max abs error 5.0;
- dropping the third scale term: only ~3% faster and unsafe for canonical
  scales up to 127;
- reusing the accumulator directly in the split: neutral;
- fp32 scale plus emulated float `aie::mul`: exact but much slower;
- four column tiles: memory-bound, extra ILP does not help.

Q8_K / Q8_0 comparison: canonical Q8_K without `bsums` is 260 bytes per 256
weights (8.125 bits/weight, 1.016 bytes/weight), only about 7% smaller than
the exact int8 Q6_K layout. It replaces per-16 scales with one fp32 scale per
256 weights, so conversion is lossy. Q8_0 is already implemented, slightly
smaller (1.0625 bytes/weight), and has finer per-32 scales. If a lossy path is
acceptable, Q8_0 is the better target than Q8_K.

### Q8_0

Canonical Q8_0 is 8.5 bits/weight. The internal record is 256 rows x 8 columns,
2176 bytes:

```text
2048 bytes  int8 payloads (32 tiles of 64 bytes)
 128 bytes  per-block fp16 d (8 blocks x 8 columns)
```

Payloads come first so every 64-byte tile load is 64-byte aligned. Each tile
is one 8x8 row-major matrix, so decode is `load 64 signed bytes -> int8 vector
-> bf16 vector`, then `q * d`. An fp16 `d` needs only two bf16 terms; the
`hi + lo` split is exact and accumulates to one bf16 rounding.

Results, 1.5 GHz:

| shape | cycles | cycles/tile | weights/cycle | 32-core packed GB/s |
| --: | --: | --: | --: | --: |
| 256x64 | 8245 | 257.656 | 1.987 | 101.3 |
| 256x128 | 16453 | 257.078 | 1.992 | 101.6 |
| standalone unpack 256x64 | 5098 | 159.313 | 3.214 | - |

What helped: payload-first 64-byte alignment, signed int8 to bf16 without a
centering bias, two-term `d`, `broadcast_vector` for scale expansion (about
9%), a 256-row record instead of 128 (about 10%), and two-column fusion with
one accumulator per column. Four columns was about 65% slower from register
pressure.

Q8_0 is the fastest format per packed byte; Q4_0 is faster per weight. Its
constant-dequantizer profile is about 3786 cycles of mmul/loop overhead per
256x64 (46%), so it is limited more by mmul/input scheduling than payload
decode. The standalone unpack test is limited to 64 columns by the 64 KiB
output buffer; the fused kernel works at 64/128 columns. 256 columns need about
68 KiB of packed weights and do not fit the current single-tile harness.

## Testing and exactness

- Keep a standalone exact unpack test as format/layout conformance. It catches
  scale, packing, and dequant errors that an end-to-end mmul tolerance test can
  hide.
- Keep the mmul test focused on end-to-end fused correctness.
- Share scale/dequant helpers between the reference unpack and fused mmul so
  the tests and production path cannot drift.
- Q6_K `mmul.py --cols 256` exceeds the single-tile harness memory (about
  70 KiB of int8-packed weights plus inputs/outputs); use 64 or 128 columns
  until the stream is split.
- The Q6_K generator uses scales of 1-32, but canonical scales can reach 127.
  A two-term bf16 split can still pass the end-to-end test because final bf16
  rounding hides the residual. Add a direct split-residual test if the scale
  representation changes.
- Code-size measurements from earlier kernels are stale after the host-scale
  and int8-payload changes; re-measure only if code size matters.

## Target model context

`gemma-4-E2B_q4_0-it.gguf` is the first end-to-end target: 35 dense blocks,
about 1.605 GB/token of active NPU weights with the current repacks, or roughly
34.3 tok/s at 55 GB/s. The traffic breakdown and bring-up checklist live in
`TODO.md`; the summary lives in `README.md`.
