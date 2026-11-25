> Role: reference. Current as of 2026-11-24. History: `git log -- src/quants/DESIGN.md`.

# Quant kernel design

Reference for `src/quants`: the AIE2 techniques, the shared kernel structure,
the per-format layouts, and the contracts the rest of the repository depends
on. Performance tables are in [`../../docs/measurements.md`](../../docs/measurements.md);
do not retry anything in [`../../docs/rejected.md`](../../docs/rejected.md)
without reading why it lost. Open work is in `TODO.md`.

## Layers

Three layers are tightly coupled:

1. **host repack** from canonical GGUF bytes into blocked transport tiles;
2. **device unpack** of those tiles to bf16;
3. **fused dequantize + matrix multiply** against bf16 activations.

All three are header-only. Format names are canonical GGUF types; suffixes
denote packed variants (`q4_0_packed16`, `q4_0_hilo`, `q6_k_int8`).

```text
<format>/layout.h       packed transport-tile constants and pointer helpers
<format>/repack.h       host-side canonical-to-tile conversion
<format>/unpack.aie.h   device-side dequantization
<format>/mmul.aie.h     device-side fused dequantization and matrix multiply
common/                 shared tile shape, scale, repack, and mmul helpers
tests/                  host repack shims, NPU harnesses, NumPy references
```

## Conventions

- State `weights_rows` (K), `weights_cols` (N), `inputs_vecs` (M = 4), and
  `outputs_len`; never M/N/K in prose. The full naming table is under
  "Contracts" below.
- Measure with `event0()` / `event1()` and the trace stack. Use **one outer
  event pair per kernel**: per-chunk or per-tile events distort the measurement
  and cost 150-200 cycles by themselves. Keep the trace buffer at 64 KiB.
- Cycle counts are deterministic enough that a single run is useful, but repeat
  after code-layout changes.
- Report logical `weights/cycle` for one core:
  `weights_rows * weights_cols / cycles`.
- Use the physical packed size, not the canonical format size, for bandwidth.
- `mmul.py` reports each core ELF's `.text` as a share of the 16 KiB program
  memory and its largest per-function stack frame. That is a single-frame
  maximum, not a transitive call-chain total, and a lower bound when a reachable
  function has no stack-size information.
- Set `QUANTS_KERNEL_FLAGS` to append Peano flags. The harness always enables
  `-Rpass-missed=loop-unroll` and prints captured warnings as `compiler: ...`.
  Add `-Werror=pass-failed` to make failed `#pragma unroll` transformations
  fatal; without it a failed `unroll(full)` silently stays rolled.

## Loop optimization and Peano idioms

This playbook applies to every format. It comes from a standalone copy
microbenchmark and a fixed-shape Q4_0 experiment; the measured tables are in
`docs/measurements.md` under "Loop and compiler experiments".

### General playbook

1. **Classify the bottleneck before changing code.** Address arithmetic, scale
   decode, the mmul recurrence, code size and memory traffic need different
   fixes. Measure and inspect the disassembly first.
2. **Write pointer-increment loops** where addressing is the bottleneck:
   `ptr += stride` lets Peano use post-increment addressing and often discovers
   a single-register load/store rotation.
3. **Sweep unroll pragmas on the actual loop:** `unroll(disable)`,
   `unroll(enable)`, `unroll_count(N)`, `unroll(full)`. Check cycles, `.text`
   and warnings. Treat "loop not unrolled" as no change, not success.
4. **Keep two-column (or two-channel) fusion** where it exists: it shares input
   loads and exposes independent accumulators. Do not split it without
   measuring.
5. **Use backend flags only after the disassembly shows a problem.** The common
   `aie-*` unroll, pipeliner and register flags were neutral on copy kernels.
6. **Prefer host precomputation when the repack contract allows it.** Moving
   fp16 decode, hi/lo splitting, or 6-bit scale/min expansion into host repack
   removes device work. This is a format/bytes trade, not a scheduling trick.
7. **Hand-write assembly only when the compiler cannot express the schedule.**
   The copy benchmark shows the compiler matching a hand-written 8-deep load
   pipeline when the source shape is right.

### Pointer increments: when they do not transfer

Pointer increments are not universally faster. In a q4_0 experiment that kept
the exact two-column fusion and changed only which addresses advance, advancing
the input or record pointers regressed 2-4%: the loop is recurrence-bound and
the pointer form adds loop-carried state and live ranges that disturb the
schedule. Advance the payload pointer only.

- Use pointer increments when address arithmetic is the bottleneck, as in the
  copy loop.
- For a dequant+mmul recurrence, measure both forms. Indexed code can be better
  even when the same rule wins elsewhere.
- In `src/program`, pointer increments were applied only to address-bound copy
  paths (`program_zero`, carry copy), where the disassembly confirms
  post-increment addressing.

### Bottleneck-to-action guide

| symptom | first action | notes |
| :-- | :-- | :-- |
| Lots of `movs`/`add` index arithmetic | pointer increments | biggest copy-loop win |
| `vldb` followed by five nops | pointer rotation or more unrolling | load result latency is 7 |
| Scale decode dominates; repack can change | host-precompute or hi/lo layout | Q4_0_hilo vs Q4_0 is the model |
| Scale decode dominates; repack fixed | accept it or use a named relaxed mode | device fp16 decode is near best |
| mmul accumulator chain serializes | two-column fusion, static unroll, two accumulators | more accumulators often spill worse |
| Compiler says "loop not unrolled" | manual unroll or a smaller body | `unroll(full)` can silently fail |
| Code size is the constraint | roll outer loops, partial unroll | full unroll is a latency/code trade |
| Memory-bound (bytes/weight) | change repack precision or host layout | scheduling will not help |
| Relaxed accuracy allowed | single-bf16 scale or bfp16-emulated mmul | must be a named variant |

For Q6_K/Q6_K_int8/Q8_0 the same actions apply to the `payloads`, `scales` and
`d` pointers. For IQ4_NL/IQ4_XS keep the codebook in lookup memory and use
pointer increments for index streams. For Q4_K/Q5_K host-expand the 6-bit
scale/min metadata first.

### Compiler diagnostics

- `llc -pass-remarks-output=...` on the kernel IR shows pipeliner II/NS, ZOL
  conversion, spills and bundle counts.
- `llvm-objdump -d --no-show-raw-insn` compares the actual bundle/issue
  schedule before and after a change.
- `-Rpass-analysis=aie-hardware-loops` and pipeliner remarks help, but many
  diagnostics come from `llc`, not the clang driver.
- The q4_0 recurrence cannot be scheduled by the pre-RA pipeliner and the
  post-RA pipeliner reports `II=63, NS=1`; static unrolling is the only
  effective ILP tool there. Do not generalise "the compiler is always good
  enough" from the copy result, or "hand asm is always faster" either.

### ZOL and hand-written assembly facts

- A ZOL is `add.nc lc, rN, #adj`, `movxm ls, #.Lbody`, `movxm le, #.Lend`.
- **`le` must name a full 16-byte bundle.** A 14-byte final bundle turns the ZOL
  into a fall-through loop that executes once.
- Loop setup must be at least ~112 bytes (7 bundles) from `le`; a shorter body
  does not loop. Pad before `.Lbody` or move the prologue after the `ls`/`le`
  setup, as the compiler does.
- The `event #0` / `event #1` mnemonics are rejected by the AIE2P asm parser
  even though `llc` prints them. Raw encodings: `event #0 = .4byte 0x10100018`,
  `event #1 = .4byte 0x12100018`.
- `ExternalFunction` compiles one C/C++ source; a separate `.s` file is not
  linked. Embed assembly in a global `__asm__("...")` block or link a prebuilt
  object explicitly.

### Machine model

Upstream file paths and pinned versions are in
[`../../docs/upstream.md`](../../docs/upstream.md). Key latencies: `vldb` 7,
`vlda` 7, `vmul`/`vmac` to accumulator 5, `vadd`/`vsub`/`vextbcst`/`vshuffle`
2, `vconv` 2, `vups` 3, `vsrs` 4, `vst` completion 5, ALU 1.

## Reusable AIE2 techniques

Checked against the mlir-aie docs, the programming guide, the installed
`aie_api` headers, and upstream examples.

### int4 vectors and conversions

- `aie::vector<int4, N>` / `vector<uint4, N>` are native for
  `N = 32/64/128/256`; `N = 128` is one 512-bit register.
- `vector<uint8, 32>.cast_to<int4>()` is a reinterpret, not a conversion: 32
  bytes become 64 signed nibbles in byte order (low nibble first), so Q4_0 needs
  no extraction or sign extension. Use it only when the packed bit order already
  matches lane order.
- `to_float` has a `shift` argument implemented with magic bf16 constants
  (`0x4b01 - 128*shift`, plus `0x5301 - 128*shift` for 16/32-bit inputs), so a
  compile-time shift costs no extra instruction. `to_float(v, s) = v / 2^s`, so
  a left shift by `k` is `to_float(v, -k)`. This can fold a power-of-two scale.
- `to_fixed<int4/uint4>` and `to_fixed_sign` exist for bf16/float vectors and
  use `conv_even`; useful for repacking, not the hot dequant path.
- `aie::op_sign(v, bool)` gives a runtime signed/unsigned view without changing
  the type.
- For sub-byte payloads, pack bit planes so lane order reconstructs with shifts,
  masks and in-order `concat`. Native Q6_K uses a low-nibble plane plus a
  two-bit high plane.

### float16 on XDNA2

- XDNA2 is AIE2P / arch 21. The AIE API defines `float16` only for arch 22
  (AIE-MLv2); on XDNA2 it is a placeholder. The capability probe in
  `tests/bench/float16_support.*` confirms `sizeof(float16) == 1`, no
  `aie::vector<float16, N>`, and no fp16 load/store, conversion, arithmetic or
  mmul.
- Treat fp16 as a storage and scale format only. Canonical fp16 `d` values are
  decoded on device with the exponent-add split (Q4_0, Q8_0), pre-split on the
  host (Q6_K, Q4_0_hilo), or replaced by a named host encoding
  (`q4_0_packed16`). Do not build a kernel around fp16 arithmetic.
- Shift-based decode when it is hot: zero-extend, `(u << 17) >> 4` to form
  `(h & 0x7fff) << 13`, OR the sign, add `0x38000000` for the f32 bias. Exact
  for normal fp16; flushes zero/subnormal.
- Host re-biasing alone does not reduce device work: the field placement and
  zero select remain. `cast_to<float>` and the accfloat constructor are
  bit-level moves, so building directly in an accfloat accumulator saves
  nothing.
- The host-packed 16-bit encoding keeps two bf16 terms in the two canonical
  bytes: `sign | (exp8 - 102) << 9 | top7 << 2 | residual2`, residual
  `{0, 1.5, 2, 3}` in 1/512 units. `hi + lo` reproduces a 9-bit fraction within
  0.5 ulp; zero is exact; normals and subnormals share one path. Cost: one
  dropped mantissa bit and no inf/nan.
- Exponent-add split for canonical bytes: `filter_even`/`filter_odd` split the
  bytes; the low byte goes through `to_float<bfloat16>` and `(E - 25) << 7` is
  added to its exponent; the high byte supplies `hi` with bit ops. A denorm
  variant uses `hi = top2 * 2^-16` and a `-24` delta for `lo`.
- A scalar packed immediately before an aligned tile load is unsafe: a probe
  loading at `in + 2` silently read from `in` because `load_v<64>` requires
  64-byte alignment. Reserve aligned blocks for scales and payload tiles.

### Bitwise and shift operations

- On XDNA2 `bit_and`/`bit_or`/`bit_xor`/`bit_not` for 8/16/32-bit are native
  `band`/`bor`/`bxor`/`bneg`. The "scalar operation on AIE" note in the Doxygen
  comments applies to AIE1.
- `bit_and(x, -1)` and `bit_or(x, 0)` fold at compile time.
- For 4-bit vectors, bitwise ops unpack to 8-bit and repack. Mask the raw
  `uint8` payload first and cast to `int4` only for the final conversion.
- `upshift`, `downshift` and `logical_downshift` are not raw shifts: each goes
  through an accumulator ups/srs pair, int4 shifts unpack/repack, and
  `max_shift` is 59. Both downshift variants set `rounding_mode::floor` per
  call. Avoid them in the innermost loop when a `to_float` shift can absorb the
  scaling. Use `logical_downshift` for unsigned nibble extraction and
  `downshift` for arithmetic sign-extending shifts.

### Lane expansion and permutations

- `broadcast_vector<N>(v, index)` broadcasts one N-element chunk. When
  `N * type_bits >= 128` it is `v.extract<N>(index).grow_replicate<Elems>()`.
  Q8_0's `broadcast_vector<8, bfloat16, 64>` is already the `grow_replicate`
  path.
- Repeating a 128-bit lane by a runtime or constant index is cheaper through
  `broadcast_elem_128` (XDNA2): one `vextbcst.128` takes the lane as a register
  operand, while `broadcast_vector` emits a `vshift` first. See
  `common/lane_broadcast.aie.h` (`repeat_lane32`/`repeat_lane64`). Measured in
  Q6_K's subgroup expansion; it regressed Q4_0, so the two formats use
  different forms. Details in `docs/rejected.md`.
- `vector::grow_replicate<ElemsOut>()` replicates the whole short vector. A
  64-lane bf16 value is two 512-bit registers, so each 8-lane expansion needs
  about four 128-bit copies per register (about 12 instructions), and four are
  live per block. `concat` ladders, halving the live set, and moving split
  constants to `.rodata` all grew `.text` and lost.
- `aie::broadcast<T, Elems>({...})` accepts a small repeating array when
  `type_bits * N <= 64` and N is a power of two; useful for byte masks.
- `concat()` has a tuple overload, so `concat(interleave_zip(even, odd, 1))`
  joins a deinterleave; `interleave_unzip` is the inverse.
- Q6_K's native decode keeps low nibbles in `uint4` until the final combine;
  the high plane expands to `uint8` lanes, where the even 2-bit field is masked
  and the odd field shifts right by 2. One `interleave_zip` assembles the 64
  high-bit lanes. A 32-lane `uint16` intermediate was a workaround for an old
  compiler crash; the direct shift is faster now.
- `filter_even`/`filter_odd(v, chunk_size)` select alternating chunks in one
  permute-style op. For 8-bit data this is byte granularity, so it fits ql/qh
  separation, not nibbles. `*_mode` variants allow runtime-selected modes.
- `transpose(v, Row, Col)` is one permute for small matrices; push layout
  transforms into host repack when possible.
- `interleave_custom_static<S0..S31>` implements arbitrary two-source lane
  selects and can replace chains of `shuffle_up`/`shuffle_down`.
- `shuffle_up_replicate`, `shuffle_down_fill`, `reverse`, and
  `select(v1, v2, mask)` are single ops; `mask<Elems>::from_uint64` builds
  compile-time masks.

### Codebook lookup tables

- `aie::parallel_lookup<T, aie::lut<4, Value>>` gives four parallel LUT accesses
  per instruction and is the promising path for IQ4_NL/IQ4_XS gathers.
- Index types must be at least 8-bit, so unpack 4-bit indices to `int8` first.
  `fetch` accepts at most 32 indices and returns at most 32 8/16-bit values
  (16 for 32-bit), so a 64-lane tile needs two fetches.
- 8-bit values are stored as 16-bit LUT entries. The
  `aie::lut<4, OffsetType, SlopeType>` wrapper hides the 4-way interleaved
  layout; see the `lookup_table` example shipped with `aie_api`. Construct the
  object once, outside the loop. Out-of-range policy is `saturate` by default,
  `truncate` optional.
- Store the codebook in the LUT output type: for IQ4_NL, bf16 `kvalues_iq4nl`
  avoids a post-lookup conversion at 2x LUT storage, negligible for 16 entries.

### mmul modes and emulation

- Dense modes on XDNA2: int8 x int4 `4x16x16`; int8 x int8 `4x8x8`, `8x8x8`;
  bf16 x bf16 `8x8x4`, `4x8x8`, `4x8x4`, `8x1x8`, and `8x8x8` only through
  bfp16 emulation. There is no bf16 x int4 mode, so bf16-activation
  dequantization must produce bf16 weights. `4x4x8` bf16 is unavailable.
- int8 x int16 (`8x2x8`, `4x4x8`) and int16 x int16 exist; not usable with bf16
  activations, but relevant to a future integer-activation path.
  `aie::mul<int8, int16>` returns `acc32`.
- `mmul<..., int8, int4>` consumes packed int4 B directly; a drop-in win only
  for a future W4A8 path with int8 activations.
- `mmul<8, 8, 8, bfloat16, bfp16ebs8>` lets bf16 A consume block-float8 B. It
  is a relaxed path, not exact. `mx4`/`mx6`/`mx9` block vectors are AIE-MLv2
  only.
- The current `4x8x8` bf16 mmul is emulated from `mul_elem_32`/`mac_elem_32`;
  the manual bf16 scale splits sit on top.
- `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1` switches to bfp16 hardware:
  faster but not exact. A named relaxed mode only.
- Element-wise float `aie::mul` is emulated via `__AIE_API_FP32_EMULATION__`
  and is much slower than the manual bf16 splits. Exact, but not a shortcut.
  `AIE2_FP32_EMULATION_ACCURACY_*` does not change the 32/64-lane paths.
- A default-constructed `aie::mmul` is zero-qualified: the first `mac` behaves
  as `mul`. `aie::op_zero(acc, first_iteration)` folds a runtime zero into it;
  `accum_ownership::ref` aliases an output buffer.
- `aie::op_neg(v, bool)` may fold into a subsequent mul/mmul; it does not fold
  `q - 32` centering or a `-dmin*min` bias.

### Loads, alignment, and transport-tile layout

- `load_v<Elems>` requires full vector alignment: 16 bytes for 128-bit, 32 for
  256-bit, 64 for 512-bit. Use `aie::vector_ldst_align_v<T, Elems>` in
  `static_assert`s.
- `load_unaligned_v<Elems>(p, aligned_elems)` is correct when the base is
  aligned to a known smaller boundary. `load_floor_bytes_v` deliberately rounds
  down. A 64-byte load from a 32-byte-aligned address silently returns the wrong
  tile.
- `vector<int4, 128>` is one 512-bit register and valid for `load_v<128>`.
- Q6_K and Q6_K_int8 pool both records' payloads, scales and bf16 `d` pairs at
  tile level; the `d` records occupy the former shared 32-byte tail pad. Q8_0
  keeps payload-first records and groups two per tile. Q4_0 variants keep
  288-byte records (320 for `q4_0_hilo`) and group eight per tile with scales
  first; their 32-byte payload loads need only 32-byte alignment.
- Host repack must stay plain bytes; host-compiled headers must not include
  `aie::vector`.
- Device-only layout views are fine for `static_assert`s, but hot-path struct
  member access can cost more than vector-pointer access.

## Shared kernel patterns

### Two-column fusion

Used by every format: Q4_0, its variants, Q8_0, Q6_K_int8 and native Q6_K.
Native Q6_K decodes both records into separate value sets and then issues both
accumulators' mmuls, which is what made two-record fusion worthwhile there.

- one `unpack_mmul<weights_rows, weights_cols>` per format;
- process two independent column tiles at a time;
- share the input tile between both columns;
- two mmul accumulators;
- fully unroll the inner tile loop;
- dequantize both columns, then issue both mmuls;
- store once per column tile.

What helped: inlining unpack directly into the fused kernel instead of cursor
objects; aligning tile payloads so `load_v` is valid; `aie::vector` pointers for
input, output and payload tiles; sharing scale/dequant helpers with the
standalone unpack test; exact bf16 scale terms in host repack when precision
fits; `broadcast_vector` for 8-to-64 expansion on Q4_0 (Q6_K uses
`repeat_lane64`).

What did not help (see `docs/rejected.md`): partial unrolling without measuring
the specific loop, `noinline` helpers, four columns, hoisting accumulators
across records, multiple accumulators per column, pointer increments,
`__restrict__`, `-O3`/`-Os`, copying the input tile, and dequantizing all tiles
before issuing mmuls.

### Transport-tile grouping

- Records are row-block-major then column-pair-major inside a tile; the tile is
  padded only to the next 64-byte boundary.
- Native Q6_K decodes four adjacent 8x8 tiles (32 rows) together, sharing one
  64-byte high-plane load; a 64-row chunk needs six 64-byte payload loads
  instead of eight.
- Q4_0 variants group eight 64-row records per tile; per-record scales-first is
  slightly faster than tile-wide pooling, and 32-byte payload loads stay valid.
- `record_count()` is the legacy record count; `packed_bytes()` is
  `tile_count * kPackedTransportTileBytes`.

### Scale handling

- Match the number of bf16 terms to the exactness requirement. fp16 `d` alone
  needs two exact terms (Q8_0). `d * int8_scale` has up to 18 significand bits
  and needs three (Q6_K).
- Q6_K stores each column's canonical fp16 `d` as a host-precomputed exact bf16
  `hi + lo` pair. The pair is exact for normal and subnormal values, so
  canonical subnormal scales no longer flush. The record grows from 16 to 32
  bytes per 8 columns but the 32-byte tail pad absorbs it: native Q6_K stays
  3392 bytes/tile, Q6_K_int8 4416.
- Q8_0 keeps canonical 2-byte fp16 slots; the device decodes four 32-row
  blocks' scales into one 32-lane batch with the exact exponent-add split,
  including subnormals. The old `|d| < 64` pre-bias limit and its repack failure
  path are gone.
- Q4_0 keeps canonical 288-byte records and decodes 32 scales per batch with
  the same shared split. `q4_0_hilo` pre-splits on the host and pays 320 bytes.
- The shared split computes `E << 7` once, builds `hi` with one `(E | top) << 5`
  plus the bias add, and folds the denormal `lo` exponent step into the same
  add. It is exact and the cheapest formulation measured.
- Expand the eight scale lanes to 64 once per block, not per tile. Q6_K uses
  `repeat_lane64` (register-lane `vextbcst.128`) and gains on both Q6_K variants;
  Q4_0 keeps `broadcast_vector<8, bfloat16, 64>` because the same change
  regressed it. 32-lane and 8-lane caches, and per-tile reload/split, were
  slower.
- Q6_K hoists `d` to once per record; each chunk's 32 int8 scales are expanded
  to 32 lanes, multiplied by `d`, and split into hi/mid/lo once.
- Rejected scale approaches are listed in `docs/rejected.md`.

### Register pressure and scheduling

- Two-column fusion is faster despite spilling more; do not trade its ILP for
  fewer spills.
- Per-column `noinline` helpers reduce pressure but lose too much ILP.
- Struct-based record views regressed the fused Q4_0 kernel; vector-pointer
  access is faster in the hot path.
- The hot loop keeps two accumulators, two records' scale terms and two payload
  streams live. Native Q6_K also keeps two decoded value sets. Four columns is
  too much.

### Mmul shape

- The kernels use `aie::mmul<4, 8, 8>` with bf16 inputs and weights.
- `4x4x8` is not defined for bf16; `4x8x4` is supported but doubles mmul
  operations and needs output-column splitting, and was not faster.
- The `vextbcst` instructions in the fused loop are inherent to `4x8x8`.

### Profiling method

Replace the dequantizer with a constant to separate mmul/loop overhead from
decode/scale work. This is how the scale path was shown to dominate Q6_K and
why Q8_0 is limited by mmul/input scheduling rather than payload decode.

## Format-specific design

### Q4_0

Three variants share one payload layout.

`q4_0` (canonical exact) uses 288-byte records (4.5 bits/weight):

```text
record:  64 rows, 32 bytes of scales then 256 bytes of payloads
scales:  8 canonical fp16 values per 32-row block
payload: 8 x 16-byte signed-int4 4x8 payloads per 32-row block
tile:    eight records, 2304 bytes
```

The device decodes both column records' 16 scales into one 32-lane batch with
the exact exponent-add split. This is the production layout.

`q4_0_packed16` uses the same 288-byte record and payload layout with a
host-packed 16-bit scale code (9-bit fraction). It is near-exact, not bit-exact:
about 0.78% maximum relative weight error and ~3.9% maximum output error because
cancellation amplifies the scale error. It therefore has a named mmul policy of
5% of the per-column output scale and cannot be the unprefixed Q4_0.

`q4_0_hilo` uses 320-byte records (5.0 bits/weight):

```text
scales:  8 pre-split bf16 hi/lo pairs per 32-row block, 32 bytes
record:  64 rows, 64 bytes of scales then 256 bytes of payloads
tile:    2560 bytes
```

The host writes the exact bf16 `hi + lo` split so the device skips fp16 decode,
at the cost of 11% more bytes. Exact.

All three order records by local column tile then local row record; every
32-byte payload tile is 32-byte aligned, so `load_v<32>` is valid.

### Q6_K

Canonical Q6_K is 6.5625 bits/weight; both repack variants target a 256x16
transport tile. The former 32-byte tail pad now holds the bf16 `d` pair, so
neither variant grows.

`q6_k` preserves the native six-bit payload, 3392 bytes/tile (6.625 bits/weight):

```text
3072 bytes  2 records x 4 chunks x (256-byte low-nibble + 128-byte high plane)
 256 bytes  2 x 4 x 32 per-column int8 subgroup scales
  64 bytes  2 x 32 per-column bf16 d hi/lo pair
```

The decoder processes four adjacent 8x8 tiles (32 rows) at a time: two 64-byte
low-plane vectors and one 64-byte high-plane vector per group, with the
high-plane masks shared across all four tiles. Planes expand in `uint4`/`uint8`
lanes before the bf16 scale terms. The standalone unpack processes a whole
record per call and loads the host `d` pair once.

`q6_k_int8` expands every 6-bit weight to signed int8 (`q - 32`) in host repack,
4416 bytes/tile (8.625 bits/weight):

```text
4096 bytes  2 records x 4 chunks x 512-byte int8 payloads
 256 bytes  2 x 4 x 32 per-column int8 subgroup scales
  64 bytes  2 x 32 per-column bf16 d hi/lo pair
```

Device decode is a 64-byte load plus `to_float<bfloat16>`; the per-16-row scale
structure is unchanged, so the expansion is lossless.

The largest win was processing both 8-column records together, sharing input
vectors and exposing two independent accumulators. The second was grouping four
row tiles per decode, which shares the high-plane load and masks and gives the
independent vector chains enough work to cover emulated-FP32 latency. The exact
scale path multiplies each chunk's int8 scales by `d` and splits into hi/mid/lo
once, exact for canonical scales up to 127.

### Q8_0

8.5 bits/weight. A record is 256 rows x 8 columns, 2176 bytes; two records form
a 4352-byte tile:

```text
2048 bytes  int8 payloads (32 tiles of 64 bytes)
 128 bytes  per-block canonical fp16 d (8 blocks x 8 columns)
```

Payloads come first so every 64-byte tile load is aligned. Each tile is one 8x8
row-major matrix: `load 64 signed bytes -> int8 vector -> bf16 vector`, then
`q * d`. An fp16 `d` needs only two bf16 terms; the exact exponent-add split
produces `hi + lo` with no rounding and the two-term accumulation is one bf16
rounding. Four blocks (32 columns) of scales decode per batch.

What helped: payload-first alignment, signed int8 to bf16 without a centering
bias, two-term `d`, `broadcast_vector` expansion, 256-row records instead of
128, and two-column fusion with one accumulator per column.

## Contracts

### Naming and canonical layouts

Use vector/length names for inputs and outputs, rows/cols only for weights:

- `inputs_vecs` = M = 4, the number of input vectors;
- `inputs_len` = K, the length of each input vector;
- `weights_rows` = K, the same dimension as `inputs_len`;
- `weights_cols` = N, output values per vector;
- `outputs_vecs` = M = 4;
- `outputs_len` = N.

```text
inputs:  [inputs_vecs][inputs_len]
weights: [weights_rows][weights_cols]
outputs: [outputs_vecs][outputs_len]
```

GGUF stores the weight matrix column-major: each of the `weights_cols` columns
is `weights_rows` contiguous values.

One `aie::mmul<4,8,8>` tile:

```text
inputs:  [weights_rows_tile][kMmulInputsVecs][kMmulInputsLen] bf16
weights: [weights_rows_tile][kMmulWeightsRows][kMmulWeightsCols] bf16
outputs: [kMmulOutputsVecs][kMmulOutputsLen] f32, accumulated
```

`kMmulInputsLen` and `kMmulWeightsRows` are the same dot-product dimension; the
outer `weights_rows_tile` index tiles that dimension and is not an
`inputs_vecs` tile.

### Transport-tile contract

Repack and transport operate on 256-row x 16-column tiles:

- `kTransportTileRows = 256`, `kTransportTileCols = 16`;
- `kTransportTileBytes` is the packed size rounded up to 64 bytes; every tile
  starts on a 64-byte boundary;
- tiles are row-block-major then column-pair-major:
  `tile_index = rows_tile * (weights_cols / kTransportTileCols) + col_pair`;
- Q8_0, Q6_K and Q6_K_int8 have 256-row records, so one tile holds two;
- Q4_0 variants have 64-row records, so one tile holds eight (four row records
  x two columns).

Each format defines tile-local pointer helpers in `layout.h`, typically
`record_payloads()`, `record_scales()`, `block_payloads()`, `block_scales()` and
`chunk()`.

### Repack API

```cpp
constexpr int64_t record_count(int64_t weights_rows, int64_t weights_cols);
constexpr int64_t tile_count(int64_t weights_rows, int64_t weights_cols);
constexpr int64_t packed_bytes(int64_t weights_rows, int64_t weights_cols);

int64_t repack_weights(const uint8_t* src,
                       int64_t src_col_stride_bytes,
                       uint8_t* dst,
                       int64_t dst_capacity,
                       int64_t weights_rows,
                       int64_t weights_cols);
```

`packed_bytes()` is `tile_count * kTransportTileBytes`. Repack requires
`weights_rows % 256 == 0` and `weights_cols % 16 == 0`.
`src_col_stride_bytes` supports padded views. Invalid dimensions or
insufficient capacity return `0`.

### Unpack API (test-only)

Compiled only when the test shim defines `QUANTS_BUILD_UNPACK_TEST`. Cycle
counts are diagnostic, never a format-selection criterion.

```cpp
void unpack_record(const uint8_t* packed, int weights_cols_tile,
                   int weights_rows_record, int cols_pairs,
                   bfloat16* weights_tile);

void unpack_weights_chunk(const uint8_t* packed, int weights_cols_tile,
                          int weights_rows_record, int cols_pairs,
                          int chunk_index, bfloat16* weights_tile);
```

`unpack_record()` resolves the transport tile, loads per-record metadata once,
and writes all chunks; Q6_K variants hoist `d` here.

### Fused API

Each format provides:

```cpp
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed, const bfloat16* inputs,
                        float* outputs);
```

The caller must zero `outputs` and select `conv_even` rounding. All formats
process the two column tiles of a transport tile together; native Q6_K also
decodes both records into separate value sets before issuing both mmuls.

### Shared implementation

- `common/tile.h` holds the `mmul<4,8,8>` shape constants.
- `common/scale.aie.h` provides `split_fp16_exact<N>()`,
  `split_fp16x16_exact()` and `split_fp16x32_exact()`. Q4_0 uses the 16-lane
  helper in standalone unpack and the 32-lane one in `mmul`; Q8_0 uses
  `split_fp16_exact<kScaleLanes>`; `q4_0_hilo` splits in host repack; Q6_K loads
  host `hi`/`lo` directly.
- `common/q6_scale.aie.h` holds the shared Q6_K path: `d` load, exact three-term
  chunk scales, subgroup broadcast, scale application.
- `common/mmul.aie.h` provides the vector mmul tile helper.
- `q4_0/unpack.aie.h` provides `dequantize_tile()` and the reference unpack
  entries; the fused path reuses `dequantize_tile()`.
- Keep host repack format-specific; do not leak format layout into
  `common/mmul.aie.h`.

### Constraints

- C++17, no exceptions, no RTTI.
- Headers use `#pragma once`; functions are `inline` or `constexpr`.
- `repack.h`/`layout.h` must not need AIE headers. Device code lives in
  `.aie.h`.
- `GGML_LOG_*` may be used by callers, not inside these headers.
- Layout and shared constants are namespace-scope `constexpr`, **not**
  `inline constexpr`.
- Names live in `spiral::quants::<name>`; shared constants in `spiral::quants`.
- Tests and benchmarks require `weights_rows % kRowsMultiple (256) == 0` and
  `weights_cols % 16 == 0`.
- Host repack byte counts, capacities, strides and offsets are `int64_t`;
  NPU-side counters and indices are `int32_t`.
- Production headers contain no tracing; test shims select
  `aie::rounding_mode::conv_even` once and call `event0()`/`event1()`.
- Use device-side vector pointers in hot loops; struct member access and cursor
  objects lost.

Format `layout.h` must `static_assert`:

- `kTileCols == kMmulWeightsCols`;
- `kUnpackTileRows % kMmulWeightsRows == 0`;
- record and chunk size consistency;
- `kPackedTransportTileBytes % 64 == 0`;
- payload alignment wherever the format uses 32/64-byte tile loads.

Keep `layout.h` constants in `constexpr int NAME = expression;` form; the parser
in `tests/common.py` reads them and treats `inline constexpr` functions as
constants.

## Testing and exactness

A format without a variant suffix (`q4_0`, `q6_k`, `q8_0`) must keep the
canonical tile size and be exact over the canonical format's finite value range:

- `tile_bytes == ceil64(4096 * canonical_bits_per_weight / 8)` for a 256x16
  tile: Q4_0 2304, Q6_K 3360 -> 3392, Q8_0 4352;
- unpacked bf16 weights equal the canonical dequantization bit-for-bit for every
  quant code and every finite fp16 scale, including subnormals and zero.
  Non-finite scales are out of scope. A format that grows the tile or relaxes
  exactness takes a name.

The size clause holds for all three and is pinned only by per-format
`static_assert`s; nothing checks it against the canonical formula yet. Exactness
holds for all three after Q8_0 moved to the exact split.

- Keep `unpack.py` as the exact format/layout conformance test: it catches
  scale, packing and dequant errors that an end-to-end tolerance test hides.
- Keep `mmul.py` for end-to-end fused correctness. Format-specific policies:
  `q4_0_packed16` uses 5% of the column scale; the default exact test uses
  `max(1e-5, 1e-3 * column_scale)`.
- Share scale/dequant helpers between the reference unpack and the fused mmul
  so tests and production cannot drift.
- Q6_K `mmul.py --cols 256` exceeds the single-tile harness memory; use 64 or
  128 columns.
- The Q6_K generator uses scales of 1-32 while canonical scales reach 127, so a
  wrong split can still pass end-to-end; add a direct split-residual test if the
  scale representation changes.
- Cover subnormal, zero and signed edge scales for every format with an fp16
  scale slot. `q4_0_packed16` has no fp16 slot: the host normalizes subnormals
  and rejects inf/nan.

Golden and end-to-end policies for the program live in
`src/program/DESIGN.md`; the shipped tolerance is
`max(1e-3, 6e-3 * column_scale)`.

## Target model

The kernels are developed against the artifacts in
[`../../docs/targets.md`](../../docs/targets.md): provenance, format allocation
and tensor shapes for each, with the current target first. The traffic figures
are in [`../../docs/measurements.md`](../../docs/measurements.md); the bring-up
checklist is in [`../program/TODO.md`](../program/TODO.md).
