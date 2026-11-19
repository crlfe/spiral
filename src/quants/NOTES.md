# NPU kernel notes

Findings from optimizing the `src/quants` Q4_0, Q4_0_packed16, Q4_0_hilo,
Q6_K, Q6_K_int8, and Q8_0 kernels on XDNA2 / AIE2P using Peano and IRON. The material is
organized as reusable AIE2 techniques, shared kernel patterns, and then
format-specific results. Current status and future plans live in `TODO.md`;
the quick summary lives in `README.md`.

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
  - Q4_0: 4.5 bits/weight (`q4_0`, 0.5625 bytes/weight), 4.5 bits/weight
    (`q4_0_packed16`, near-exact), or 5.0 bits/weight (`q4_0_hilo`, 0.625
    bytes/weight)
  - Q6_K: 6.625 bits/weight (`q6_k`, 0.828125 bytes/weight) or 8.625
    bits/weight (`q6_k_int8`, 1.078125 bytes/weight)
  - Q8_0: 8.5 bits/weight (1.0625 bytes/weight)
- Full-device estimate at clock `f`:
  `min(weights/cycle * f * 32, 55 GB/s / bytes-per-weight)`.
  The first term is the compute cap, the second the packed-weight memory cap.
  Only the compute cap scales with clock. Report 1.5 GHz as the typical NPU
  clock and 1.8 GHz as the measured one; `README.md` tabulates both.
- Cycle counts and weights/cycle are clock-independent, while weights/s and
  bytes/s scale linearly with clock. The per-format tables below carry the
  1.8 GHz measurement; their compute-implied GB/s columns should be multiplied
  by 0.833 for 1.5 GHz. `README.md` reports est weights/s using the 1.5 GHz
  compute cap and the fixed 55 GB/s memory cap.
- 60 GB/s at 1.8 GHz requires about 1.67 weights/core-cycle at the 5.0-bit
  Q4_0 layout, or 1.85 at canonical 4.5 bits (scale by 1.2 for 1.5 GHz).
- One AIE2 VLIW bundle is 16 bytes.
- The `mmul.py` harness reports each core ELF's `.text` size as a share of the
  16 KiB AIE2P compute-tile program memory, and its largest per-function
  `.stack_sizes` frame. That figure is a single-frame maximum, not a transitive
  call-chain total, and is a lower bound when a reachable function was built
  without stack-size information. Fully unrolling is not automatically faster;
  wide unrolled loops can lose more to instruction fetch than they gain in ILP.
- Set `QUANTS_KERNEL_FLAGS` to append Peano flags to the test builds. The
  harness always enables `-Rpass-missed=loop-unroll` and prints captured
  compiler warnings (including `-Wpass-failed=transform-warning` for failed
  `#pragma unroll` transformations) to stderr as `compiler: ...`, with the
  loop line. Add `-Werror=pass-failed` to make those failures fatal.
- `#pragma unroll` maps to `llvm.loop.unroll.enable`, a hint equivalent to
  `unroll(enable)`. `#pragma clang loop unroll(full)` maps to
  `llvm.loop.unroll.full`. `-O3` produced identical code and cycles to `-O2`
  for every quants kernel.
- Unroll pragmas matter on a per-loop basis. The 2-iteration block loop in
  `q4_0_hilo` is worth keeping unrolled: dropping the pragma regressed the
  256x16 kernel by 5.9% (2008 vs 1896 cycles). The equivalent `q4_0` and
  `q8_0` block loops compile the same with or without it.

## Loop optimization and Peano idioms

This is the general optimization playbook for Peano/AIE2P kernels in this
package. It applies to every quant format, including Q4_0, Q4_0_packed16, Q4_0_hilo,
Q6_K, Q6_K_int8, Q8_0, the K-quants, and the IQ formats. The measured examples below
come from a standalone 64-byte copy microbenchmark and a fixed-shape Q4_0
experiment; the scratch sources were not kept, but the results are retained
here. The conclusions are about Peano code generation, not about one quant
layout.

### General playbook

1. **Classify the bottleneck before changing code.** Address arithmetic,
   scale decode, the mmul recurrence, code size, and memory traffic need
   different fixes. Measure the current kernel and inspect the disassembly
   first.
2. **Write pointer-increment loops.** Prefer `ptr += stride` over indexing.
   This lets Peano use post-increment addressing and often discovers a
   single-register load/store rotation. It is the biggest single win on
   address-bound loops.
3. **Sweep unroll pragmas on the actual loop.** Try `unroll(disable)`,
   `unroll(enable)`, `unroll_count(N)`, and `unroll(full)`. Check cycles,
   `.text`, and the compiler warning output. `unroll(full)` can fail with
   "loop not unrolled" on large bodies; treat that as no change, not success.
4. **Keep two-column (or two-channel) fusion** where it exists. It shares
   input loads and exposes independent mmul accumulators. Do not split it
   without measuring; the lost ILP is usually worse than the extra registers.
5. **Use backend flags only after the disassembly identifies a problem.**
   The common `aie-*` unroll, pipeliner, and register flags had no effect on
   the copy kernels, and the register-rewrite variants were slower on q4_0
   (9168/9104 cycles vs 8720 for aggressive-realloc/swpaware). The neutral
   list is below.
6. **Prefer host precomputation when the repack contract allows it.** Moving
   fp16 decode, hi/lo splitting, or 6-bit scale/min expansion into host repack
   removes device work. This is the main reason Q4_0_hilo is faster than Q4_0.
   It is a format/bytes trade, not a scheduling trick.
7. **Only hand-write assembly when the compiler cannot express the schedule.**
   The copy benchmark shows the compiler matching a hand-written 8-deep load
   pipeline when the source shape is right. Hand assembly is justified by a
   measured compiler limitation, not by the hope of beating the compiler.

### Pointer increments and the copy microbenchmark

Write the loop so the compiler can see pointer increments:

```cpp
const uint8_t* src = in;
uint8_t* dst = out;
for (int i = 0; i < n; ++i) {
  const auto v = aie::load_v<64>(src);
  aie::store_v(dst, v);
  src += 64;
  dst += 64;
}
```

This generates `vldb x2, [p0], #0x40` / `vst x2, [p1], #0x40`. Indexed forms
generate `movs dj0, r0; add r0, r0, r1` and cannot use post-increment
addressing. The compiler also discovers a **single-register rotation**: a
one-bundle ZOL body with `vldb x2; vst x2`, a seven-load prologue, a seven-store
epilogue, and a loop count of `n - 7`. The in-order writeback makes the store
read the value loaded seven cycles earlier.

Measured on a 16 KiB copy (256 x 64 B):

| C++ form / hand asm | cycles | bytes/cycle | kernel `.text` | program `.text` |
| :-- | --: | --: | --: | --: |
| indexed, rolled | 2313 | 7.1 | 176 B | 512 B |
| indexed, `unroll(enable)` | 516 | 31.8 | 272 B | 608 B |
| indexed, `unroll_count(16)` | 441 | 37.2 | 320 B | 656 B |
| indexed, `unroll(full)` | 263 | 62.3 | 1568 B | 1904 B |
| **pointer, no pragma** | **266** | **61.6** | **208 B** | **544 B** |
| pointer, `unroll_count(8)` | 263 | 62.3 | 272 B | 560 B |
| indexed, manual 8x | 670 | 24.5 | 576 B | 912 B |
| hand asm, 8-deep | 271 | 60.5 | 436 B | 800 B |

`unroll(enable)` is a real 4.5x win on the indexed loop, but pointer increments
alone match it with half the code. `unroll(full)` is fastest but costs ~1.9 KiB
of program memory. Hand-written assembly is competitive but not faster than the
compiler's pointer loop, and uses more text. For simple load/store loops, try
the pointer form first.

### Bottleneck-to-action guide

| symptom | first action | notes |
| :-- | :-- | :-- |
| Lots of `movs`/`add` index arithmetic | pointer increments | biggest copy-loop win |
| `vldb` followed by five nops | pointer rotation or more unrolling | load result latency is 7 |
| Scale decode dominates; repack can change | host-precompute/hi-lo layout | Q4_0_hilo vs Q4_0 is the model |
| Scale decode dominates; repack fixed | accept it or use a named relaxed mode | device fp16 decode is already near best |
| mmul accumulator chain serializes | two-column fusion, static unroll, two accumulators | more accumulators often spill worse |
| Compiler says "loop not unrolled" | manual unroll or a smaller body | `unroll(full)` can silently fail |
| Code size is the constraint | roll outer loops, partial unroll | full unroll is a bandwidth/latency trade |
| Memory-bound (bytes/weight) | change repack precision or host layout | scheduling will not help |
| Relaxed accuracy allowed | single-bf16 scale or bfp16-emulated mmul | must be a named variant |

For Q6_K/Q6_K_int8/Q8_0, the same actions apply to the `payloads`, `scales`,
and `d` pointers. For IQ4_NL/IQ4_XS, keep codebook/LUT data in the lookup
memory and use pointer increments for the index streams. For Q4_K/Q5_K,
host-expand the 6-bit scale/min metadata first, then apply the loop rules.

### Counterexample: pointer increments in the q4_0 mmul

Pointer increments are not universally faster. A q4_0 experiment kept the
exact two-column fusion and changed only which addresses advance:

| variant (256x16 / 256x64) | 256x16 cycles | 256x64 cycles |
| :-- | --: | --: |
| q4_0 baseline (indexed records, payloads, inputs) | **2204** | **8720** |
| payload pointer only | 2204 | 8720 |
| input pointer only | 2280 | 8934 |
| record/scale/payload pointers, indexed input | 2293 | 9061 |
| all structural pointers + input pointer | 2298 | 9100 |

Advancing the payload pointer is neutral (the compiler already turns
`payloads[block][tile]` into the same loads). Advancing the input or record
pointers regresses by 2-4%: the q4_0 loop is recurrence-bound, and the pointer
form adds loop-carried pointer state and live ranges that disturb the
compiler's existing schedule. The indexed form lets the compiler keep the
input tile address independent of the loop-carried accumulator chain.

Use pointer increments when address arithmetic is the bottleneck, as in the
copy loop. For a dequant+mmul recurrence, measure both forms; indexed code can
be better even when the same rule wins elsewhere.

### Backend flags are usually not the lever

On the copy kernels, these left cycles and code size unchanged:

- `-O3`
- `-mllvm -aie-unroll-max-count=1/2/4/8/16`
- `-mllvm -aie-unroll-only-pragma-loops=false`
- `-mllvm -aie-prefer-swp-over-unroll=1/4/9`
- `-mllvm -aie-pipeliner-track-regpressure=false`
- `-mllvm -aie-pipeliner-max-stagecount=8`
- `-mllvm -aie-postpipeliner-target-ii=4`
- `-mllvm -aie-postpipeliner-force-stagecount=2`
- `-mllvm -aie-aggressive-realloc=true`
- `-mllvm -aie-reg-rewrite-mode=swpaware`
- `-mllvm -aie-stack-minimize=false`
- `-mllvm -aie-merge-congruent-ivs=true`
- `-mllvm -aie-loop-idiom-unrolling-threshold=350/1000`
- `-mllvm -aie-unroll-max-cost=50/500`

Prefer source shape and explicit pragmas; use backend flags only after the
disassembly shows a specific problem.

### Compiler diagnostics and loop inspection

- Set `QUANTS_KERNEL_FLAGS` to append Peano flags to test builds.
- Add `-Werror=pass-failed` to make failed `#pragma unroll` transformations
  fatal. Without it, a failed `unroll(full)` is only a warning and the loop
  stays rolled.
- Use `llc -pass-remarks-output=...` on the kernel IR to see pipeliner II/NS,
  ZOL conversion, register spills, and bundle counts.
- Use `llvm-objdump -d --no-show-raw-insn` to compare the actual bundle/issue
  schedule before and after a change.
- `-Rpass-analysis=aie-hardware-loops` and the pipeliner remarks are useful,
  but many diagnostics are emitted by `llc`, not by the clang driver.

### ZOL and hand-written assembly facts

- A ZOL is written with `add.nc lc, rN, #adj`, `movxm ls, #.Lbody`, and
  `movxm le, #.Lend`.
- **`le` must name a full 16-byte bundle.** A 14-byte final bundle silently
  turns the ZOL into a fall-through loop that executes once.
- The loop setup must be at least ~112 bytes (7 bundles) from `le`. A shorter
  body does not loop; pad before `.Lbody` or put the prologue after the
  `ls`/`le` setup, as the compiler does.
- The `event #0` / `event #1` mnemonics are rejected by the AIE2P asm parser
  even though `llc` prints them. The raw encodings are
  `event #0 = .4byte 0x10100018`, `event #1 = .4byte 0x12100018`.
- `ExternalFunction` compiles one C/C++ source; a separate `.s` file is not
  linked automatically. Embed hand-written assembly with a global
  `__asm__("...")` block, or link a prebuilt object explicitly.

### Where to look for the machine model

- VLIW slots and bundle forms: `AIE2Slots.td`, `AIE2MultiSlotPseudoInstrInfo.td`.
- Latencies, bypasses, and resource hazards: `AIE2Schedule.td`
  (`InstrItinData` / `InstrStage`). Key values: `vldb` result 7, `vlda` 7,
  `vmul`/`vmac` to accumulator 5, `vadd`/`vsub`/`vextbcst`/`vshuffle` 2,
  `vconv` 2, `vups` 3, `vsrs` 4, `vst` completion 5, ALU 1.
- Assembly tests and bundle validation: `llvm/test/MC/AIE/`.
- Compiler loop guidance: `~/Projects/llvm-aie/PERF_OPT_GUIDE.md`.

### Reference results and scope

The copy benchmark is a "can the compiler already do this?" test. The q4_0
dequant+mmul recurrence is different: the pre-RA pipeliner cannot schedule it
and the post-RA pipeliner reports `II=63, NS=1`, so static unrolling remains
the only effective ILP tool there. Do not generalize "the compiler is always
good enough" from the copy result, and do not generalize "hand asm is always
faster" either. The order is: pointer-increment C++, pragma sweep, measure,
inspect the disassembly, then hand-write only the part the compiler cannot
reach.

A fixed-shape 256x16 Q4_0 experiment is a useful counterexample: pointer
increments, removing the outer column loop, shift-based fp16 decode, and manual
record unrolling reached 2316 cycles, but the existing `q4_0` schedule is 2204.
The loop is recurrence-bound, not address-bound, so the copy-loop win does not
transfer. `q4_0_hilo` (1896 cycles) is faster and still exact, but it changes
the repack/size contract; a relaxed single-bf16 scale variant (2168 cycles,
2.14 max abs error) is faster still and gives up exactness.

## Reusable AIE2 techniques

These are the API and code-generation findings worth carrying into future
quantized operations. They were checked against the mlir-aie docs, the
programming guide, the `aie_api` headers shipped with mlir-aie
(`include/aie_api/aie.hpp`), and its examples.

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
  with shifts, masks, and in-order `concat`. The native Q6_K layout uses a
  low-nibble plane and a two-bit high plane this way. Vectorizing the expansion
  removed most of the decode cost, but it is still the native variant's
  bottleneck.

### float16 on XDNA2

- XDNA2 is AIE2P / arch 21. The AIE API defines `float16` only for arch 22
  (AIE-MLv2); on XDNA2 it is a placeholder. The capability probe in
  `tests/bench/float16_support.*` confirms:
  - `sizeof(float16) == 1`, so it cannot hold a 16-bit value;
  - `aie::vector<float16, N>` fails its `static_assert` constraints;
  - there is no fp16 load/store, int8/f32/accfloat conversion, add, mul, or
    `mmul`;
  - `to_v16float16` / `to_v32float16` exist only in the `aie2ps` intrinsics,
    not in `aie2p`.
- Treat fp16 as a storage and scale format only. Canonical fp16 `d` values
  must be decoded exactly on the device with the exponent-add split (Q4_0 and
  Q8_0), pre-split on the host as Q6_K and Q4_0_hilo do, or replaced by a named
  host-side encoding (`q4_0_packed16`). Do not build a kernel around fp16
  arithmetic.
- Prefer the shift-based decode when fp16 decoding is hot: zero-extend the
  pattern, `(u << 17) >> 4` to form `(h & 0x7fff) << 13`, OR in the sign, and
  add `0x38000000` for the f32 exponent bias. The original probe measured 33.2
  vs 37.3 cycles per eight scales with zero/subnormal flush, and 43.2 vs 47.3
  with the bf16 hi/lo split. No format uses this f32 decode today: Q4_0 and
  Q8_0 consume canonical fp16 scales and decode them with the exact
  exponent-add split below.
- One-core probe timings, cycles per iteration over 32 iterations:

  | operation per iteration | cycles |
  | :-- | --: |
  | 128-byte uint16 load + store | 11.0 |
  | 128-byte bf16 load + store | 11.0 |
  | 8 fp16 -> f32, mask/rebias | 37.3 |
  | 8 fp16 -> bf16 hi/lo, mask/rebias | 47.3 |
  | 8 fp16 -> f32, shift | 33.2 |
  | 8 fp16 -> bf16 hi/lo, shift | 43.2 |
  | 8 pre-biased fp16 -> bf16 hi/lo, shift | 43.2 |
  | 16 host-packed fp16 -> bf16 hi/lo | 28.2 (14.1/8) |
  | 16 fp16 -> bf16 hi/lo, exp add, denorm flush | 43.3 (21.6/8) |
  | 16 fp16 -> bf16 hi/lo, exp add, denorm exact | 60.3 (30.2/8) |
  | 64 int8 -> bf16, one bf16 scale | 27.1 |
  | 64 int8 -> f32, one f32 scale (emulated) | 109.2 |

  The emulated f32 scale path is 4x the bf16 path, consistent with the
  element-wise float emulation noted above.
- Host-side re-biasing alone does not reduce the device f32 expansion. Storing
  `d * 2^10` and decoding with the +102 bias measures the same 43.2 cycles as
  the canonical +112 shift decode with a subnormal flush. The re-bias removes
  the subnormal case but zero still needs an exponent select, and the f32 field
  placement still needs two shifts/masks plus the bias add. `cast_to<float>`
  and the accfloat construction are bit-level moves, so building the value
  directly in an accfloat accumulator does not remove any instructions.
- A 16-value host-packed encoding keeps the two bf16 terms in the same 32
  bytes as the canonical fp16 scales without decoding fp16 on the device. The
  host normalizes each value to a bf16 exponent and a 9-bit significand
  fraction, then packs `sign | (exp8 - 102) << 9 | top7 << 2 | residual2`.
  The residual code is `{0, 1.5, 2, 3}` in 1/512 units, so its exponent delta
  and mantissa bit are already encoded and the device only does shifts, masks,
  adds, and two selects. `hi + lo` reproduces the 9-bit fraction within 0.5
  ulp (about 0.1% relative); zero is exact; normals and subnormals share one
  path because the host normalizes both. The 16-value probe measures 28.2
  cycles, i.e. about 14.1 per eight scales, versus 43.2 for the shift-based
  fp16 decode plus bf16 split and 61.4 for the exact direct split. The cost is
  one dropped mantissa bit (9-bit fraction instead of the full 11-bit fp16
  significand) and no representation for inf/nan, which host repack must map.
  This is a way to keep `d` at two bytes where a host-precomputed hi/lo pair
  has no tail padding to spare; the exact pair remains preferable when it
  fits.
- For canonical fp16 bytes, the low byte can go through the native uint8 ->
  bf16 conversion and then have `(E - 25) << 7` added to its exponent field;
  the high byte supplies `hi` with bit ops. This avoids both the lookup table
  and the bf16 multiply. The denorm-flush variant measures 43.3 cycles per 16,
  or 21.6 per eight. Keeping denorms exact needs a second high-byte form,
  `hi = top2 * 2^-16`, and a `-24` delta for `lo`; that costs 60.3 per 16, or
  30.2 per eight, so denorm support adds about 8.5 cycles per eight here. The
  host-packed 16-value format is still faster and handles denorms without a
  special case at its usual 0.1% bound, so use the exponent-add path only when
  the two canonical fp16 bytes must be preserved. Both exponent-add probes are
  bit-exact against their host models; the flush model drops `E == 0`.
- A small scalar packed immediately before an aligned tile load is unsafe. A
  probe that loaded int8 at `in + 2` silently read from `in` because
  `load_v<64>` requires 64-byte alignment. Reserve an aligned block for scales
  and payload tiles rather than relying on an offset within a vector load.

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
  broadcast plus a two-register concat. A 64-lane bf16 value is two native
  512-bit registers, so each 8-lane expansion needs about four 128-bit copies
  per register (about 12 instructions) and four of them are live per block.
  A `concat` ladder, processing one column record at a time to halve the live
  set, and moving the split constants into `.rodata` were all measured on
  q8_0 and all regressed `.text` (1016-1229 slots against 1008).
- `aie::broadcast<T, Elems>({...})` accepts a small repeating array on Gen2
  when `type_bits * N <= 64` and N is a power of two. It packs the pattern into
  a scalar and broadcasts it; useful for repeating byte masks such as
  `{0x0F, 0xF0}` or 2-bit plane masks.
- `concat()` has a tuple overload, so `concat(interleave_zip(even, odd, 1))`
  joins the two halves of a deinterleave. `interleave_unzip` is the inverse.
- Q6_K's native six-bit decode keeps the low nibbles in `uint4` until the
  final combine. The high plane is expanded to `uint8` lanes, where the even
  2-bit field is masked directly and the odd field uses a logical right shift
  by 2. A single `interleave_zip` then assembles the 64 high-bit lanes. An
  earlier 32-lane `uint16` intermediate was a workaround for a compiler crash
  on 8-bit shifts; the current compiler handles the direct shift and it is
  faster (24453 vs 25957 cycles, 2352 vs 2416 bytes).
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
  SlopeType>` wrapper hides the required 4-way interleaved layout; see the
  `lookup_table` example shipped with `aie_api`. Construct the
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
- XDNA2 also exposes int8 x int16 (`8x2x8`, `4x4x8`) and int16 x int16 mmul
  modes. They are not usable with bf16 activations, but they matter for a
  future integer-activation path. `aie::mul<int8, int16>` is also available and
  returns an `acc32` accumulator.
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

### Loads, alignment, and transport-tile layout

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
- Repack is defined on 256-row x 16-column transport tiles. Tiles are laid out
  row-block-major then column-pair-major, and each tile is padded only to the
  next 64-byte boundary. This keeps every tile contiguous and 64-byte aligned
  for DMA.
- Q6_K and Q6_K_int8 pool both records' payloads, scales, and bf16 hi/lo `d`
  pairs at tile level; the `d` records occupy what used to be a shared 32-byte
  tail pad. Q8_0 keeps its payload-first
  2176-byte records and groups two per tile. Q4_0 and q4_0_packed16 keep
  288-byte records, q4_0_hilo keeps 320-byte records; all group eight per tile
  with scales first. Their 32-byte payload loads only need 32-byte alignment.
- Host repack must stay plain bytes; host-compiled headers must not include
  `aie::vector`.
- Device-only layout views are useful for static assertions, but hot-path
  struct member access can cost more than vector-pointer access.

## Shared kernel patterns

### Two-column fusion

The two-column fused structure is used by Q4_0, q4_0_packed16, Q4_0_hilo,
Q8_0, Q6_K_int8, and native Q6_K. Native Q6_K decodes both records into separate value sets and
then issues both accumulators' mmuls, which is what made the two-record fusion
worthwhile there.

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

- partial unrolling (`unroll_count(4)`) or changing unroll pragmas without
  measuring the specific loop; per-loop effects vary;
- `noinline` on `next()` or the mmul helper;
- four columns processed concurrently;
- hoisting accumulators across records for Q4_0 (it already scales well);
- multiple accumulators per column;
- pointer increments instead of indexed addresses;
- `__restrict` on the fused kernel pointers;
- `-O3` / `-Os` instead of the default optimization;
- copying the input tile before both mmuls;
- dequantizing all tiles before issuing mmuls.

### Transport-tile grouping

- Repack consumes 256-row x 16-column tiles. Records are laid out
  row-block-major then column-pair-major inside the tile, and the tile is
  padded only to the next 64-byte boundary.
- Q6_K and Q6_K_int8 pool both records' payloads, scales, and bf16 hi/lo `d`
  at the tile level. The two 32-byte `d` records occupy what was the shared
  32-byte tail pad, so the tile size did not change.
- Native Q6_K decodes four adjacent 8x8 tiles (32 rows) together, sharing one
  64-byte high-plane load between them. A 64-row chunk then needs exactly six
  64-byte payload loads instead of eight.
- Q4_0, q4_0_packed16, and Q4_0_hilo group eight 64-row records per tile. Their
  per-record scales-first layout is slightly faster than pooling payloads
  tile-wide, and their 32-byte payload loads remain valid because records stay
  32-byte aligned.
- `record_count()` remains the legacy record count; `packed_bytes()` is now
  `tile_count * kPackedTransportTileBytes`.

### Scale handling

- Match the number of bf16 terms to the exactness requirement:
  - fp16 `d` alone: two bf16 terms are exact (Q8_0).
  - fp16 `d * int8_scale`: up to 18 significand bits, so three bf16 terms are
    required (Q6_K).
- Q6_K stores the per-column canonical fp16 `d` as a host-precomputed exact
  bf16 `hi + lo` pair. The pair is exact for normal and subnormal fp16 values,
  so canonical subnormal scales no longer need flushing. The record grows from
  16 to 32 bytes per 8 columns, but the 32-byte tile tail pad absorbs it:
  native Q6_K remains 3392 bytes/tile and Q6_K_int8 remains 4416 bytes/tile.
  The device loads the two bf16 planes directly and does not decode fp16.
- Q8_0 keeps its canonical 2-byte fp16 scale slots and stores them unchanged.
  The device decodes four 32-row blocks' scales into one 32-lane batch with the
  exact exponent-add split, including subnormals, so the `|d| < 64` pre-bias
  limit and its repack failure path are gone. This is exact over the whole
  finite fp16 scale range and also faster than the pre-biased 8-wide path:
  8242 -> 7707 cycles for 256x64 (7792 after the intermediate step),
  1.988 -> 2.126 weights/cycle. It costs
  program memory (2624 -> 7168 B) and stack (1216 -> 2816 B) because the exact
  split is inlined. The pre-bias helpers (`fp16_prebias_scale`,
  `load_fp16x8_biased`) and the then-unused `split_bf16_hi_lo` were removed;
  Q4_0 and Q8_0 now share `split_fp16_exact()`. Every int8 value, including
  `-128`, is exactly representable in bf16, so the old `-128` repack warning
  and its `RepackWarnings` helper were removed too.
- The real Gemma 4 E2B Q6_K `token_embd` output head contains subnormal fp16
  scales. The host hi/lo `d` split above makes those values exact; the synthetic
  Q6_K golden now mixes normal and subnormal `d`, and its verification checks
  relative error so a flushed block cannot pass. The `ggml-spiral` backend (`integrations/llama.cpp`) now
  re-enables native Q6_K after regenerating packed weights; its real-head slice
  check passes (max_abs 0.026 on max_cpu 10.07) and the Q4_0 Gemma model
  output matches CPU.
- `q4_0_hilo` has the host repack the exact bf16 `hi + lo` pair of each fp16
  scale. The record grows from 288 to 320 bytes, but the device avoids the
  fp16 decode and split. `q4_0` keeps the canonical 288-byte layout and now
  stores canonical fp16 scales too; the device uses the exact exponent-add
  split, which handles subnormals without the old `|d| < 64` pre-bias limit.
- The exact exponent-add Q4_0 path decodes both column records' 16 scales into
  one 32-lane batch in the fused kernel. It measures 7883 cycles for 256x64
  against 8720 for the previous pre-biased path (8328 after the first tuning
  step described below), with program memory growing from 2752 to 4048 bytes and the stack frame to 2496 bytes. Standalone unpack
  regressed from 5261 to 6579 cycles, but that path is now a diagnostic
  correctness reference and does not drive the production decision.
- The shared split now computes `E << 7` once and reuses it for both terms,
  builds `hi` with one `(E | top) << 5` plus the bias add, and folds the
  denormal `lo` exponent step into the same add. It stays exact and is the
  cheapest formulation measured: Q4_0 8328 -> 7883 cycles and 4160 -> 4048 B,
  Q8_0 7792 -> 7707 and 7248 -> 7168 B.
- Expand the eight scale lanes to 64 once per block, not per tile.
  `broadcast_vector<8, bfloat16, 64>` is the preferred expansion.
- The 64-lane active scale cache is the best balance found. 32-lane scales
  plus per-tile expansion, 8-lane scales plus per-tile expansion, and
  reload/split per tile were all slower.
- Q6_K's `d` is hoisted to once per record as an exact bf16 hi/lo pair; each
  chunk's 32 int8 scales are expanded to 32 lanes, multiplied by `d`, and split
  into hi/mid/lo once.
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
- The hot loop keeps two accumulators, two records' scale terms, and two
  payload streams live. Native Q6_K also keeps two decoded value sets live for
  the two-record fusion. Four columns remains too much register pressure.

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
decode/scale work. On the pre-quad Q6_K body, for example, the mmul and loop
alone were 3211 cycles (31%); payload load plus int8-to-bf16 reached 4297
(42%); the exact scale split and three multiplies accounted for the remaining
6042 cycles (58%). Q8_0's constant-dequantizer profile is about 3786 cycles
(46% of total).

## Format-specific results

### Q4_0

Two exact layouts and one near-exact layout are available.

`q4_0_hilo` (host-expanded) is one 64-row x 8-column record of 320 bytes
(5.0 bits/weight):

```text
 64 bytes  2 x 32-byte pre-split bf16 hi/lo scale blocks
256 bytes  8 x 32-byte int4 tile payloads
```

The host repack writes the exact bf16 `hi + lo` split of each fp16 scale.
Scales come first, payloads second; every 32-byte tile is naturally aligned.
All Q4_0 variants group eight records into a 256x16 transport tile: 2304 bytes
for `q4_0` and `q4_0_packed16`, 2560 bytes for `q4_0_hilo`.

`q4_0` (canonical exact) is 288 bytes (4.5 bits/weight): the same payload
layout with canonical fp16 scales, decoded on the device with the exact
exponent-add split. `q4_0_packed16` also uses 288 bytes but stores a
host-packed 16-bit scale code; it is near-exact and has its own verification
policy.

Results at the measured 1.8 GHz H clock (the host-interface clock of the
AIE array; per-kernel cycles come from `event0`/`event1`):

| variant | shape | cycles | cycles/tile | weights/cycle | compute-implied 32-core GB/s |
| :-- | --: | --: | --: | --: | --: |
| q4_0 (canonical exact) | 256x64 | 7883 | 246.344 | 2.078 | 67.3 |
| q4_0_packed16 | 256x64 | 7397 | 231.156 | 2.215 | 71.8 |
| q4_0_hilo | 256x64 | 7268 | 227.125 | 2.254 | 81.1 |

At 288 bytes, `q4_0` and `q4_0_packed16` retain the bandwidth advantage over
the 320-byte `q4_0_hilo` (about 98B weights/s at 1.8 GHz versus 88B). The exact
path is compute-bound at 1.5 GHz (about 94B); packed16 raises the compute
ceiling, but it was trialled in the program and dropped (the exact path left a
6.2% gap), so it stays a named quants-format experiment.

Standalone unpack at 256x64: 6579 cycles for the canonical exact `q4_0`,
5459 for `q4_0_packed16`, and 2916 for `q4_0_hilo`. Unrolling the packed16
block loop is worth about 1.1k cycles here, unlike the exact path.

`q4_0_packed16` keeps the 288-byte record and stores a host-packed 16-bit scale
code with a 9-bit fraction. The fused kernel measures 7397 cycles, close to the
320-byte `q4_0_hilo`, while standalone unpack is 5459 cycles. Its weights are
near-exact rather than bit-exact: the Q4_0 reference reports about 0.78% maximum
relative weight error, and the fused mmul check has about 3.9% maximum relative
output error (0.18% mean) because cancellation amplifies the scale error. It
therefore uses a named near-exact mmul policy (5% of the per-column output
scale) and does not pass the exact tolerance used by the unprefixed Q4_0, which
is why it cannot be the production path.

The fast single-bf16 mode measured 2119 cycles at 64 columns on the old layout
but is not exact.

### Q6_K

Canonical Q6_K is 6.5625 bits/weight. Both repack variants target a 256x16
transport tile. The former 32-byte tail pad now holds the exact bf16 hi/lo `d`
pair, so neither variant grows in size.

`q6_k` preserves the native six-bit payload. Its 3392-byte tile is 6.625
bits/weight:

```text
3072 bytes  2 records x 4 chunks x (256-byte low-nibble + 128-byte high plane)
 256 bytes  2 x 4 x 32 per-column int8 subgroup scales
  64 bytes  2 x 32 per-column bf16 d hi/lo pair
```

The device decoder processes four adjacent 8x8 tiles (32 rows) at a time. Each
group loads two 64-byte low-plane vectors and one 64-byte high-plane vector,
shares the high-plane masks across all four tiles, and expands the planes in
`uint4`/`uint8` lanes before applying the bf16 scale terms. A 64-row chunk now
needs six 64-byte payload loads instead of eight. The standalone unpack
processes a whole record per call, loading the host-precomputed bf16 hi/lo `d`
pair once instead of once per 64-row chunk. The host hi/lo pair also removes
the device fp16 decode; 256x64 unpack now measures 8498 cycles.

`q6_k_int8` expands every 6-bit weight to signed int8 (`q - 32`) in host
repack. Its 4416-byte tile is 8.625 bits/weight:

```text
4096 bytes  2 records x 4 chunks x 512-byte int8 payloads
 256 bytes  2 x 4 x 32 per-column int8 subgroup scales
  64 bytes  2 x 32 per-column bf16 d hi/lo pair
```

Device decode is a 64-byte load plus `to_float<bfloat16>`. The per-16-row scale
structure is unchanged, so the expansion is lossless.

Results at the measured 1.8 GHz H clock:

| shape | cycles | cycles/tile | weights/cycle | bytes/weight |
| :-- | --: | --: | --: | --: |
| 256x64 q6_k native | 10180 | 318.125 | 1.609 | 0.828125 |
| 256x64 q6_k native standalone unpack | 8498 | 265.562 | 1.928 | - |
| 256x64 q6_k_int8 payload | 9101 | 284.406 | 1.800 | 1.078125 |
| 256x64 q6_k_int8 standalone unpack | 4210 | 131.562 | 3.892 | - |
| Q8_0 256x64 reference | 7707 | 240.844 | 2.126 | 1.0625 |

At 1.5 GHz the native kernel is memory-bound at about 66B weights/s, with a
compute ceiling of about 79B; at 1.8 GHz the compute ceiling rises to about
95B but memory still binds at 66B. `q6_k_int8` is memory-bound at about 51B.
The int8 payload is still slightly faster per cycle, but the native format
uses about 23% less traffic (6.625 vs 8.625 bits/weight) and has a higher
full-device throughput estimate.

The largest single win was processing both 8-column records of each transport
tile together, sharing input vectors and exposing two independent mmul
accumulators. The second win was grouping four row tiles per decode: sharing
the high-plane load and masks cuts the payload load count from eight to six per
64-row chunk and gives the independent vector chains enough work to cover the
emulated-FP32 latency. The exact scale path is unchanged except that host
repack now pre-splits `d` into an exact bf16 hi/lo pair (including canonical
subnormals); each chunk's int8 scales are multiplied by `d` and split into
hi/mid/lo terms once, and the three-term product is applied with bf16
multiplies. This remains exact for canonical scales up to 127.

Instruction memory: native `q6_k` uses 4992 of the 16384-byte AIE2P
compute-tile program memory (30.5%), `q6_k_int8` uses 4368 bytes (26.7%) with
its chunk loop rolled, `q4_0_hilo` uses 3632 bytes (22.2%), `q4_0` uses 4048
bytes (24.7%), `q4_0_packed16` uses 3808 bytes (23.2%), and `q8_0` uses 7168
bytes (43.8%).

Rejected or alternate approaches:

- host-precomputing all three bf16 scale terms: 8239 cycles on one core on the
  old record layout, but 2816-byte records (1.375 bytes/weight), worse at the
  32-core memory budget;
- applying the exact scale after the mmul: 14258 cycles and max abs error 5.0;
- dropping the third scale term for native Q6_K: about 3% faster unpack and
  9% faster mmul, but not exact for canonical scales up to 127 (five random
  seeds produced up to 1.0 absolute error);
- reusing the accumulator directly in the split: neutral;
- fp32 scale plus emulated float `aie::mul`: exact but much slower;
- four column tiles: memory-bound, extra ILP does not help.

Q8_K / Q8_0 comparison: canonical Q8_K without `bsums` is 260 bytes per 256
weights (8.125 bits/weight, 1.016 bytes/weight), about 6% smaller than the
exact int8 Q6_K layout (8.625 bits/weight). It replaces per-16 scales with one
fp32 scale per 256 weights, so conversion is lossy. Q8_0 is already implemented,
slightly smaller (1.0625 bytes/weight), and has finer per-32 scales. If a lossy
path is acceptable, Q8_0 is the better target than Q8_K.

### Q8_0

Canonical Q8_0 is 8.5 bits/weight. The internal record is 256 rows x 8 columns,
2176 bytes; two records form a 4352-byte 256x16 transport tile:

```text
2048 bytes  int8 payloads (32 tiles of 64 bytes)
 128 bytes  per-block canonical fp16 d (8 blocks x 8 columns)
```

Payloads come first so every 64-byte tile load is 64-byte aligned. Each tile
is one 8x8 row-major matrix, so decode is `load 64 signed bytes -> int8 vector
-> bf16 vector`, then `q * d`. An fp16 `d` needs only two bf16 terms; the
exact exponent-add split produces `hi + lo` with no rounding, and the two-term
accumulation is one bf16 rounding. Four blocks (32 columns) of canonical fp16
scales are decoded per batch.

Results at the measured 1.8 GHz H clock:

| shape | cycles | cycles/tile | weights/cycle | compute-implied 32-core GB/s |
| --: | --: | --: | --: | --: |
| 256x64 | 7707 | 240.844 | 2.126 | 130.1 |
| standalone unpack 256x64 | 2920 | 91.250 | 5.611 | - |

What helped: payload-first 64-byte alignment, signed int8 to bf16 without a
centering bias, two-term `d`, `broadcast_vector` for scale expansion (about
9%), a 256-row record instead of 128 (about 10%), and two-column fusion with
one accumulator per column. Four columns was about 65% slower from register
pressure.

Q8_0 is the fastest format per packed byte; Q4_0 is faster per weight. Its
constant-dequantizer profile is about 3786 cycles of mmul/loop overhead per
256x64, so it is limited more by mmul/input scheduling than payload decode. The standalone unpack test is limited to 64 columns by the 64 KiB
output buffer; the fused kernel works at 64/128 columns. 256 columns need about
68 KiB of packed weights and do not fit the current single-tile harness.

## Testing and exactness

### Unprefixed exactness rule

A quantization without a variant suffix (`q4_0`, `q6_k`, `q8_0`) must keep the
canonical tile size and be exact over the canonical format's finite value
range:

- `tile_bytes == ceil64(4096 * canonical_bits_per_weight / 8)` for a 256x16
  tile: Q4_0 2304, Q6_K 3360 -> 3392, Q8_0 4352;
- unpacked bf16 weights equal the canonical dequantization bit-for-bit for
  every quant code and every finite fp16 scale, including subnormals and zero.
  Non-finite scales are out of scope. A format that grows the tile or relaxes
  exactness takes a name (`q4_0_hilo`, `q6_k_int8`, `q4_0_packed16`).

Current status: the size clause holds for all three. Exactness holds for Q4_0
and Q6_K; Q8_0 was the exception until its pre-biased path was replaced by the
exact exponent-add split. The size clause is only pinned by per-format
`static_assert`s; nothing checks it against the canonical formula yet.

- Keep a standalone exact unpack test as format/layout conformance. It catches
  scale, packing, and dequant errors that an end-to-end mmul tolerance test can
  hide. The standalone loops are compiled only under
  `QUANTS_BUILD_UNPACK_TEST`; their cycle counts are diagnostic and should not
  drive format selection.
- Keep the mmul test focused on end-to-end fused correctness.
- Share scale/dequant helpers between the reference unpack and fused mmul so
  the tests and production path cannot drift; only the outer standalone loop
  is test-only.
- Q6_K `mmul.py --cols 256` exceeds the single-tile harness memory (about
  70 KiB of int8-packed weights plus inputs/outputs); use 64 or 128 columns
  until the stream is split.
- The Q6_K generator uses scales of 1-32, but canonical scales can reach 127.
  A two-term bf16 split can still pass the end-to-end test because final bf16
  rounding hides the residual. Add a direct split-residual test if the scale
  representation changes.
- Code-size figures in `README.md` are current; older per-format `.text` numbers
  in this file may predate the host-scale and int8-payload changes.

## Target model context

`gemma-4-E2B_q4_0-it.gguf` is the first end-to-end target: 35 dense blocks,
about 1.381 GB/token of active NPU weights with the canonical `q4_0` repack
and the 6.625-bit native `q6_k` output head, or roughly 39.8 tok/s at 55 GB/s.
With `q4_0_hilo` those rise to 1.497 GB and 36.7 tok/s. The 8.625-bit
`q6_k_int8` head instead gives about 1.482 GB and 37.1 tok/s but is
memory-bound at about 51B weights/s versus about 66B for native `q6_k`. The
traffic figures are memory-bound and therefore clock-invariant. The breakdown
and bring-up checklist live in `TODO.md`; the performance summary lives in
`README.md`.
