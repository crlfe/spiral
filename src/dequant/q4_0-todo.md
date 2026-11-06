# Q4_0 dequant optimization TODO

## Current state

`formats/q4_0/q4_0.cc` unpacks one 16-block Q4_0 tile:

- 16 canonical 18-byte blocks = 8 output columns x 2 reduction blocks.
- 64 reduction values x 8 columns = 512 weights.
- Input is a host-repacked tile of aligned signed-int4 values.
- Output is eight 8x8 bf16 subtiles in `aie::mmul` weights tile layout.
- The kernel is bracketed by `TRACE_ENTER()` / `TRACE_LEAVE()`.

Current measurement (`run.py --format q4_0`):

```text
event0 -> event1: 227 cycles
512 weights / 227 cycles = 2.256 weights/cycle
```

At an assumed 1.4 GHz, one core streams:

```text
288 bytes / (227 / 1.4e9 s) = ~1.78 GB/s
32 cores                    = ~56.8 GB/s
```

At 1.6 GHz the same cycle count corresponds to ~64.9 GB/s aggregate.

This is right at the 55 GB/s total cycle budget at 1.4 GHz (235 cycles/tile)
and leaves about 41 cycles/tile for the mmul at 1.6 GHz (268 cycles/tile).

The kernel compiles to **123 VLIW bundles**, uses no stack frame, and has no
loop branches.

Accuracy note: the fp16 scale is split into hi/lo bf16 components and both
terms are accumulated in f32 before the final bf16 round. On random fp16
scales, the output is bit-identical to the float dequant path. The earlier
single-bf16-scale path is faster (188 cycles) but loses fp16 scale precision.

## Completed steps

Progression:

| step | cycles/tile | weights/cycle | notes |
| :-- | --: | --: | :-- |
| canonical raw Q4_0 blocks | 1940 | 0.264 | unaligned 18-byte blocks, nibble deinterleave |
| aligned signed-int4 host repack | 1528 | 0.335 | 16-byte payloads, no zero-point subtract |
| transposed 4x8 repack | 1317 | 0.389 | no in-kernel transpose, vector scale pattern |
| fully unrolled | 974 | 0.526 | 851 bundles, no stack frame, no loop branches |
| vector scale conversion | 842 | 0.608 | 392 bundles, packed fp16-to-f32 bit trick |
| bf16 conversion + multiply | 188 | 2.723 | 103 bundles, fast approximate path |
| hi/lo bf16 scales | **227** | **2.256** | 123 bundles, matches float scale precision |

### 1. Host repack (done)

Canonical GGUF blocks are repacked once on the host:

- signed `(q - 8)` nibbles in natural K order;
- 16-byte-aligned payloads;
- grouped fp16 scales;
- still exactly 4.5 bits/weight.

This removed unaligned loads, nibble deinterleaving, and the zero-point add.

### 2. Transposed 4x8 layout (done)

The repack now transposes values into 4 reduction x 8 outputs payloads:

```text
per k block: [8 x 16-byte payloads][8 fp16 scales]
```

The kernel emits the weights tile layout directly, with no in-kernel transpose and
no `col_vals` temporary. Each payload is multiplied by a 32-wide scale pattern
`[s0..s7, s0..s7, s0..s7, s0..s7]`.

### 3. Stack-frame and spill removal (done)

The transposed layout and unrolled kernel eliminated the old multi-KiB stack
frame. The current kernel uses no stack and has no loop branches.

### 4. Full unrolling (done)

`#pragma unroll` was added to the scale-load, k-block, and payload loops.
Result: 1317 -> 974 cycles, 0.389 -> 0.526 weights/cycle, 851 bundles.

### 5. Vector scale conversion (done)

`load_scales()` uses a packed fp16-to-f32 bit trick:

```text
f32 = sign | ((exp16 + 112) << 23) | (mantissa << 13)
```

This replaced eight scalar decodes with one vector path. Result: 974 -> 842
cycles, 0.526 -> 0.608 weights/cycle, 392 bundles.

### 6. bf16 conversion and multiply (done)

The kernel now uses `to_float<bfloat16>()` instead of converting through
float. Result: 842 -> 188 cycles, 0.608 -> 2.723 weights/cycle, 103 bundles.

The single-bf16-scale version rounds the fp16 scale to bf16 and is slightly
approximate. It is not in the current kernel; it remains the measured speed
reference.

### 7. hi/lo bf16 scale split (done)

Each fp16 scale is split into `hi_bf16 + lo_bf16`; both products are
accumulated in f32 before the final bf16 round. The split is derived from the
existing f32 scale vector, so only the eight scales per k block go through
f32, not the weight values. Result: 227 cycles, 2.256 weights/cycle, 123
bundles. Output is bit-identical to the float dequant path on random fp16
scales.

This is the current default: it retains fp16 scale precision and still fits
inside the 55 GB/s total cycle budget at 1.4 GHz.

## Cycle budget for 55 GB/s

Assumptions:

- 55 GB/s aggregate across 32 compute tiles.
- Q4_0 is 4.5 bits/weight = 0.5625 bytes/weight.
- One tile is 512 weights / 288 bytes.
- Each core owns 1/32 of the weight stream.

Per-core targets:

| clock | bytes/cycle/core | weights/cycle/core | cycles/weight | cycles per 512-weight tile |
| :-- | --: | --: | --: | --: |
| 1.4 GHz | 1.23 | 2.18 | 0.458 | 235 |
| 1.6 GHz | 1.08 | 1.91 | 0.524 | 268 |

The mmul still has to run in the same loop. For a 64x8 weights tile with 4 input rows, the
mmul is 8 `aie::mmul<4,8,8>` MACs, plus A/C loads and stores. Reserve roughly
20-50 cycles/tile for it.

Practical dequant budget:

| clock | total cycles/tile | mmul reserve | dequant budget |
| :-- | --: | --: | --: |
| 1.4 GHz | 235 | 20-50 | ~185-215 cycles/tile |
| 1.6 GHz | 268 | 20-50 | ~218-248 cycles/tile |

That is about **12-15 cycles per 32-weight Q4_0 block**, or roughly
**2.4-2.8 weights/cycle** for the dequant portion alone.

Milestone ladder:

| milestone | cycles/tile | weights/cycle | approx aggregate at 1.4 GHz |
| :-- | --: | --: | --: |
| current (hi/lo scales) | 227 | 2.26 | ~56.8 GB/s |
| fast bf16-scale path | 188 | 2.72 | ~68.6 GB/s |
| 55 GB/s total budget at 1.4 GHz | 235 | 2.18 | ~55 GB/s |
| 55 GB/s total budget at 1.6 GHz | 268 | 1.91 | ~55 GB/s |

The hi/lo path fits inside the 55 GB/s budget at 1.4 GHz with about 8
cycles/tile left for the mmul. The fast bf16-scale path leaves about 47
cycles/tile. The next question is how much the mmul actually costs and how well
it overlaps.

## Remaining next steps

Ordered by expected value.

### 1. Choose exact vs fast scale path after measuring the mmul

The hi/lo path is exact but leaves only about 8 cycles/tile of headroom at
1.4 GHz before the 235-cycle 55 GB/s budget is consumed by dequant alone.

The single-bf16-scale path costs 188 cycles and leaves ~47 cycles/tile for the
mmul. If end-to-end model accuracy tolerates the scale rounding, that extra
headroom may matter more than the last few bits of scale precision.

Measure the actual mmul cost before choosing. A compile-time switch between
the two paths would make the comparison easy.

### 2. Revisit tile shape

The current tile is 64 reduction x 8 outputs. Sweep:

- 8 blocks = 32 reduction x 8 outputs;
- 16 blocks = 64 reduction x 8 outputs (current);
- 32 blocks = 64 reduction x 16 outputs (two adjacent weights tiles);
- 64 blocks = 128 reduction x 8 outputs.

Larger tiles amortize scale-pattern and loop overhead but increase register
pressure and code size. The 16-block version is a correctness reference, not
necessarily the best throughput point.

If the mmul uses a 2x2 expansion, produce two adjacent N tiles per dequant tile
so the consumer can reuse inputs across two weights tiles.

### 3. Fuse with the mmul loop and overlap

Once the dequant primitive is fast enough:

- Keep the fused per-tile loop: dequantize into a small B buffer and consume it
  immediately.
- Ensure weight FIFO depth is enough to overlap DMA with dequant and mmul.
- Measure the actual mmul cost per 64x8 weights tile; reserve it explicitly in the
  budget.
- Re-test a split dequant/mma design only if the fused loop is limited by
  scheduling; the earlier split prototype was slower.
- Keep the accumulator in f32 and store the final output through the existing
  epilogue seam.

### 4. Compile flags and scheduling

- Try `-O3` instead of the default `-O2`.
- Try `chess_prepare_for_pipelining` / `chess_loop_range`.
- Watch program size and I-cache pressure; the kernel is already fully
  unrolled into 123 bundles.

### 5. Scale format experiments

Grouped fp16 scales and vector fp16-to-f32 conversion are already done.

The remaining option is an f32 scale stream. It is faster to load but raises
the format to 5.0 bits/weight (11% more bytes), which is hard to justify at a
55 GB/s target. Measure only if scale handling is still significant.

### 6. Raw-GGUF fallback

If the model loader cannot repack once at load time, the kernel must consume
canonical 18-byte blocks directly. That path needs:

- an optimized nibble deinterleave at the vector level;
- unaligned or staged payload loads;
- a zero-point correction.

Keep this as a fallback; the repacked path is the fast path.

### 7. int8 x int4 alternative

The earlier int8 x int4 path removed dequantization entirely and was ~1.3x
faster at c4. It is not the same as the bf16 dequant+mmul target, but it is a
useful reference point:

- If the 55 GB/s target is dominated by dequant conversion, native int4 MMA
  may be the only way to approach it.
- Keep it as a separate bucket for decode and compare against the optimized
  bf16 dequant path.

## Measurement plan

- Use `run.py --format q4_0` with the `TRACE_ENTER()` / `TRACE_LEAVE()` macros.
- Keep the trace buffer small (4 KiB) and report cycles/tile and
  weights/cycle.
- Run one NPU process at a time.
- Record cycles for at least a warm-cache and a cold-cache run.
- After each change, record:
  - cycles/tile;
  - weights/cycle;
  - implied GB/s at 1.4 and 1.6 GHz;
  - any change in stack frame or program size.
- Compare traced cycles first, not wall-clock across processes.

## Risks / open questions

- The 55 GB/s target assumes dequant and mmul overlap well. If the mmul loop
  cannot hide behind the weight stream, the dequant budget shrinks further.
- The prior "3 weights/cycle" note for the grouped-scale prototype appears to
  have counted a 64x64 tile (4096 weights), while `dequant_ni` actually
  processed 64x8 (512 weights). The corrected rate is ~0.37 weights/cycle,
  so the 55 GB/s target is materially harder than it looked.
- Host repacking is acceptable only if the model loader can do it once at load
  time. The raw-GGUF fallback remains unoptimized.
- 4.5 bits/weight is fixed; do not trade format size for speed without
  accounting for the bandwidth budget.
