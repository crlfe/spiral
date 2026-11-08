# NPU optimization notes

Practical lessons from optimizing the `src/quants` Q4_0 kernels on XDNA2 /
AIE2P using Peano and IRON.

## Measurement

- Always measure with `event0()` / `event1()` and the trace stack.
- Use **one outer event pair per kernel**. Per-chunk or per-tile events distort
  the measurement and cost 150-200 cycles by themselves.
- Keep the trace buffer large enough for long kernels. 4 KiB was enough for
  64 columns but overflowed at 256 columns; 16 KiB is safe.
- Cycle counts are deterministic enough that single runs are useful, but repeat
  after code-layout changes.
- Report logical weights/cycle for one core:
  `weights/cycle = weights_rows * weights_cols / cycles`.
- Packed Q4_0 throughput assumes 4.5 bits/weight, or 0.5625 bytes/weight:
  `GB/s per core = weights/cycle * 0.5625 * 1.5`.
- For the full NPU, multiply by 32 compute cores, assuming ideal scaling.
- 60 GB/s requires about 71.1 weights per device clock, or
  **2.22 weights/core-cycle**.
- One AIE2 VLIW bundle is 16 bytes.

## Q4_0 packed layout

- Canonical Q4_0 is 4.5 bits/weight.
- One 64-row x 8-column record is 288 bytes:
  - 2 x 16 bytes of fp16 scales
  - 8 x 32 bytes of int4 tile payloads
- Store all scales first, then all payloads. This keeps the record at 288 bytes
  and makes every 32-byte tile payload 32-byte aligned.
- The old 144-byte unit layout left the second 32-row block only 16-byte
  aligned, forcing `load_unaligned_v<32>(..., 16)`. Reordering the record made
  `load_v<32>` valid and improved both unpack and fused mmul.
- Host repack must stay plain bytes; `aie::vector` types cannot be included in
  host-compiled headers.
- Device-only layout views are useful for static assertions, but hot-path
  struct member access can cost more than vector-pointer access.

## Kernel structure

The best fused structure found so far:

- one `unpack_mmul<weights_rows, weights_cols>` implementation per format;
- process **two independent column tiles** at a time;
- share the input tile between both columns;
- use two mmul accumulators;
- fully unroll the 8-tile inner loop;
- load one 32-byte int4 payload per tile;
- dequantize both columns, then issue both mmuls;
- store outputs once per column tile.

What helped:

- inlining unpack directly into the fused kernel instead of using cursor
  objects;
- reordering the packed record to align tile payloads;
- 32-byte aligned `load_v<32>`;
- `aie::vector` pointers for input, output, and payload tiles instead of
  element-pointer arithmetic;
- sharing the scale/dequant helpers with the standalone conformance test.

What did not help:

- partial unrolling (`unroll_count(4)` was much slower);
- removing the unroll pragma entirely;
- `noinline` on `next()` or the mmul helper;
- four columns processed concurrently;
- hoisting accumulators across records;
- multiple accumulators per column;
- pointer increments instead of indexed addresses;
- `__restrict` on the fused kernel pointers;
- `-O3` or `-Os` instead of the default optimization;
- copying the input tile into a local before both mmuls;
- dequantizing all tiles before issuing mmuls.

## Scale handling

- Split fp16 scales into bf16 hi/lo components once per 32-row block.
- Expand the 8 scales to 64 lanes per block, not per tile.
- The 64-lane active scale cache is the best balance found.
- 32-lane scales plus per-tile expansion, 8-lane scales plus per-tile
  expansion, and reload/split per tile were all slower.
- Duplicating scales in the packed stream does not help; scales can already be
  reloaded from the existing layout, and duplication only increases bytes.
- Moving scale multiplication after the mmul was mathematically appealing but
  slower and outside the exact verification tolerance.
- A single-bf16 fast scale mode is about 11-12% faster but has max abs error
  around 2.0, so it is only useful for a future relaxed-accuracy mode.

## Mmul shape

- The current kernel uses `aie::mmul<4, 8, 8>` with bf16 inputs and weights.
- `aie::mmul<4, 4, 8>` is not defined for bf16.
- `aie::mmul<4, 8, 4>` is supported and uses 32 weights, but it doubles the
  number of mmul operations and requires splitting/interleaving output
  columns. It was not pursued after the two-column path proved faster.
- The 128 `vextbcst` instructions in the fused loop are inherent to the
  current `4x8x8` mmul formulation; they also appear in the standalone mmul.

## Register spills

- The two-column direct kernel reduced the stack frame from 0x940 to 0x4c0 and
  vector stack stores from 40 to 18.
- Spills still matter. The hot loop keeps two accumulators, two 64-lane scale
  pairs, and two payload streams live.
- Splitting into per-column `noinline` helpers reduced pressure but lost too
  much ILP.
- Struct-based record views and member access regressed the fused kernel by
  about 4.6%; vector-pointer access is faster in the hot path.
- Reducing spills by giving up the two-column ILP is not worth it. The
  two-column version is faster despite spilling more.

## Current results

Q4_0, 64 rows, exact:

| weights_cols | cycles | cycles/tile | weights/cycle | 32-core packed GB/s |
| --: | --: | --: | --: | --: |
| 64 | 2400 | 300.000 | 1.707 | 46.08 |
| 128 | 4764 | 297.750 | 1.720 | 46.43 |
| 256 | 9268 | 289.625 | 1.768 | 47.73 |

- Q4_0 standalone exact unpack: 1499 cycles.
- Fast single-bf16 mode: 2119 cycles at 64 columns, about 52 GB/s across 32
  cores, but not accurate enough for the exact path.

## Instruction footprint

Using 16-byte VLIW bundles:

| Component | Bytes | Bundles | KiB |
| :-- | --: | --: | --: |
| Q4_0 fused kernel, 64 cols | 2112 | 132 | 2.06 |
| Q4_0 fused wrapper | 256 | 16 | 0.25 |
| Q4_0 fused total, 64 cols | 2368 | 148 | 2.31 |
| Q4_0 fused total, 128/256 cols | 2048 | 128 | 2.00 |
| Q4_0 standalone unpack | 736 | 46 | 0.72 |
| Q6_K mmul skeleton | 192 | 12 | 0.19 |

The 128- and 256-column kernels have the same code size; only the 64-column
case is more heavily unrolled.

## Testing

- Keep a standalone exact unpack test as format/layout conformance. It catches
  scale, packing, and dequant errors that an end-to-end mmul tolerance test can
  hide.
- Keep the mmul test focused on end-to-end fused correctness.
- Share the scale and dequant helpers between the reference unpack and the
  fused mmul so the tests and production path cannot drift.
- Q6_K remains a skeleton; its unpack and mmul tests are expected to fail.
