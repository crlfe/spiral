> Role: reference. Current as of 2026-11-24. History: `git log -- src/funcs/DESIGN.md`.

# Format-independent functions

`src/funcs` holds device kernels that are not quant-format specific. Separation
from `src/program` keeps the FIFO/dataflow shell out of the vector math, and
separation from `src/quants` keeps activation and reduction functions out of the
format/transport tree.

## RMS norm

### Contract

The kernel consumes the fixed program's input-group layout: one group is
`kRowsPerGroup = 4` K-row steps of `kRowStep = 256` values, with
`kVecs = 4` input vectors interleaved in 8-value chunks:

```text
group: [row_step 4][chunk 32][vec 4][8] bf16     4096 values
weight: [chunk][hi 8][lo 8] bf16                 2*K values
```

`kChunksPerGroup = 32`; `kGroupInputElems = 4096`. K=1536 uses two groups; the
second group has two real row steps and two padded ones. Two consecutive
per-chunk `[hi8][lo8]` pairs are contiguous, so `scale_group` fetches both
chunks' weights with one 512-bit load.

### Functions

- `sum_squares_group_at(group_input, base_row, rows_steps, sums[4])` adds one
  acquired group's squared values into four f32 sums, one per input vector.
  `group_input` is exactly one 4096-value group; `base_row` is its first global
  row-step and selects the weight slice. It uses four 32-lane
  `aie::accum<accfloat, 32>` partials and eight chunk loads per iteration so the
  mac chain is not serial.
- `sum_squares_group(input, group, ...)` is the full-K wrapper.
- `finish_r(sums, eps, inv_k, r_hi[4], r_lo[4])` computes
  `r = rsqrt(mean + eps)` and splits `r` into bf16 hi/lo.
- `scale_group_at(group_input, base_row, rows_steps, weight, r_hi, r_lo)` scales
  one acquired group in place. It processes two chunks per iteration: their two
  `[hi8][lo8]` pairs are one 512-bit load, each 8-value weight piece is repeated
  across the four vectors with one `vextbcst.128` (`repeat_lane<>`), and `r` is a
  32-lane vector built once. `r*w` is combined in f32, split to bf16 hi/lo, then
  applied to the activation.
- `scale_group(input, group, ...)` is the full-K wrapper.
- `normalize(data, weight, rows_steps, eps, inv_k)` is the full-K convenience
  path used by the standalone test.

No function acquires a FIFO or reads a config object. The program prelude
wrapper belongs in `src/program/kernels`.

### Precision

The sum is f32. The activation is bf16, `r` and `w` are bf16 hi/lo, and the
scaled activation is stored bf16. The test reference normalizes the same bf16
input in f32.

The multiply order is deliberate: combine `r*w` first, split that product to
hi/lo, then apply it to `x`. Doing `(x*r)*w` rounds the normalized activation
to bf16 before the weight multiply; across five seeds it measured a 0.763 worst
tolerance ratio versus 0.464 for `x*(r*w)`. Do not swap the order to save a few
cycles without re-running that sweep.

Current measured error at K=1536, four vectors, random N(0,1) input and weight:

```text
max abs error         0.0156
worst tolerance ratio 0.464   across seeds 1-5; 0.450 at seed 1
                             (policy: max(1e-3, 6e-3 * row_scale))
```

Do not widen this policy to hide a regression. If a future change needs a
different policy, name it and measure the final matmul output.

### Cost

One compute tile, K=1536 (two groups), trace around each pass. Numbers live in
`docs/measurements.md` § 3:

| pass | cycles |
| :-- | --: |
| sum of squares | 765 |
| in-place scale | 3904 |
| **total** | **5315** |
| `.text` | 3600 B |
| max stack frame | 384 B |

At 1408 MHz the full norm is ~3.8 us, below the CPU's ~5 us per norm. The
program still replicates the norm on every compute tile because every compute
sees all K, so the prelude is close to viable.

Both passes are now issue-bound rather than dominated by one dependency chain:
the scale loop retires about one VLIW word per cycle at ~14 vector operations
per 32-value chunk. Further gains need fewer vector operations (a layout or
dataflow change), not more unrolling.

What was tried:

- native 32-lane scale (repeat the 8-value weight chunk and build a 32-lane
  `r`): reduced the old 74k-cycle version to 48k;
- per-vector f32 partial vectors in the sum: 48k -> 35k;
- a single 32-lane `accfloat` partial accumulated with `aie::mac`: 35k -> 10.1k;
- `aie::broadcast_vector<kChunk>` for the repeated weight chunk: scale
  8.3k -> 7.5k, total 10.1k -> 9.3k;
- per-chunk `[hi8][lo8]` weight packing with one 16-lane load: neutral on
  cycles, but fewer loads and slightly smaller `.text`; kept;
- `unroll_count(4)` on the chunk loops: neutral; `8` was neutral and grew
  `.text`, so 4 stays;
- two chunks per iteration with interleaved statements: scale 7.5k -> 4.2k.
  The one-chunk loop was latency-bound; four unrolled copies still ran
  sequentially in the emitted code;
- four f32 partials in the sum with eight loads per iteration: 1.06k -> 765;
- `repeat_lane<>` (one `vextbcst.128` per constant lane) instead of
  `broadcast_vector` (which shifts first): scale 4.0k -> 3.9k;
- one 512-bit load for the two chunks' `[hi8][lo8]` pairs: neutral on cycles,
  half the weight loads; kept;
- bf16 square plus bf16 reduce: faster-looking but inaccurate (worst tolerance
  ratio 5.16), so rejected;
- a bf16-only scale apply: 7.3k cycles but worst ratio 1.169, rejected;
- hi/lo scale product followed by a bf16-only apply: 8.8k cycles and worst
  ratio 0.850; too close to the policy limit, so the full hi/lo apply stays;
- `(x*r)*w` order instead of `x*(r*w)`: 7.5k cycles total, but worst ratio
  0.763 across five seeds versus 0.464 for `x*(r*w)`; rejected for margin;
- `aie::msc(scale, one, hi)` for the low term (one op instead of widen+sub):
  exact, but 2% slower; rejected;
- flattening the sum over rows into one chunk run: 765 -> 853; rejected;
- eight sum partials: 1822 cycles and a 2688 B stack frame; rejected;
- applying the scale with an f32 multiply: f32 vector arithmetic is emulated
  on AIE2P with several bf16 products, so the hand-written two-term apply is
  the cheaper form;
- pointer-increment addressing in the scale loop: neutral on cycles, larger
  stack frame; rejected.

The remaining cost is the scale pass, and it is now issue-bound rather than
stalled: the loop retires about one VLIW word per cycle at ~14 vector operations
per chunk. The next lever is structural, not unrolling: the weight broadcast
and the per-chunk scale split both exist because the four interleaved rows share
one 8-value weight chunk. A row-major activation layout would remove the
broadcast; folding `gamma` into the quantised weights on the host
(`(x*g)W = x*(gW)`) would leave the device applying only the per-row scalar
`r`.

## Testing

`tests/rmsnorm.py` covers:

- the full K=1536 input with a partial final group;
- interleaved four-vector packing and all K chunks;
- f32 reference normalization of the original bf16 input;
- a per-row relative tolerance, not a fixed absolute bound;
- cycles separately for the sum and scale passes, plus `.text` and stack.

Add new functions in their own subdirectory with the same pattern: pure
`*.aie.h`, a thin `.aie.cc` test wrapper, and a Python oracle/benchmark.
