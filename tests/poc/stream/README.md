# Quantized Weight Stream PoC

`stream.py` tests the memory path a quantized matrix multiply would use.  The
NPU network is fixed: weights arrive as a mode-specific byte stream over
fixed-size DMA objects, and each compute tile multiplies a 256x16 tile whose
emulated packed size matches a real quantization format.  The weights are
still bfloat16, so the multiply is emulated; only the byte counts and stream
layout approximate the real kernels.

The outputs are checked against a NumPy reference that models the same
vector-level wrapping.

## Files

| file | role |
| :-- | :-- |
| `stream.py` | design, host packing/reference, CLI, benchmark driver |
| `stream_common.h` | shared helpers (64-byte vector copy) |
| `stream_init.aie.cc` | reset the per-compute stream state |
| `stream_config_decode.aie.cc` | copy the broadcast config into a local buffer |
| `stream_zero.aie.cc` | clear the output object |
| `stream_fused.aie.cc` | drain one L1 block: in-place tiles + carry + mmul |

## Emulation

| mode | active rows | tile bytes | real packed size |
| :-- | --: | --: | --: |
| Q4_0 | 72 | 2304 | 2304 |
| Q6_K | 104 | 3328 | 3360 |
| IQ1_S | 25 | 832 (800 used + pad) | 800 |

Each tile is padded to a 64-byte multiple.  The kernel reads only complete
64-weight vectors and wraps back to the first vector after the last, so the
full 256-row multiply is exercised even when the emulated tile contains fewer
than 256 rows.  A trailing partial 8-row group is not packed or read.

## Geometry

```
one compute tile   256 rows x 16 cols
one group          256 rows x 64 cols  (4 computes)
whole array        256 rows x 512 cols (8 groups)
```

The four compute tiles in a group share one L2 weight object, split into four
L1 children.  Inputs are broadcast once for the whole input matrix (all row
steps), and each group writes one output object covering all bands.  This
removes the per-band input/output FIFO traffic of the earlier design.

## Streaming algorithm

The worker processes one L1 weight block per `stream_fused` call:

- If the next tile is wholly inside the block, it is multiplied directly from
  the transport buffer.  Nothing is copied.
- If the tile straddles the block boundary, only the missing prefix is copied
  into the carry buffer.  The next block completes the carry, the tile is
  multiplied from it, and the carry is reused for the following tile.
- The block is drained tile by tile until the next tile would run past its
  end.  Bytes are never compacted or rearranged inside the work buffer.

State persists across blocks in four words: `carry_len`, `tile_index`, `band`,
and `row_step`.  The position inside the current block is local to the call,
because every call receives a fresh block.

## Host weight fill

The host sends one contiguous stream per compute tile:

```
[group][compute][band][row_step][tile_bytes]
```

Each compute stream is padded only at its end, up to a whole L1 block, so the
padding stays under 2% for the large modes.  The shim weight fill is one BD per
group per repetition:

```text
sizes   = [l2_blocks, computes, hi, lo]
strides = [l1_stream_words, stream_stride_words, lo, 1]
transfer_len = l2_stream_words
```

`sizes[0]` is the hardware repeat dimension; each repeat moves one L1 block
for all computes in the group.  `l1_stream_words` is the L1 block size in
32-bit words.

A shim BD access dimension is limited to `[0:1023]` elements.  A 2 KiB block is
512 words and needs no split; a 4 KiB block is 1024 words and would be one
past the limit.  The fill therefore splits the innermost word dimension into
`hi x lo`, using 2 KiB inner chunks when they divide evenly and 1 KiB chunks
otherwise.  A 4 KiB child becomes `[l2_blocks, 4, 2, 512]`.  `--l1-stream-bytes`
must be a multiple of 1 KiB, which keeps this split exact and simple.

The current host fill supports at most 64 L2 blocks per compute stream; a 1 KiB
L1 child exceeds that for Q4_0/Q6_K and is rejected while packing.

## Config protocol

The config travels as the first broadcast input object and is decoded into a
small per-compute buffer:

| word | value |
| --: | :-- |
| 0 | `tile_bytes` |
| 1 | `tile_vectors` (complete 64-weight vectors) |
| 2 | `row_steps` |
| 3 | `col_steps` (bands) |
| 4 | `vec_steps` |
| 5 | unused |
| 6 | `weights_blocks` (L1 blocks per compute stream) |

The worker loops `weights_blocks` times per repetition and releases the output
object once at the end.

## Kernels

- `stream_init` resets the four state words.
- `stream_config_decode` copies eight 32-bit config words.
- `stream_zero` clears the whole all-band output object.
- `stream_fused` contains the tile drain and an inlined two-accumulator mmul.
  The two column-tile accumulators share each input load, and increment-and-wrap
  counters replace a runtime modulo.  The copy is an explicit 64-byte vector
  loop with a scalar tail.

## Design object and caching

All geometry-dependent values and NumPy buffer types live in the immutable
`StreamDesign` dataclass.  `main()` builds one from the CLI options and passes
it through `run_case`; library callers can build their own:

```python
from tests.poc.stream.stream import EMU_IQ1_S_ROWS, StreamDesign, run_case

design = StreamDesign(l1_stream_bytes=4096)
run_case(design, "iq1", EMU_IQ1_S_ROWS, 2048, 2048, 4, 4, reps=1)
```

The design is a compile-time parameter of `program`, and the kernel `.cc`/`.h`
files are listed in `@iron.jit(source_files=...)`.  Both are part of the JIT
cache key, so a geometry change or a kernel edit invalidates the cached xclbin
automatically; there is no manual version counter to bump.

## Options

```text
--l1-stream-bytes N   L1 block size, multiple of 1024 up to 1023 KiB (2048)
--work-bytes N        work buffer, at least the largest tile (5120)
--l2-depth N          L2 weight FIFO depth (2)
--l1-depth N          L1 weight FIFO depth (2)
--reps N              repetitions per launch (1)
--weight-tile-bytes N override the emulated tile size
--skip-copy           skip the carry copy (verification skipped)
--skip-mmul           skip the mmul call (verification skipped)
```

## Measurements

Best samples at `--reps 64` on the 2048x2048 workload:

| mode | GB/s | % of 57 GB/s peak |
| :-- | --: | --: |
| Q4_0 | 52.8 | 93% |
| Q6_K | 54.6 | 96% |
| IQ1_S | 33.5 | 59% |

These run noisy; representative averages are several GB/s lower.  Q6_K is at
the practical ceiling of the eight-shim-channel topology.  Q4_0/Q6_K residual
time is DMA/task overhead, not the copy or mmul: `--skip-copy --skip-mmul` does
not reach 57 GB/s.  IQ1_S moves too few bytes per tile for its fixed per-tile
cost.

L1 child sweep (best samples, GB/s):

| L1 child | Q4_0 | Q6_K | IQ1_S |
| --: | --: | --: | --: |
| 2 KiB | 52.8 | 54.6 | 33.5 |
| 4 KiB | 51.5 | 54.5 | **37.7** |
| 8 KiB | 50.0 | 53.8 | 39.2 |

A 4 KiB child is the useful point: IQ1_S gains about 4 GB/s from fewer blocks.
At 8 KiB its stream padding grows to 23%, so the average drops despite a higher
best sample.  Q4_0/Q6_K are within noise across these sizes; 2 KiB remains the
default.  L2 depth 3 is the maximum at 2 KiB (deeper fails with DMA BD-ID
exhaustion); IQ1_S preferred deeper L1 queues in earlier sweeps.

## Known limits

- The host fill does not split a compute stream over more than 64 L2 blocks.
- The Python JIT host sequence is specialized on `tile_bytes`, `row_steps`,
  `col_steps`, and `vec_steps`.  Generating the transaction at runtime, as
  `invoke.py` does, would avoid reloading the xclbin when the shape changes.
- The emulation should be replaced by the real Q4_0/Q6_K unpack+mmul kernels;
  the stream and config layout stay the same.
- A native driver and pipelined submissions for multiple operations are still
  TODO.
