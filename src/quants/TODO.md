# `src/quants` implementation TODO

Production code lives in `spiral::quants`. Host repack and NPU unpack are
header-only. Keep the original reference implementation unchanged.

## Current state

The initial Q4_0 pass is complete and verified on XDNA2:

| item | state |
| :-- | :-- |
| Q4_0 layout, host repack, scalar unpack | implemented |
| Q4_0 `tests/unpack.py` | passes, exact bf16 match |
| Q4_0 `tests/mmul.py` | passes, exact mmul match |
| Q6_K layout and Python reference | implemented |
| Q6_K C++ repack/unpack | deliberate no-ops; tests fail on accuracy |
| NPU-less execution | skips cleanly |
| Host header compilation | C++17, no AIE in `repack.h` |

Baseline Q4_0 64x64 result: **804,045 cycles for 8 tiles**, or
**100,506 cycles per 512-weight tile**. The earlier vectorized Q4_0 path
was roughly **227 cycles per 512-weight tile**, so scalar unpack is currently
dominating by a large margin. Vectorization is the largest single optimization
opportunity.

Baseline Q4_0 fused result: unpack **803,790 cycles**, mmul **1,712 cycles**.
The mmul is already cheap relative to unpack.

## Constraints that still apply

- C++17, no exceptions, no RTTI.
- Headers use `#pragma once`; all header functions are `inline` (or
  `constexpr` for compile-time helpers).
- `repack.h` must be usable by the host loader without AIE headers.
- `GGML_LOG_*` may be used by callers, not inside these headers.
- Layout and shared constants are namespace-scope `constexpr`, **not**
  `inline constexpr`.
- Constants and functions live in `spiral::quants::<name>`; shared constants
  live in `spiral::quants`.
- Keep the format/chunk API stable while optimizing behind it.
- Host repack byte counts, capacities, strides, and offsets stay `int64_t`.
  NPU-side loop counters and record/chunk indices use `int32_t`; per-launch
  shapes are small enough that this is safe.
- Do not modify the original reference implementation.

## Naming and canonical layouts

Use vector/length names for inputs and outputs, and rows/cols only for weights:

- `inputs_vecs` = M = 4, the number of input vectors.
- `inputs_len` = K, the length of each input vector in scalar elements.
- `weights_rows` = K, the same dimension as `inputs_len`.
- `weights_cols` = N, the number of output values per vector.
- `outputs_vecs` = M = 4.
- `outputs_len` = N.

Canonical layouts:

```text
inputs:  [inputs_vecs][inputs_len]
weights: [weights_rows][weights_cols]
outputs: [outputs_vecs][outputs_len]
```

GGUF stores the weight matrix column-major: each of the `weights_cols` columns
is `weights_rows` contiguous scalar values.

One `aie::mmul<4,8,8>` tile is:

```text
inputs:  [weights_rows_tile][kMmulInputsVecs][kMmulInputsLen] bf16
weights: [weights_rows_tile][kMmulWeightsRows][kMmulWeightsCols] bf16
outputs: [kMmulOutputsVecs][kMmulOutputsLen] f32, accumulated
```

`kMmulInputsLen` and `kMmulWeightsRows` are the same shared dot-product
dimension. The outer `weights_rows_tile` index in the inputs layout tiles that
dimension; it is not an `inputs_vecs` tile.

## Record/chunk contract

Separate repack record depth from per-compute-tile unpack depth:

- `kTileCols = 8`, equal to `spiral::quants::kMmulWeightsCols`.
- `kUnpackTileRows` rows handled by one unpack call.
- `kRecordRows` rows in one repacked record.
- `kChunksPerRecord = kRecordRows / kUnpackTileRows`.
- `kCommonScaleBytes` metadata shared by all chunks in a record.
- `kPackedChunkBytes` per-chunk payload.
- `kPackedRecordBytes = kCommonScaleBytes + kChunksPerRecord * kPackedChunkBytes`.

Helpers in `spiral::quants::<name>`:

```cpp
const uint8_t* common_scales(const uint8_t* record);
const uint8_t* chunk(const uint8_t* record, int chunk_index);
```

`common_scales()` points at the start of the record's shared metadata;
`chunk()` points past that metadata to a specific chunk.

### Repack API

```cpp
constexpr int64_t record_count(int64_t weights_rows,
                               int64_t weights_cols);
constexpr int64_t packed_bytes(int64_t weights_rows,
                               int64_t weights_cols);

int64_t repack_weights(const uint8_t* src,
                       int64_t src_col_stride_bytes,
                       uint8_t* dst,
                       int64_t dst_capacity,
                       int64_t weights_rows,
                       int64_t weights_cols);
```

Destination order:

```text
[weights_cols_tile][weights_rows_record][packed_record]
```

The record index is:

```text
weights_cols_tile * (weights_rows / kRecordRows) + weights_rows_record
```

`src_col_stride_bytes` supports views with padding. `packed_bytes()` is
independent of the source size so a format may expand or shrink during repack.

There is no tail or padding path. Invalid dimensions or insufficient capacity
return `0`.

### Unpack API

```cpp
void unpack_weights_chunk(const uint8_t* common_scales,
                          const uint8_t* packed_chunk,
                          bfloat16* weights_tile);
```

The unpack function writes the canonical bf16 weights tile directly. Formats
with `kChunksPerRecord == 1` may also provide:

```cpp
void unpack_weights_tile(const uint8_t* record, bfloat16* weights_tile);
```

Multi-chunk formats use `unpack_weights_chunk` only.

## Current format constants

### Q4_0

```text
canonical block: 32 rows, 18 bytes
payload: 8 x 16-byte signed-int4 4x8 payloads
scales:  8 fp16, one per column, per 32-row block
record:  64 rows, one chunk, no common scales
packed:  288 bytes per record
```

### Q6_K

```text
canonical block: 256 rows, 210 bytes
ql[128], qh[64], scales[16] int8, d fp16
record:  256 rows, four 64-row chunks
common:  16 bytes, eight fp16 d values
packed:  416 provisional bytes per chunk, 1680 per record
```

The Q6_K intra-chunk packing is still deferred. The provisional sizes equal the
unrepacked Q6_K bytes for a 256-row x 8-column region; the next implementation
must choose an explicit layout.

## Format support roadmap

### Local model inventory

Type counts across the current target model set:

| type | tensors | elements | notes |
| :-- | --: | --: | :-- |
| Q8_0 | 820 | 39.85G | dominant in Q8_K_XL and DFlash Q8_0 models |
| Q4_K | 233 | 16.32G | dominant in UD-Q4_K_XL and Q4_K_M models |
| Q5_K | 231 | 15.31G | dominant in UD-Q4_K_XL models |
| Q4_0 | 563 | 15.06G | implemented |
| BF16 | 581 | 12.96G | Gemma E2B and control paths |
| Q6_K | 70 | 6.56G | embeddings/output tensors and K-quant tails |
| IQ4_XS | 70 | 3.13G | UD-Q4_K_XL |
| IQ4_NL | 6 | 0.28G | UD-Q4_K_XL |
| Q3_K | 3 | 0.12G | UD-Q4_K_XL |
| IQ3_S | 1 | 0.04G | UD-Q4_K_XL |
| F16 | 1 | 0.01G | control |
| F32 | 3330 | 0.09G | mostly small/metadata tensors |

### Priority

1. **Q6_K real implementation.** Finish the existing stub because it already
   has a layout, Python reference, and failing tests. It validates the
   four-chunk record model and shared-scale metadata.
2. **Q8_0.** Simplest remaining format and the largest element share in the
   local model set. It should be the second non-K format and a useful vector
   conversion control.
3. **Q4_K and Q5_K.** Largest K-quant families in the local models. Implement
   them together with shared K-scale/min helpers.
4. **BF16 and F16 pass-through controls.** Needed for Gemma E2B and for
   isolating conversion costs from dequantization costs. F32 is likely limited
   to reference/small tensors.
5. **IQ4_XS and IQ4_NL.** Codebook formats present in UD-Q4_K_XL. Implement
   after the K-scale infrastructure.
6. **Q3_K and IQ3_S.** Low local usage, but they share 3-bit/hmask patterns and
   broaden model support.
7. **Other host-supported quantized types later:** Q2_K, IQ2_XXS/XS/S, IQ1_S/M, Q4_1,
   Q5_1, MXFP4, NVFP4, and TQ formats. Do not implement until a target model or
   benchmark requires them.

## Next implementation steps

### 1. Finish Q6_K

- Keep `kRecordRows = 256`, `kUnpackTileRows = 64`, `kChunksPerRecord = 4`,
  `kCommonScaleBytes = 16`.
- Decide the intra-chunk layout explicitly. Two candidates:
  - **Native 6-bit packed:** preserves 6.5625 bits/weight and the provisional
    416-byte chunk (384-byte payload plus 32 bytes of per-column sub-block
    scales).
  - **Int8 repack:** simpler first bring-up, 8 bits/weight (+22% bytes).
- For the native path, the host repack should remove cross-byte ql/qh
  interleaving so the kernel only does aligned loads, shifts, and masks.
- Hoist `d * int8_scale` per 16-row sub-block and per column. Reuse the scale
  vector across the four 4x8 payloads in that sub-block.
- Use hi/lo bf16 scales for the accurate path; retain a single-bf16 fast path
  for later comparison.
- Update the Python packed-byte reference only after the layout is fixed.
- Expected result: Q6_K `tests/unpack.py` and `tests/mmul.py` pass on a device.

### 2. Q8_0

- Canonical block: 32 rows, 34 bytes, fp16 `d` plus 32 int8 values.
- Choose a record depth (64 or 256 rows) that reuses the Q4_0 chunk machinery.
- Host repack: copy scales and transpose int8 values into 4x8 or 8x8 payloads.
- Unpack: int8 to bf16, multiply by `d`, store directly into the canonical tile.
- Add `tests/formats/q8_0.py` with independent numpy generation and reference.
- Expected result: exact or near-exact match with power-of-two scales.

### 3. Q4_K and Q5_K

- Both use 256-row superblocks with eight 32-row groups and 6-bit packed
  scale/min metadata (`get_scale_min_k4`).
- Record: 256 rows x 8 columns.
- Hoist per-column superblock `d` and `dmin` into common record metadata.
- Keep per-chunk 6-bit scale/min groups for the two 32-row groups in each
  64-row chunk.
- Dequant formula: `d * scale * q - dmin * min`.
- Q5_K adds the `qh` high bit; reuse the Q4_K scale/min path.
- Add a shared K-quant helper for unpacking 6-bit scale/min pairs from the
  12-byte scale block.
- Validate against the host reference dequant output for random blocks before NPU
  bring-up.

### 4. BF16 and F16 pass-through

- Treat these as conversion controls, not quantized formats.
- Choose a record layout and use the existing mmul path.
- Use them to measure pure data movement and bf16 conversion cost.
- Do not add a separate mmul implementation.

### 5. IQ4_XS and IQ4_NL

- Both use the 16-entry `kvalues_iq4nl` codebook.
- IQ4_NL: 32-row block, fp16 `d`, 4-bit indices.
- IQ4_XS: 256-row superblock, fp16 `d`, 16-bit `scales_h`, 4-bit low scales,
  and 4-bit indices.
- Add the codebook as shared `constexpr` or static data and account for it in
  the L1/constant budget.
- Dequant: `scale * kvalues_iq4nl[index]`, with IQ4_XS scales reconstructed
  from `scales_l` and `scales_h`.
- Add a lookup-table path to the Python reference first.

## Expected new files

```text
src/quants/
  common/
    scale.h                 fp16x8 load and hi/lo bf16 scale helpers
    k_quant.h               shared 6-bit K scale/min unpack
    iq4nl.h                 shared IQ4_NL codebook
  q8_0/
    layout.h
    repack.h
    unpack.h
  q4_k/
    layout.h
    repack.h
    unpack.h
  q5_k/
    layout.h
    repack.h
    unpack.h
  bf16/
    layout.h
    repack.h
    unpack.h
  iq4_nl/
    layout.h
    repack.h
    unpack.h
  iq4_xs/
    layout.h
    repack.h
    unpack.h
  tests/
    conformance.py
    formats/
      q8_0.py
      q4_k.py
      q5_k.py
      bf16.py
      iq4_nl.py
      iq4_xs.py
```

Exact filenames may change if formats share a common layout, but each format
must keep its own layout/repack/unpack units so tests and production kernels
share one implementation.


## Shared implementation work

- Port or recreate the useful parts of the earlier scale helper into
  `common/scale.h`:
  - fp16x8 load and hi/lo bf16 scale splitting.
- Use `aie::scoped_mode rounding{aie::rounding_mode::conv_even};` around bf16
  conversions. The default tile rounding mode truncates; `scoped_mode` restores
  the previous mode on scope exit.
- Add a K-quant scale/min helper for 6-bit packed scales.
- Add a shared codebook helper for IQ formats.
- Keep host repack format-specific; do not leak format layout into `common/mmul.h`.
- Add format layout static assertions for:
  - `kTileCols == kMmulWeightsCols`
  - `kUnpackTileRows % kMmulWeightsRows == 0`
  - record and chunk size consistency
- Keep all `layout.h` constants one declaration per line so `tests/common.py`
  can keep parsing them.

## Optimization roadmap

### Q4_0 vectorized unpack

This is the highest-value next optimization.

- Load the 16-byte packed payload as `aie::vector<int4, 32>` and convert with
  `aie::to_float<bfloat16>`.
- Load the eight fp16 scales once per 32-row block, not once per payload.
- Split scales into bf16 hi/lo components and use a multiply-plus-mac path, as
  in the earlier vectorized reference.
- Store directly into 8x8 mmul tiles with `aie::store_v`.
- Compare against the current 100,484 cycles/tile baseline and against the
  earlier vectorized Q4_0 reference (~227 cycles/tile).

### Fused unpack and mmul

- Fuse at chunk level first: unpack one 64x8 chunk, then immediately run
  `mmul_weights_tile<kUnpackTileRows / kMmulWeightsRows>`.
- Avoid materializing multiple chunks at once.
- Keep separate `TRACE_ENTER/LEAVE` scopes for unpack and mmul so stage timing
  remains visible.
- Measure whether the current full-chunk materialization costs anything before
  changing the public API.

### Tile-at-a-time cursor experiment

Only after the vectorized chunk path is measured:

- Prototype an internal `UnpackCursor` that holds the chunk pointer, common
  scale pointer, tile index, and cached decoded scales.
- Use it to generate one 8x8 tile at a time for a fused mmul loop.
- Keep `unpack_weights_chunk()` as the public correctness reference; implement
  it as a loop over the cursor if the experiment wins.
- Do not expose the cursor publicly unless a pipelined or multi-tile consumer
  demonstrably needs it.
- Add a test that concatenating cursor tiles equals `unpack_weights_chunk()`.

### Chunk routing and larger tiles

- Route Q6_K/Q4_K/Q5_K chunks to separate compute tiles only after single-tile
  correctness and timing are stable.
- Compare 64-row chunks against 256-row records for metadata amortization and
  L1 pressure.
- Keep `mmul_weights_tile<weights_rows_tiles>` generic over tile count.

### Pipelining and memory

- Add double buffering only after single-buffer performance is understood.
- Watch the AIE DMA descriptor limit: a single FIFO output is limited to 16,383
  32-bit words. Large optional shapes must be split by record or chunk.
- Measure packed-weight bandwidth against the 55 GB/s aggregate budget.
- Preserve the existing trace-stack convention and balanced-stack validation.

## Test and benchmark roadmap

- Add a `tests/conformance.py` or per-format fixture that compares dequantized
  values against the host reference dequant output for random canonical blocks.
- Add a real-model test path that extracts a small slice of a representative
  tensor, repacks it, and compares against the Python reference. This catches
  assumptions that synthetic data may miss.
- Keep the no-NPU skip path and invalid-shape rejection tests.
- Benchmark at least:
  - Q4_0 64x64 and 128x128;
  - Q8_0 64x64 and 128x128;
  - Q6_K 256x64;
  - Q4_K/Q5_K 256x64 and 512x64, split output if DMA limits require it.
- Report cycles/tile, cycles/weight, weights/cycle, implied GB/s at 1.4 and
  1.6 GHz, and separate unpack/mmul stages from the trace stack.
- Record a baseline table in this file or a results file after each format.

## Definition of done for the next pass

- [ ] Q6_K native or int8-repacked unpack and mmul pass on a device.
- [ ] Q8_0 unpack and mmul pass on a device.
- [ ] Q4_K and Q5_K host repacks and dequantization match the host reference.
- [ ] Q4_K and Q5_K NPU unpack and mmul pass on a device.
- [ ] Q4_0 vectorized unpack removes the scalar baseline bottleneck.
- [ ] Q4_0 mmul stage remains correct after unpack optimization.
- [ ] Invalid shapes are rejected for every format.
- [ ] No format-specific runner or duplicate mmul implementation is introduced.
- [ ] The original reference implementation remains unchanged unless a future
      task explicitly requests it.

## Deliberately deferred

- Public tile-at-a-time unpack cursor API.
- Chunk-to-compute-tile routing for multi-chunk formats.
- Software pipelining and double buffering.
- Larger mmul variants and paired column tiles.
- IQ2/IQ1/TQ/MXFP4/NVFP4 support.
- Host backend integration beyond the header/API contract.
