# `src/quants` TODO

Production code lives in `spiral::quants`. Host repack and NPU unpack are
header-only. Optimization findings are recorded in `NOTES.md`.

## Status

| item | state |
| :-- | :-- |
| Q4_0 layout, host repack, exact unpack | implemented |
| Q4_0 fused unpack + mmul, two columns | implemented |
| Q4_0 `tests/unpack.py` | passes, exact bf16 match |
| Q4_0 `tests/mmul.py` | passes, exact mmul match |
| Q6_K layout and Python reference | implemented |
| Q6_K C++ unpack/mmul | skeleton; tests fail as expected |
| NPU-less execution | clear error; NPU required |
| Host header compilation | C++17, no AIE in `repack.h` |
| Test shims | `unpack.aie.cc`, `mmul.aie.cc`, `repack.cc` |

Current Q4_0 exact results, the standalone unpack baseline, and the fast-mode
comparison are recorded in `NOTES.md`.

Reaching 60 GB/s needs about **2.22 weights/core-cycle**, or 71.1 weights per
device clock across 32 cores.

## Next implementation steps

Ordering is by model impact: finish Q6_K first, then Q8_0, then the K-quant
families and conversion controls, then codebook formats. Defer other types until
a target model or benchmark requires them.

### 1. Finish Q6_K

Q6_K has a layout, Python reference, and failing tests. It validates the
four-chunk record model and shared-scale metadata.

- Keep `kRecordRows = 256`, `kUnpackTileRows = 64`, `kChunksPerRecord = 4`,
  `kCommonScaleBytes = 16`.
- Choose the intra-chunk layout explicitly:
  - **Native 6-bit packed:** preserves 6.5625 bits/weight and the provisional
    416-byte chunk (384-byte payload plus 32 bytes of per-column sub-block
    scales).
  - **Int8 repack:** simpler first bring-up, 8 bits/weight (+22% bytes).
- For the native path, remove cross-byte ql/qh interleaving during host repack
  so the kernel only does aligned loads, shifts, and masks.
- Hoist `d * int8_scale` per 16-row sub-block and per column. Reuse the scale
  vector across the four 4x8 payloads in that sub-block.
- Use hi/lo bf16 scales for the accurate path; retain a single-bf16 fast path
  for later comparison.
- Update the Python packed-byte reference only after the layout is fixed.
- Implement the real `q6_k/unpack.aie.h` and `q6_k/mmul.aie.h`; expected result:
  Q6_K `tests/unpack.py` and `tests/mmul.py` pass on a device.

### 2. Q8_0

Q8_0 is the simplest remaining format and the largest remaining element share
in the local model set. Use it as the second non-K format and as a vector
conversion control.

- Canonical block: 32 rows, 34 bytes, fp16 `d` plus 32 int8 values.
- Choose a record depth (64 or 256 rows) that reuses the Q4_0 chunk machinery.
- Host repack: copy scales and transpose int8 values into 4x8 or 8x8 payloads.
- Unpack: convert int8 to bf16, multiply by `d`, and store directly into the
  canonical tile.
- Add `tests/formats/q8_0.py` with independent numpy generation and reference.
- Expected result: exact or near-exact match with power-of-two scales.

### 3. Q4_K and Q5_K

Q4_K and Q5_K are the largest K-quant families in the local models and should
share a scale/min implementation.

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
- Validate against the host reference dequant output for random blocks before
  NPU bring-up.

### 4. BF16 and F16 pass-through

Treat these as conversion controls, not quantized formats.

- Choose a record layout and reuse the existing mmul path.
- Use them to measure pure data movement and bf16 conversion cost.
- Do not add a separate mmul implementation.

### 5. IQ4_XS and IQ4_NL

Both use the 16-entry `kvalues_iq4nl` codebook.

- IQ4_NL: 32-row block, fp16 `d`, 4-bit indices.
- IQ4_XS: 256-row superblock, fp16 `d`, 16-bit `scales_h`, 4-bit low scales,
  and 4-bit indices.
- Add the codebook as shared `constexpr` or static data and account for it in
  the L1/constant budget.
- Dequant: `scale * kvalues_iq4nl[index]`, with IQ4_XS scales reconstructed
  from `scales_l` and `scales_h`.
- Add a lookup-table path to the Python reference first.

### Later formats

Q3_K and IQ3_S share 3-bit/hmask patterns and broaden model support. Q2_K,
IQ2_XXS/XS/S, IQ1_S/M, Q4_1, Q5_1, MXFP4, NVFP4, and TQ formats are deferred
until a target model or benchmark requires them.

## Constraints

- C++17, no exceptions, no RTTI.
- Headers use `#pragma once`; all header functions are `inline` (or
  `constexpr` for compile-time helpers).
- `repack.h` and `layout.h` must be usable by the host loader without AIE
  headers. Device-only code belongs in `.aie.h`.
- `GGML_LOG_*` may be used by callers, not inside these headers.
- Layout and shared constants are namespace-scope `constexpr`, **not**
  `inline constexpr`.
- Constants and functions live in `spiral::quants::<name>`; shared constants
  live in `spiral::quants`.
- Keep the format/chunk API stable while optimizing behind it.
- Host repack byte counts, capacities, strides, and offsets stay `int64_t`.
  NPU-side loop counters and record/chunk indices use `int32_t`; per-launch
  shapes are small enough that this is safe.
- Production headers contain no tracing. Test shims select
  `aie::rounding_mode::conv_even` once and call `event0()` / `event1()`.
- Use device-side vector pointers for hot tile access. Avoid struct member
  access in the hot loop; it regressed the fused kernel.

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
const uint8_t* chunk(const uint8_t* record, int chunk_index);
```

Shared metadata is always at the start of the record. `chunk()` points past
that metadata to a specific chunk; callers pass the record pointer directly
when they need the shared metadata.

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
void unpack_weights_chunk(const uint8_t* record,
                          const uint8_t* packed_chunk,
                          bfloat16* weights_tile);
```

`record` points at the start of the shared metadata. The unpack function writes
the canonical bf16 weights tile directly. Formats call `unpack_weights_chunk`
once per chunk, including single-chunk formats such as Q4_0.

### Fused API

Each format provides `mmul.aie.h` with:

```cpp
template <int weights_rows, int weights_cols>
inline void unpack_mmul(const uint8_t* packed,
                        const bfloat16* inputs,
                        float* outputs);
```

The caller must zero `outputs` and select `conv_even` rounding. Q4_0 processes
two column tiles together; Q6_K is still a skeleton.

## Format constants

### Q4_0

```text
canonical block: 32 rows, 18 bytes
payload: 8 x 16-byte signed-int4 4x8 payloads per 32-row block
scales:  8 fp16 per 32-row block
record:  64 rows, one chunk
layout:  32 bytes of scales, then 256 bytes of payloads
packed:  288 bytes per record
```

The scales-first layout keeps every 32-byte tile payload 32-byte aligned.
`load_v<32>` is valid for all tiles.

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

## Local model inventory

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

## Planned files

```text
src/quants/
  NOTES.md                    optimization lessons and current results
  common/
    k_quant.h                 shared 6-bit K scale/min unpack
    iq4nl.h                   shared IQ4_NL codebook
  q8_0/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
  q4_k/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
  q5_k/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
  bf16/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
  iq4_nl/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
  iq4_xs/
    layout.h
    repack.h
    unpack.aie.h
    mmul.aie.h
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
must keep its own layout/repack/unpack/mmul units so tests and production
kernels share one implementation.

## Shared implementation work

- `common/scale.aie.h` provides fp16x8 loading and hi/lo bf16 scale splitting.
  Reuse it for K-quant and codebook formats that need scale conversion.
- `common/mmul.aie.h` provides the vector mmul tile helper.
- `q4_0/unpack.aie.h` provides shared `load_block_scales64()` and
  `dequantize_tile()` helpers used by both the reference unpack and the fused
  mmul.
- Select `aie::rounding_mode::conv_even` once in the kernel shim; unpack
  helpers assume that mode is already set.
- Add a K-quant scale/min helper for 6-bit packed scales.
- Add a shared codebook helper for IQ formats.
- Keep host repack format-specific; do not leak format layout into
  `common/mmul.aie.h`.
- Add format layout static assertions for:
  - `kTileCols == kMmulWeightsCols`
  - `kUnpackTileRows % kMmulWeightsRows == 0`
  - record and chunk size consistency
  - payload alignment where the format uses 32-byte tile loads
- Keep `layout.h` constants in the `constexpr int NAME = expression;` form
  parsed by `tests/common.py`; expressions may wrap across lines.

## Optimization roadmap

### Q4_0 scale modes

The exact hi/lo path is the default. A single-bf16 fast mode measured about 12%
faster but failed the exact verification tolerance.

- Add a `ScaleMode` compile-time parameter:
  - `exact`: current hi/lo bf16 scale split, bit-identical to float dequant.
  - `fast`: single bf16 scale, lower precision and fewer cycles.
- Keep the current non-template `unpack_weights_chunk()` as the default wrapper
  around `unpack_weights_chunk_impl<ScaleMode::exact>()`.
- Give each named variant an explicit verification policy. The fast variant
  should not silently pass through the exact mmul tolerance.
- Add a named variant selector to the test shims. Do not expose raw template
  arguments or blindly inject user-provided values.
- Report the variant name with cycle results.
- Measure against the current 300 cycles/tile 64x64 exact result.

### Fused unpack and mmul

The production Q4_0 implementation is the direct two-column fused path:

- share the input tile between two column tiles;
- use two independent mmul accumulators;
- load one aligned 32-byte payload per tile;
- fully unroll the eight-tile inner loop;
- use device-side vector pointers for input, output, and payload tiles.

Do not reintroduce cursor objects or struct-based hot-path access unless they
measurably beat the direct path.

### Chunk routing and larger tiles

- Route Q6_K/Q4_K/Q5_K chunks to separate compute tiles only after single-tile
  correctness and timing are stable.
- Compare 64-row chunks against 256-row records for metadata amortization and
  L1 pressure.
- Keep `unpack_mmul<weights_rows, weights_cols>` generic over record and tile
  counts.
- Investigate larger mmul shapes only if the AIE API supports them for bf16.

### Pipelining and memory

- Add double buffering only after single-buffer performance is understood.
- Watch the AIE DMA descriptor limit: a single FIFO output is limited to 16,383
  32-bit words. Large optional shapes must be split by record or chunk.
- Measure packed-weight bandwidth against the 55 GB/s aggregate budget.
- Preserve the existing trace-stack convention and balanced-stack validation.

## Test and benchmark roadmap

- Keep `unpack.py` as the exact format/layout conformance test.
- Keep `mmul.py` as the end-to-end fused correctness test.
- Add a real-model test path that extracts a small slice of a representative
  tensor, repacks it, and compares against the Python reference. This catches
  assumptions that synthetic data may miss.
- Keep the no-NPU error path and invalid-shape rejection tests.
- Benchmark at least:
  - Q4_0 64x64, 64x128, and 64x256, exact and fast scale modes;
  - Q8_0 64x64 and 128x128;
  - Q6_K 256x64;
  - Q4_K/Q5_K 256x64 and 512x64, split output if DMA limits require it.
- Report cycles/tile, cycles/weight, weights/cycle, and implied packed GB/s at
  1.5 GHz with 32 compute cores.
- Record new baselines in `NOTES.md`.

## Definition of done

- [ ] Q6_K native or int8-repacked unpack and mmul pass on a device.
- [ ] Q8_0 unpack and mmul pass on a device.
- [ ] Q4_K and Q5_K host repacks and dequantization match the host reference.
- [ ] Q4_K and Q5_K NPU unpack and mmul pass on a device.
- [x] Q4_0 fused two-column unpack + mmul passes exactly.
- [ ] Q4_0 exact and fast scale modes are exposed through named variants with
      per-variant verification.
- [ ] Invalid shapes are rejected for every format.
- [ ] No format-specific runner or duplicate mmul implementation is introduced.

## Deliberately deferred

- Chunk-to-compute-tile routing for multi-chunk formats.
- Software pipelining and double buffering.
- Larger mmul variants beyond the current 4x8x8 tile.
- IQ2/IQ1/TQ/MXFP4/NVFP4 support.
- Host backend integration beyond the header/API contract.
