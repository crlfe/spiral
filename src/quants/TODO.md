# `src/quants` TODO

Production code lives in `spiral::quants`. Host repack and NPU unpack are
header-only. Current performance is summarized in `README.md`; detailed
technique and benchmark findings are in `NOTES.md`.

## Status matrix

| format | layout | host repack | exact unpack | fused mmul | tests | current best | next |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Q4_0 | done | done | done | done | exact | 1.66 w/c | multi-core |
| Q4_0_hilo | done | done | done | done | exact | 2.26 w/c | multi-core |
| Q6_K | done | done | done | done | exact | 0.67 w/c | further optimize native decode |
| Q6_K_int8 | done | done | done | done | exact | 1.61 w/c | multi-core, optional variants |
| Q8_0 | done | done | done | done | exact | 1.99 w/c | multi-core |
| Q4_K | not started | - | - | - | - | - | plan below |
| Q5_K | not started | - | - | - | - | - | share Q4_K scale/min |
| BF16 / F16 | not started | - | - | - | - | - | pass-through controls |
| IQ4_NL | not started | - | - | - | - | - | codebook LUT |
| IQ4_XS | not started | - | - | - | - | - | codebook LUT + scales |
| Q3_K / IQ3_S | deferred | - | - | - | - | - | 3-bit/hmask |
| Q2_K / IQ2 / IQ1 / TQ / MXFP4 / NVFP4 | deferred | - | - | - | - | - | target-dependent |

Integration status:

| item | state |
| :-- | :-- |
| Single-core unpack + fused mmul correctness | done |
| Host repack shims and numpy references | done |
| No-NPU error path | done |
| Multi-core column partitioning | not started |
| DMA streaming and descriptor tiling | not started |
| 32-core aggregate bandwidth measurement | not started |
| Gemma 4 E2B shape coverage | not started |
| Tied `token_embd` row lookup + output head | not started |
| PLE tensors on CPU | decided, not integrated |
| Attention / KV cache | outside kernel package |

## Immediate work

1. **Multi-core streaming driver**
   - partition `weights_cols` across compute tiles;
   - broadcast the shared input tile;
   - stream packed records with DMA and join outputs;
   - respect the 16,383-word FIFO descriptor limit;
   - validate aggregate bandwidth with a memory-only flow before enabling
     dequantization.

2. **Gemma 4 E2B integration**
   - instantiate K = 1536/2048/4096 and N = 256/512/2048/4096/6144/12288,
     plus the Q6_K output head at K = 1536, N = 262144;
   - handle the tied `token_embd` row lookup and full-matrix output head;
   - keep PLE `per_layer_model_proj` and `per_layer_token_embd` on the CPU;
   - start with a short context to keep attention/KV work small;
   - verify single-token decode with the M=4 mmul zero-padded to four rows.

3. **Q4_0 scale-layout decision**
   - both `q4_0` (canonical-size fp16 scales) and `q4_0_hilo` (expanded bf16
     hi/lo scales) are available;
   - `q4_0` is compute-bound (about 80B weights/s at 1.5 GHz, 96B at 1.8 GHz)
     and `q4_0_hilo` is memory-bound at 88B, so the default depends on clock:
     `q4_0_hilo` wins at 1.5 GHz, `q4_0` at 1.8 GHz, crossing over near
     1.66 GHz at the 55 GB/s target;
   - re-measure actual aggregate bandwidth (below about 50 GB/s the smaller
     `q4_0` layout wins at either clock) before fixing the default for the
     final driver.

4. **Output-head decision**
   - keep int8 Q6_K for the exact path;
   - evaluate a lossy Q4_0 output head only after a model-quality check;
   - Q8_0 is not worth the extra conversion for a ~0.2 ms saving.

5. **Test coverage**
   - add full-shape streaming tests once the multi-core harness exists;
   - add real-model slices from the target tensor set;
   - keep the no-NPU and invalid-shape tests;
   - cover host repack's subnormal-scale flush and the one-line warning, plus
     a Q8_0 block containing `-128`.

## Format bring-up plans

### Q4_K and Q5_K

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
- Prefer host-expanding the 6-bit scale/min fields to bytes: 16 six-bit
  values are 12 bytes, so the cost is only +4 bytes per 256-weight block.
- Validate against the host reference dequant output for random blocks before
  NPU bring-up.
- Measure whether folding `-dmin*min` into host-precomputed terms is exact
  enough; do not move scale application after the mmul without a tolerance
  analysis.

### BF16 and F16 pass-through

Treat these as conversion controls, not quantized formats.

- Choose a record layout and reuse the existing bf16 mmul path.
- Use them to measure pure data movement and bf16 conversion cost.
- Do not add a separate mmul implementation.

### IQ4_XS and IQ4_NL

Both use the 16-entry `kvalues_iq4nl` codebook.

- IQ4_NL: 32-row block, fp16 `d`, 4-bit indices.
- IQ4_XS: 256-row superblock, fp16 `d`, 16-bit `scales_h`, 4-bit low scales,
  and 4-bit indices.
- Add the codebook as shared `constexpr` or static data and account for it in
  the L1/constant budget.
- Dequant: `scale * kvalues_iq4nl[index]`, with IQ4_XS scales reconstructed
  from `scales_l` and `scales_h`.
- Prototype `aie::parallel_lookup` with a bf16 codebook; it is the most
  promising replacement for per-lane codebook gathers.
- Add a lookup-table path to the Python reference first.

### Later formats

Q3_K and IQ3_S share 3-bit/hmask patterns and broaden model support. Q2_K,
IQ2_XXS/XS/S, IQ1_S/M, Q4_1, Q5_1, MXFP4, NVFP4, and TQ formats are deferred
until a target model or benchmark requires them.

## Fast and relaxed modes

These are separate named variants with explicit verification policies. They
must not silently pass through the exact tolerances.

### Q4_0 relaxed single-bf16 mode

- Exact variants: `q4_0` (raw fp16 scales, 288-byte records) and `q4_0_hilo`
  (pre-split bf16 hi/lo scales, 320-byte records).
- Relaxed `fast` mode: a single host-precomputed bf16 scale, 288-byte records,
  lower precision and fewer device cycles.
- Keep the exact non-template `unpack_weights_chunk()` as the default.
- Add a named variant selector to the test shims; do not expose raw template
  arguments.
- Report the variant name with cycle results.

### Q6_K host-scale-term variant

- Precompute all three bf16 scale terms on the host.
- Measured 8239 cycles at 256x64 (1.989 weights/cycle) but 2816-byte records.
- Useful only for single-core/latency cases where the extra bytes are
  acceptable; not the default for the 32-core 55 GB/s target.

### bfp16-emulated bf16 mmul

- Enabled with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`.
- About 25% faster on one core but roughly 0.5-0.9% relative error.
- Belongs to a future relaxed-accuracy mode with its own tolerance.

### W4A8 / native int8 x int4

- XDNA2 has a native `4x16x16` int8 x int4 mmul.
- Only relevant if a future path quantizes activations to int8; not a drop-in
  for the current bf16-activation kernels.

## Technique backlog

- Q5_K high bit: fold the `to_float<bfloat16>(..., -4)` shift instead of an
  explicit `upshift`.
- Q4_0/Q8_0: compare `grow_replicate<64>()` against the current
  `broadcast_vector` scale expansion.
- Q6_K/Q4_K/Q5_K: prefer masking `uint8` payloads over `int4` vectors; reserve
  `int4` for the final `to_float`.
- Q4_K/Q5_K: expand 6-bit scale/min metadata in host repack.
- IQ formats: prototype the hardware LUT path before writing scalar gathers.
- Re-measure code size after the host-scale and int8-payload changes if code
  size becomes relevant.
- Keep `split_bf16_hi_lo()` as the shared device-side scale split; Q8_0 uses
  it for its fp16 `d`. Q4_0 no longer calls it because host repack now
  precomputes the pair.
- Do not reintroduce cursor objects, per-column `noinline` helpers, or
  struct-based hot-path access unless they measurably beat the direct
  two-column path.

## Target model composition

`gemma-4-E2B_q4_0-it.gguf` is the recommended first end-to-end target:

- 35 dense blocks, hidden 1536, FFN 6144 (first 15) / 12288 (last 20).
- Q4_0: 1.048 GB of block weights, including 876 MB of FFN.
- Q6_K: 2.257 GB total, but only 0.330 GB is the tied `token_embd` output
  head; the 1.927 GB `per_layer_token_embd` is a lookup-only tensor.
- F16: 27.5 MB `per_layer_model_proj`, which llama.cpp runs on the CPU/input
  layer.

Active per-token traffic:

| component | canonical | our repack |
| :-- | --: | --: |
| Q4_0 block weights | 1.048 GB | 1.048 GB |
| tied Q6_K `token_embd` output head | 0.330 GB | 0.440 GB |
| F16 `per_layer_model_proj` | 0.0275 GB | 0.0275 GB (CPU) |
| **NPU total, excluding CPU F16** | **1.378 GB** | **1.488 GB** |

The table uses the canonical-size `q4_0` variant. Selecting `q4_0_hilo` raises
the Q4_0 block traffic to 1.164 GB and the NPU total to 1.604 GB. At 55 GB/s
the default repacks need 27.1 ms/token, or about 37.0 tok/s; at 50 / 45 /
35 GB/s it is 29.8 / 33.1 / 42.5 ms per token, or about 33.6 / 30.2 /
23.5 tok/s. The FFN is 84% of Q4_0 block traffic. The tied output head is
about 30% of total traffic; keeping int8 Q6_K is the exact choice, while a
lossy Q4_0 head would save about 12% of token time.

`gemma-4-26B-A4B-it-qat-q4_0.gguf` is not a good first target: it is MoE with
128 experts and 8 active.

- Q4_0: 13.772 GB, of which 12.846 GB is expert weights, 0.624 GB attention,
  and 0.301 GB shared FFN.
- Q6_K: 0.606 GB, only `token_embd`.
- F32: 0.046 GB of norms, router weights/scales, and layer scales.

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
- Tests and benchmark shapes require `weights_rows` to be a multiple of
  `spiral::quants::kRowsMultiple` (256). Formats may use smaller internal
  records.
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

Shared metadata is at the start of the record (or at a format-specific offset
when payloads come first). `chunk()` points past the metadata to a specific
chunk; callers pass the record pointer directly when they need the shared
metadata.

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

`record` points at the start of the record metadata. The unpack function writes
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

The caller must zero `outputs` and select `conv_even` rounding. Q4_0,
Q4_0_hilo, and Q6_K_int8 process two column tiles together. Native Q6_K uses
a simple one-column-at-a-time baseline. Q6_K_int8 uses an int8 payload expanded
by the host repack, so its device dequantizer has no bit-plane decode.

## Format constants

### Q4_0

`q4_0` (canonical-size):

```text
canonical block: 32 rows, 18 bytes
payload: 8 x 16-byte signed-int4 4x8 payloads per 32-row block
scales:  8 raw fp16 values per 32-row block, 16 bytes
record:  64 rows, one chunk
layout:  32 bytes of scales, then 256 bytes of payloads
packed:  288 bytes per record (4.5 bits/weight)
```

The device loads the fp16 scales and splits them into bf16 hi/lo with
`split_bf16_hi_lo()` per 32-row block.

`q4_0_hilo` (host-expanded):

```text
scales:  8 pre-split bf16 hi/lo pairs per 32-row block, 32 bytes
layout:  64 bytes of scales, then 256 bytes of payloads
packed:  320 bytes per record (5.0 bits/weight)
```

The host repack computes the exact bf16 `hi + lo` split of each fp16 scale.
Both layouts keep every 32-byte tile payload 32-byte aligned; `load_v<32>` is
valid for all tiles.

### Q6_K

`q6_k` (native payload):

```text
canonical block: 256 rows, 210 bytes
ql[128], qh[64], scales[16] int8, d fp16
record:  256 rows, 1680 bytes
payload: 4 x (256-byte low-nibble plane + 128-byte high-two-bit plane)
scales:  4 x 32 per-column int8 subgroup scales
common:  16 bytes, one fp16 d per column
```

This preserves the canonical 6.5625-bit size. The device expands the bit
planes with uint4 low-nibble lanes and uint8 high-bit lanes into signed
`q - 32` bytes, then uses the same bf16 scale path as the expanded variant. It
remains compute-bound and is retained as the optimization baseline.

`q6_k_int8` (host-expanded):

```text
record:  256 rows, 2240 bytes
payload: 2048 bytes, four 512-byte int8 chunks (signed q - 32)
scales:  128 bytes, four chunks x 32 per-column int8 subgroup scales
common:  16 bytes, one fp16 d per column
padding: 48 bytes to keep every record 64-byte aligned
packed:  8.75 bits/weight
```

The host repack expands each 6-bit weight to signed int8, so the device decode
is a 64-byte load and `to_float<bfloat16>`. The per-16-row scale structure is
unchanged. Every payload chunk is 512 bytes, so all eight 64-byte 8x8 tiles in
a chunk are naturally aligned. Payloads come first, then scales, then `d`; the
2240-byte stride is the smallest 64-byte multiple that holds 2048 + 128 + 16
bytes.

The mmul keeps the four-iteration chunk loop rolled, which holds the kernel at
5008 bytes (30.6% of the 16 KiB compute-tile program memory). Fully unrolling
it grew the kernel to 12256 bytes (74.8%) without improving throughput.

### Q8_0

```text
canonical block: 32 rows, 34 bytes (fp16 d + 32 int8)
record:  256 rows, 2176 bytes
payload: 2048 bytes, 32 x 64-byte int8 tiles
scales:  128 bytes, eight blocks x eight columns of fp16 d
packed:  8.5 bits/weight
```

Payloads first, then scales, so every 64-byte tile load is naturally aligned.

## Current and planned files

```text
src/quants/
  README.md                   performance summary
  NOTES.md                    techniques and detailed results
  TODO.md                     this file
  common/
    tile.h                    mmul tile shape constants
    scale.aie.h               fp16x8 load; device bf16 split helper
    mmul.aie.h                mmul tile helper
  q4_0/
    layout.h, repack.h, unpack.aie.h, mmul.aie.h
  q4_0_hilo/
    layout.h, repack.h, unpack.aie.h, mmul.aie.h
  q6_k/
    layout.h, repack.h, unpack.aie.h, mmul.aie.h
  q6_k_int8/
    layout.h, repack.h, unpack.aie.h, mmul.aie.h
  q8_0/
    layout.h, repack.h, unpack.aie.h, mmul.aie.h
  tests/
    common.py, unpack.py, mmul.py
    unpack.aie.cc, mmul.aie.cc, repack.cc
    formats/
      q4_0.py, q4_0_hilo.py, q6_k.py, q6_k_int8.py, q8_0.py
```

Planned additions:

```text
  q4_k/  q5_k/  bf16/  iq4_nl/  iq4_xs/
  common/k_quant.h            shared 6-bit K scale/min unpack
  common/iq4nl.h              shared IQ4_NL codebook
  tests/formats/q4_k.py q5_k.py bf16.py iq4_nl.py iq4_xs.py
```

Exact filenames may change if formats share a common layout, but each format
must keep its own layout/repack/unpack/mmul units so tests and production
kernels share one implementation.

## Shared implementation work

- `common/tile.h` holds the `aie::mmul<4,8,8>` shape constants.
- `common/scale.aie.h` provides `load_fp16x8()`. `q4_0` uses it with
  `split_bf16_hi_lo()` on the device; `q4_0_hilo` performs the split in host
  repack. The same device split path is reused by both Q6_K variants.
- `common/mmul.aie.h` provides the vector mmul tile helper.
- `q4_0/unpack.aie.h` provides `load_block_scales64()` and
  `dequantize_tile()` used by both the reference unpack and fused mmul.
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
  - payload alignment where the format uses 32/64-byte tile loads
- Keep `layout.h` constants in the `constexpr int NAME = expression;` form
  parsed by `tests/common.py`; expressions may wrap across lines.

## Test and benchmark roadmap

- Keep `unpack.py` as the exact format/layout conformance test.
- Keep `mmul.py` as the end-to-end fused correctness test.
- Add a real-model test path that extracts a small slice of a representative
  tensor, repacks it, and compares against the Python reference.
- Keep the no-NPU error path and invalid-shape rejection tests.
- Add full-shape streaming tests once the multi-core harness exists.
- Benchmark at least:
  - Q4_0 256x64 canonical and hilo variants;
  - Q8_0 256x64 and 256x128;
  - Q6_K 256x64 native and int8 variants;
  - Q4_K/Q5_K 256x64 and 512x64, split output if DMA limits require it.
- Report cycles/tile, cycles/weight, weights/cycle, bytes/weight, and estimated
  weights/s at 1.5 and 1.8 GHz and 55 GB/s.
- Record new baselines in `NOTES.md` and update the summary in `README.md`.

## Definition of done

- [x] Q6_K exact int8-payload unpack and mmul pass on a device.
- [x] Q6_K int8 payload is much faster than the native bit-plane baseline;
      native decode remains the optimization target.
- [x] Q8_0 unpack and mmul pass on a device.
- [ ] Q4_K and Q5_K host repacks and dequantization match the host reference.
- [ ] Q4_K and Q5_K NPU unpack and mmul pass on a device.
- [x] Q4_0 fused two-column unpack + mmul passes exactly.
- [x] Both Q4_0 variants (canonical fp16 scales and host hi/lo) pass on a device.
- [ ] Q4_0 fast single-bf16 scale mode is exposed through a named variant with
      per-variant verification.
- [ ] Invalid shapes are rejected for every format.
- [ ] No format-specific runner or duplicate mmul implementation is introduced.
- [ ] Multi-core column partitioning and DMA streaming are validated at the
      target aggregate bandwidth.
- [ ] Gemma 4 E2B Q4_0 runs end to end on the NPU.

## Deliberately deferred

- bfp16-emulated bf16 mmul as a relaxed fast mode.
- Chunk-to-compute-tile routing for multi-chunk formats.
- Software pipelining and double buffering.
- Larger mmul variants beyond the current 4x8x8 tile.
- IQ2/IQ1/TQ/MXFP4/NVFP4 support.
- Host backend integration beyond the header/API contract.
