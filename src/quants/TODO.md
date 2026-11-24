> Role: tasks. Current as of 2026-11-24. History: `git log -- src/quants/TODO.md`.

# `src/quants` tasks

Open work only. Design and contracts are in [`DESIGN.md`](DESIGN.md); current
numbers are in [`../../docs/measurements.md`](../../docs/measurements.md).

## Status

| format | layout | host repack | exact unpack | fused mmul | tests | next |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Q4_0 | done | done | done | done | exact | multi-core |
| Q4_0_hilo | done | done | done | done | exact | multi-core |
| Q6_K | done | done | done | done | exact | multi-core |
| Q6_K_int8 | done | done | done | done | exact | multi-core, optional variants |
| Q8_0 | done | done | done | done | exact | multi-core |
| Q4_K | not started | - | - | - | - | plan below |
| Q5_K | not started | - | - | - | - | share Q4_K scale/min |
| BF16 / F16 | not started | - | - | - | - | pass-through controls |
| IQ4_NL | not started | - | - | - | - | codebook LUT |
| IQ4_XS | not started | - | - | - | - | codebook LUT + scales |
| Q3_K / IQ3_S | deferred | - | - | - | - | 3-bit/hmask |
| Q2_K / IQ2 / IQ1 / TQ / MXFP4 / NVFP4 | deferred | - | - | - | - | target-dependent |

| integration item | state |
| :-- | :-- |
| Single-core unpack + fused mmul correctness | done |
| Host repack shims and NumPy references | done |
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
   - stream packed 256x16 transport tiles with DMA and join outputs;
   - respect the 16,383-word FIFO descriptor limit;
   - validate aggregate bandwidth with a memory-only flow before enabling
     dequantization.

2. **Gemma 4 E2B integration**
   - instantiate K = 1536/2048/4096 and N = 256/512/2048/4096/6144/12288,
     plus the Q6_K output head at K = 1536, N = 262144;
   - handle the tied `token_embd` row lookup and full-matrix output head;
   - keep PLE `per_layer_model_proj` and `per_layer_token_embd` on the CPU;
   - start with a short context to keep attention/KV small;
   - verify single-token decode with the M=4 mmul zero-padded to four rows.

3. **Output-head decision**
   - native `q6_k` is the exact choice and is memory-bound at a higher
     full-NPU rate than `q6_k_int8` despite slightly more cycles;
   - evaluate a lossy Q4_0 head only after a model-quality check;
   - Q8_0 is not worth the extra conversion.

4. **Test coverage**
   - add full-shape streaming tests once the multi-core harness exists;
   - add real-model slices from the target tensor set;
   - keep the no-NPU and invalid-shape tests;
   - add the exhaustive 65536-pattern scale-split probe on device, and a host
     check that repack stores canonical scale bytes (the remaining gate on the
     exactness rule);
   - cover subnormal scales across Q4_0, Q4_0_hilo and Q8_0;
   - cover `q4_0_packed16`'s rejection of inf/nan scales, and a Q8_0 block
     containing `-128`.

## Format bring-up plans

### Q4_K and Q5_K

Largest K-quant families in the local models; share one scale/min
implementation.

- 256-row superblocks with eight 32-row groups and 6-bit packed scale/min
  metadata (`get_scale_min_k4`).
- Record: 256 rows x 8 columns.
- Hoist per-column superblock `d` and `dmin` into common record metadata.
- Keep per-chunk 6-bit scale/min for the two 32-row groups in each 64-row
  chunk.
- Dequant: `d * scale * q - dmin * min`.
- Q5_K adds the `qh` high bit; reuse the Q4_K scale/min path.
- Prefer host-expanding 6-bit scale/min to bytes: 16 values are 12 bytes, so
  the cost is +4 bytes per 256-weight block.
- Validate against the host reference for random blocks before NPU bring-up.
- Measure whether folding `-dmin*min` into host terms is exact enough; do not
  move scale application after the mmul without a tolerance analysis.

### BF16 and F16 pass-through

Conversion controls, not quantized formats. Choose a record layout, reuse the
existing bf16 mmul path, use them to measure pure data movement, and do not add
a separate mmul implementation.

### IQ4_XS and IQ4_NL

Share the 16-entry `kvalues_iq4nl` codebook.

- IQ4_NL: 32-row block, fp16 `d`, 4-bit indices.
- IQ4_XS: 256-row superblock, fp16 `d`, 16-bit `scales_h`, 4-bit low scales,
  4-bit indices.
- Add the codebook as shared static data and account for it in the L1/constant
  budget.
- Dequant: `scale * kvalues_iq4nl[index]`, with IQ4_XS scales reconstructed
  from `scales_l` and `scales_h`.
- Prototype `aie::parallel_lookup` with a bf16 codebook.
- Add a lookup-table path to the Python reference first.

### Later formats

Q3_K and IQ3_S share 3-bit/hmask patterns and broaden model support. Q2_K,
IQ2_XXS/XS/S, IQ1_S/M, Q4_1, Q5_1, MXFP4, NVFP4 and TQ are deferred until a
target model requires them.

## Fast and relaxed modes

Separate named variants with explicit verification policies; they must never
pass through the exact tolerances.

- **Q4_0 single-bf16 scale.** One host-precomputed bf16 scale, 288-byte
  records, lower precision. Add a named variant selector to the test shims and
  report the variant name with cycle results.
- **Q6_K host three-term scales.** Precompute all three bf16 terms. Only
  interesting for latency cases where a larger record is acceptable.
- **bfp16-emulated bf16 mmul.** `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`;
  about 0.5-0.9% relative error. Needs its own tolerance.
- **W4A8 / native int8 x int4.** `4x16x16` is native but needs int8 activations;
  not a drop-in for the current bf16 path.

## Technique backlog

- Q5_K high bit: fold the `to_float<bfloat16>(..., -4)` shift instead of an
  explicit `upshift`.
- Q4_0/Q8_0: compare `grow_replicate<64>()` against `broadcast_vector` for
  scale expansion (it lost for Q8_0; confirm for Q4_0).
- Q6_K/Q4_K/Q5_K: mask `uint8` payloads over `int4` vectors and reserve `int4`
  for the final `to_float`.
- IQ formats: prototype the hardware LUT path before writing scalar gathers.
- Keep `split_fp16_exact()` the one shared device scale split. Do not
  reintroduce the removed pre-bias helpers.

## Test and benchmark roadmap

- Keep `unpack.py` as the exact conformance test and `mmul.py` as end-to-end.
- Add a real-model slice test: extract a small tensor slice, repack it, compare
  with the Python reference.
- Benchmark at least every format at 256x16, Q4_0 canonical and hilo at 256x64,
  Q8_0 at 256x64/256x128, Q6_K native and int8 at 256x64, and Q4_K/Q5_K at
  256x64 and 512x64 with split output if DMA limits require it.
- Record results in `docs/measurements.md`, not here.

## Definition of done (open items)

- [ ] Q4_K and Q5_K host repacks and dequantization match the host reference.
- [ ] Q4_K and Q5_K NPU unpack and mmul pass on a device.
- [ ] Q4_0 fast single-bf16 mode is exposed through a named variant with
      per-variant verification.
- [ ] Invalid shapes are rejected for every format.
- [ ] Multi-core column partitioning and DMA streaming validate at the target
      aggregate bandwidth.
- [ ] Gemma 4 E2B Q4_0 runs end to end on the NPU.

## Deliberately deferred

- bfp16-emulated bf16 mmul as a relaxed fast mode.
- Chunk-to-compute-tile routing for multi-chunk formats.
- Software pipelining and double buffering.
- Mmul variants beyond the current 4x8x8 tile.
- IQ2/IQ1/TQ/MXFP4/NVFP4 support.
- Host backend integration beyond the header/API contract.
