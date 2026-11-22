# `src/program` tasks

One fixed NPU program (one xclbin) that executes Q4_0 and Q6_K matmuls at
runtime-varying dimensions, built from Python and called from C++.  The
target is the Gemma 4 E2B Q4_0 matmul set plus its Q6_K output head.

Information, design details, measurements, and lessons learned live in
`NOTES.md`.  Usage lives in `README.md`.

## Current status

The design is functional and verified on device:

- no-op, raw-block checksum, Q4_0, Q6_K, and cycle paths;
- runtime-sized C++ fills and the static Python fixed-max test path;
- the 30-case C++ golden matrix, including partition boundaries, padded
  256-column cases, both large orientations, and the full 262144-column Q6_K
  head as 11 chunks;
- prepared-weight reuse, weight/output column offsets, and one contiguous
  head tensor;
- the `spiral::Program` C++ API (`prepare` / `run`);
- Python suite and `--head`.

The current worker is a single `range_` loop carrying the weight block, input
group, and group index.  Weight streams are continuous and padded only at the
end.  Q4_0 and Q6_K use one block kernel per L1 block.  All of this passes.

Remaining work is performance tuning, launch consolidation, and a few
structural fusions.  Correctness and data-flow stability still take priority
over peak bandwidth.

## Done

- [x] Fixed geometry, config decode, and per-call format selection.
- [x] No-op data-flow gate: FIFO wiring, fills, carry, output join/drain,
      band-chunk offsets.
- [x] Q4_0 tile compute and representative E2B shapes.
- [x] Q6_K tile compute alongside Q4_0.
- [x] Band chunking for large `weights_cols`, including the full Q6_K head.
- [x] Runtime-sized C++ transaction fills (four input fills, two weight fills
      per group) with static BD assignment.
- [x] Real host repack via `src/quants` headers and NumPy golden checks.
- [x] Stable `spiral::Program` API with prepared-weight caching.
- [x] Direct reads for complete tiles and word-sized carry copies.
- [x] Block kernels for Q4_0 and Q6_K.
- [x] 8 KiB / 32 KiB L1/L2 tuning and weight FIFO depth 2.
- [x] Continuous weight streams with one input object per four rows.
- [x] Single-loop carried worker (replaces the deadlocking nested loops).
- [x] Removed unused split kernels, `program_row_begin`, and dead worker
      block-limit math from `run.py`.
- [x] Compile-time production mode (`build.py --production`) drops the
      diagnostic kernels and cycle path.
- [x] Loop idioms: vectorized `program_zero` (`vst`), pointer-increment carry
      copy (`lda/st [p],#8`), and attempt-loop unroll sweep (rejected; full
      unroll overflows).  Production `.text` is now 11872 bytes; diagnostic is
      14880.
- [x] Shared block loop: one non-template `program_block::run` with the
      `unpack_mmul` instantiation as the apply pointer.  Production `.text`
      12192 -> 11104 bytes; Q4_0 block 4368 -> 3232, Q6_K 5168 -> 4032.  No
      measurable timing change with five warm-up and twenty timed runs.
- [x] Dynamic output zero (`cols_bands * OUTPUT_TILE_ELEMS`), removing the
      fixed-object penalty when a call uses fewer bands.
- [x] Q6_K head chunking: 48 bands per chunk, 11 chunks instead of 21.
      Added `./test --head-timing` for prepared-weight head measurements.
- [x] Cached `xrt::run` in `Program::Impl` for reuse across calls (small,
      low-risk launch-cleanup; effect within noise).
- [x] Per-handle resident weight BOs in `PreparedWeights`, using a compact
      group stride so tensors do not need fixed-max padding.
- [x] Combined column matmuls: `Program::prepare(shape, WeightSegment[],
      count)` concatenates segments along `weights_cols`; `--combined-timing`
      benchmarks gate+up and Q+K+V and now verifies every segment.
- [x] `./test --cold-timing` alternates several resident BOs
      (`SPIRAL_COLD_BOS`) to measure the real per-layer streaming cost instead
      of the warm-cache number.
- [x] The `ggml-spiral` backend (`integrations/llama.cpp`) now chains
      consecutive `MUL_MAT`s with the same input data and K into one wide call
      (up to four segments).  Gate+up is the beneficiary in Gemma; pp8
      improved from 106.1 to 122.2 t/s and tg1 from 16.2 to 17.5 t/s in that
      session, which predates the current pmode baseline, so those absolute
      values are not comparable with `NOTES.md`'s throughput table.  Chaining across
      intervening ops is unsafe because the scheduler reuses compute buffers
      based on graph-order lifetimes.  Combined handles duplicate the paired
      weights in NPU BOs and cost ~1 s to prepare on the first graph.
- [x] Investigated the large-matmul streaming path.  Per-call kernel time fits
      `~80 us + bytes / 48 GB/s`; host stages are <20 us.  Naive bytes/time is
      misleading because the fixed launch floor dominates small calls and
      averages.  `xrt::bo` cacheable/normal flags are unusable and
      `weights_l2` FIFO depth 3 is neutral.  Reduced call count, not more
      bandwidth, is the actionable lever.
- [x] `./test --cold-timing` reports min/p50/p95/max over alternating BOs
      (`SPIRAL_COLD_BOS`, `SPIRAL_COLD_ITERS`).
- [x] Fix the weight over-read: one `kL2StreamBytes` guard block after the
      compact weights BO removes the AMD-Vi IO_PAGE_FAULTs (10/run -> 0).
- [x] Split `kernel_us` into `start_us` / `wait_us` / `sync_us`.
- [x] `SPIRAL_COLD_GAP_US` shows an NPU-idle gap slows the next call, so
      keeping the NPU busy matters.
- [x] Pipelined submissions: `Program::submit`/`finish` with a configurable
      slot pool (`SPIRAL_PIPELINE_SLOTS`, default 2).  Q6_K head pass drops
      by ~16% wall; two slots capture almost all of it.  `pp8` 95.3 -> 112.5
      t/s; `tg16` unchanged (18.0 -> 18.3) because only the head is
      multi-chunk at `rows=1`.
- [x] Cross-node look-ahead: submit the next same-input matmul early and
      defer only its host unpack.  Correct, but within noise on Gemma; kept
      behind `SPIRAL_LOOKAHEAD=1` (off by default).
- [x] Trial `q4_0_packed16` as a build-time Q4 block choice.  It was 11.2%
      faster in the kernel (8328 -> 7397), but the exact Q4_0 path then
      improved to 7883 cycles with the shared split (6.2% gap) and packed16 is
      near-exact rather than bit-exact.  Removed from `src/program`; the
      `src/quants` format and its tests remain.
- [x] Hold `/dev/cpu_dma_latency` at 0 for the Program lifetime (PM QoS).
      It is now opt-in: `SPIRAL_DMA_LATENCY=1` enables it,
      `SPIRAL_DMA_LATENCY_PATH` overrides the node, and enabling it with an
      inaccessible node throws.  Measured effect is workload-dependent:
      `pp8` 118.1 vs 111.4 t/s (+6% on), `tg16` 16.84 vs 18.11 (-7% on), and
      a 300-token generation 15.2-15.3 vs 16.8-17.3 t/s (-10% on).  Default
      off because decode dominates.
- [x] Real-query baseline: Gemma E2B Q4_0, native Q6_K head, one CPU
      fallback thread.  Spiral NPU: 17.78 t/s on a 300-token answer (56.2
      ms/token), `tg16` 18.47, `pp8` 111.5.  CPU (`-ngl 0`, 12 threads):
      35.78 t/s (28.0 ms/token), `tg16` 35.34, `pp8` 224.1.
- [x] Wait stalls: resolved.  The 45-50 ms `wait_us` spikes were host CPU
      quota throttling (kernel tracing showed the submitting thread switched
      out while runnable, no device activity in the window).  With the quota
      removed a 64-token run has zero calls above 15 ms and median wait
      ~95 us; the device job lifecycle is at most ~2.6 ms.
- [x] `tests/poc/iter_args.py` device regression for carried memrefs and
      conditional acquire/release.

## Active

- [ ] Keep `README.md`, `NOTES.md`, and this file in sync with the code.
- [ ] Re-run the full golden set, Python suite, and `--head` after any worker
      or stream change.
- [ ] Use the production baseline in `NOTES.md` for comparisons: isolated
      cases, warm-up run, then median/min over repeated runs.
- [x] Common bf16 tile epilogue (`program_output`): transpose the native tile
      to `[vec][16]`, optional bf16 vtanh softcap, bf16 output, and a bf16
      host overload that matches the next run's input layout.  See
      `NOTES.md` under "Output layout" and "Performance history".

## Planned

1. **Productive, quiet operation** (only after correctness).  Peak t/s is not
   the metric: prefer configurations that keep the machine cool and quiet,
   then reduce host CPU work and heat.  Priority order:
   - use the temperature/fan baseline (signals and procedure are in
     `NOTES.md` under "Efficiency acceptance"); judge PM QoS and thread-count
     choices with it, not with t/s alone;
   - use the production baseline in `NOTES.md`; the streaming path is now
     at the PoC throughput.  The remaining fixed cost is per-call
     launch/dispatch (~60-70 us), so the next lever is fewer calls or batching,
     not weight-fill reshaping;
   - evaluate batching sequential operations into one launch, or a persistent
     worker queue, if the host/driver model supports it;
   - reduce host-side work: direct DMA gathers from canonical
     `[row_tile][col_pair]` storage instead of repack/copy, and keep prepared
     weights resident;
   - revisit output drains with per-compute runtime consumers; contiguous
     full-group drains are safe but over-copy;
   - reduce Q6_K compute overhead: probe vs real gap is the dequant/mmul
     path, not streaming;
   - retune L1/L2 block sizes only with the stable baseline;
   - consider a small compile-time row-step unroll if per-tile call overhead
     is visible.
   Already tried and rejected: `-O3` (identical objects), `-Os` (larger and
   slower), inline kernels (larger/slower overall), FIFO depth 4 (BD
   exhaustion), and depth 3 (no gain over 2).
2. **Combine parallel column matmuls in the caller/model driver** (API and
   benchmarks are in place):
   - [x] gate+up as one wide call per FFN layer (graph-level chain in the
     llama.cpp backend);
   - Q+K+V as one wide call is not safe across the QK-norm/RoPE gaps: the
     scheduler reuses compute buffers by graph-order lifetime, so moving a
     later matmul early can clobber a gap tensor.  The opt-in look-ahead
     avoids host reordering but is within noise on Gemma.
3. **Sequential fusions that need new device dataflow:**
   - `inp_gate` + `proj`: two-stage kernel with a 256-wide intermediate,
     device gate activation, and a broadcast before `proj`;
   - FFN gate+up+down: persistent FFN design with a global barrier,
     intermediate broadcast, and device activation.
4. **Q6_K head one-launch expansion:**
   - `MAX_WEIGHT_BLOCKS=1272`, `MAX_TOTAL_TILES>=3072`, ~333 MB weight BO,
     ~20 fill parts per group; verify BD IDs, instruction size, and XRT BO
     allocation.  A 256-block / 4-part step with `MAX_BANDS_PER_CHUNK=64`
     would give 64-band chunks and 8 head launches at 64 MiB of weight BO.
5. **Python runner path.**  It still uses the fixed-max layout (64 input
   objects: one config plus 63 groups, and 128 weight blocks).  Decide whether
   to move it to the runtime partition path used by C++ or keep it as a simple
   test harness.
6. **Head plumbing.**  The C++ wrapper already handles one contiguous head
   tensor through column offsets.  The Python `--head` check still uses one
   output tensor per chunk; consolidate if useful.
7. **Coverage.**  Extend the manifest if new E2B shapes appear.  Current
   representative set passes; `weights_cols=256` cases are covered by two
   padded cases.
8. **Remaining loop experiment:** `always_inline` on `unpack_mmul`; measure
   `.text`, stack, and cycles.  Do not pointer-rewrite the record/input loops
   (q4_0 counterexample shows a 2–4% regression).
9. **NPU+iGPU split (parked).**  The first llama.cpp attempt places a 75/25
   layer split correctly but fails on cross-device views; it is parked until
   the NPU path is closer to competitive, ideally at a 50/50 split on a much
   larger model.  The per-call, per-tensor interface is kept for that.

## Open risks

- **Hangs**: almost always a mismatch between configured counts, host
  partition fills, and drained padding.  Keep those derived from one
  partition function.
- **Nested carried-memref loops** deadlocked on device.  Keep the worker as a
  single loop carrying all state, or validate any new structure in
  `tests/poc/iter_args.py` first.
- **Dynamic BD limits**: the shim BD repeat/iteration dimension is capped at
  64 in our fills (the PoC documents a separate 1023-element contiguous access
  limit).  Verify the limit for every runtime-shaped fill/drain, especially
  if variable fills or strided output drains return.
- **Dynamic host loops / BD pool**: straight-line runtime-sized fills with
  static BD assignment are validated; runtime loops plus the BD pool
  deadlocked against the FIFOs.
- **JIT cache**: inline-kernel objects can be stale after source edits.
  `build.py` deletes the active build directory's `*.o`; also clear stale
  `~/.npu/cache` entries when working with inline kernels.
- **Padding columns**: `weights_cols=256` (`attn_k`/`attn_v`/`inp_gate`) pads
  to 512, roughly doubling those columns' work.
- **Head overhead**: the Q6_K head is 11 band chunks, up to 48 bands each
  (120 blocks maximum; the last chunk is 32 bands).
- **Program memory**: diagnostic build is 14880 bytes per core (91% of
  16 KiB); production is 11872 bytes (73%).  Recheck after adding code to a
  kernel; full attempt-loop unroll already overflows.
- **Buffer sizing**: the host weight BO is 32 MiB at 128 blocks; the host
  canonical packed tensor can be ~333 MB for the Q6_K head.

## Definition of done

One loaded C++ program executes sequential Q4_0 and Q6_K calls covering the
Gemma 4 E2B matmul set:

- all Q4_0 attention, FFN (6144 and 12288), `inp_gate`, and `proj` shapes;
- the Q6_K output head (`weights_rows=1536`, `weights_cols=262144`) via band
  chunking and one contiguous host tensor;
- `weights_cols=256` cases padded to 512;
- outputs matching a NumPy reference, with no hangs and no recompilation per
  shape.

This is met for the representative matrix in `build/golden/index.txt`.
Remaining work is performance and any additional shapes.
