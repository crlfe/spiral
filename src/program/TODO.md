# `src/program` tasks

One fixed NPU program (one xclbin) that executes Q4_0 and Q6_K matmuls at
runtime-varying dimensions, built from Python and called from C++.  The
target is the Gemma 4 E2B Q4_0 matmul set plus its Q6_K output head.

Information, design details, measurements, and lessons learned live in
`NOTES.md`.  Usage lives in `README.md`.

## Current status

The design is functional and verified on device:

- no-op, Q4_0, Q6_K, stream probe, and cycle paths;
- runtime-sized C++ fills and the static Python fixed-max test path;
- the 27-case C++ golden matrix, including partition boundaries, padded
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
      unroll overflows).  Production `.text` is now 11232 bytes; diagnostic is
      13984.
- [x] Dynamic output zero (`cols_bands * OUTPUT_TILE_ELEMS`), removing the
      fixed-object penalty when a call uses fewer bands.
- [x] Q6_K head chunking: 48 bands per chunk, 11 chunks instead of 21.
      Added `./test --head-timing` for prepared-weight head measurements.
- [x] Cached `xrt::run` in `Program::Impl` for reuse across calls (small,
      low-risk launch-cleanup; effect within noise).
- [x] Combined column matmuls: `Program::prepare(shape, WeightSegment[],
      count)` concatenates segments along `weights_cols`; `--combined-timing`
      benchmarks gate+up and Q+K+V.  Gate+up saves ~3.2 ms/token and QKV
      ~2.1 ms/token (~5.4 total), taking the launch-limited estimate to
      ~19.8 tok/s.
- [x] `tests/poc/iter_args.py` device regression for carried memrefs and
      conditional acquire/release.

## Active

- [ ] Keep `README.md`, `NOTES.md`, and this file in sync with the code.
- [ ] Re-run the full golden set, Python suite, and `--head` after any worker
      or stream change.
- [ ] Use the locked-clock baseline in `NOTES.md` for comparisons: isolated
      cases, warm-up run, then median/min over repeated runs.

## Planned

1. **Performance tuning** (only after correctness).  Priority order:
   - use the locked-clock baseline in `NOTES.md`; the streaming path is now
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
   - use gate+up as one wide call per FFN layer;
   - use Q+K+V as one wide call for the 15 KV layers;
   - keep the existing per-shape fallback for shared-KV layers.
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
- **Program memory**: diagnostic build is 13984 bytes per core (85% of
  16 KiB); production is 11232 bytes (69%).  Recheck after adding code to a
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
