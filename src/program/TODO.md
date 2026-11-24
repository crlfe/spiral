> Role: tasks. Current as of 2026-11-24. History: `git log -- src/program/TODO.md`.

# `src/program` tasks

One fixed NPU program that executes Q4_0 and Q6_K matmuls at runtime-varying
dimensions, built from Python and called from C++. Design is in `DESIGN.md`,
usage in `README.md`, numbers in
[`../../docs/measurements.md`](../../docs/measurements.md).

## Status

The design is functional and verified on device:

- no-op, raw-block checksum, Q4_0, Q6_K and cycle paths;
- runtime-sized C++ fills and the static Python fixed-max test path;
- the 30-case C++ golden matrix, including partition boundaries, padded
  256-column cases, both large orientations, and the full 262144-column Q6_K
  head as 11 chunks;
- prepared-weight reuse, weight/output column offsets, and one contiguous head
  tensor;
- the `spiral::Program` C++ API (`prepare` / `run`, `WeightSegment`,
  `prepare_chain` / `run_chain`);
- pipelined submissions (`submit` / `finish`) and the shared bf16 epilogue with
  softcap and GEGLU kinds;
- Python suite and `--head`.

The worker is a single `range_` loop carrying the weight block, input group and
group index. Weight streams are continuous and padded only at the end. Q4_0 and
Q6_K use one block kernel per L1 block.

Remaining work is performance tuning, launch consolidation and a few structural
fusions. Correctness and data-flow stability still take priority.

## Active

- [ ] Keep `README.md`, `DESIGN.md` and this file in sync with the code.
- [ ] Re-run the full golden set, Python suite and `--head` after any worker or
      stream change.
- [ ] Use the production baseline in `docs/measurements.md`: isolated cases,
      warm-up run, then median/min over repeated interleaved runs.
- [ ] `always_inline` on `unpack_mmul`: measure `.text`, stack and cycles. Do
      not pointer-rewrite the record/input loops (see `docs/rejected.md`).

## Planned

1. **Productive, quiet operation** (only after correctness). Peak t/s is not
   the metric: prefer configurations that keep the machine cool and quiet, then
   reduce host CPU work and heat.
   - judge PM QoS and thread-count choices with the thermal baseline, not t/s
     alone;
   - the streaming path is at the PoC throughput; the remaining fixed cost is
     per-call launch/dispatch (~60-70 us), so the lever is fewer calls or
     batching, not weight-fill reshaping;
   - evaluate batching sequential operations into one launch, or a persistent
     worker queue, if the host/driver model supports it;
   - reduce host-side work: direct DMA gathers from canonical
     `[row_tile][col_pair]` storage instead of repack/copy; keep prepared
     weights resident;
   - revisit output drains with per-compute runtime consumers (contiguous
     full-group drains are safe but over-copy);
   - reduce Q6_K compute overhead: the probe-vs-real gap is the dequant/mmul
     path, not streaming;
   - retune L1/L2 block sizes only with the stable baseline;
   - consider a small compile-time row-step unroll if per-tile call overhead is
     visible.

2. **Combine parallel column matmuls in the caller/model driver** (API and
   benchmarks in place):
   - [x] gate+up as one wide call per FFN layer;
   - Q+K+V as one wide call is not safe across the QK-norm/RoPE gaps: the
     scheduler reuses compute buffers by graph-order lifetime, so moving a
     later matmul early can clobber a gap tensor. The opt-in look-ahead avoids
     host reordering but is within noise on Gemma.

3. **Sequential fusions that need new device dataflow:**
   - `inp_gate` + `proj`: a two-stage kernel with a 256-wide intermediate,
     device gate activation, and a broadcast before `proj`;
   - FFN gate+up+down: a persistent FFN design with a global barrier,
     intermediate broadcast and device activation.

4. **Q6_K head one-launch expansion:**
   - `MAX_WEIGHT_BLOCKS=1272`, `MAX_TOTAL_TILES>=3072`, ~333 MB weight BO,
     ~20 fill parts per group; verify BD IDs, instruction size and XRT BO
     allocation;
   - a smaller 256-block / 4-part step with `MAX_BANDS_PER_CHUNK=64` would give
     64-band chunks and 8 head launches at 64 MiB of weight BO.

5. **Python runner path.** It still uses the fixed-max layout. Decide whether to
   move it to the runtime partition path used by C++ or keep it as a simple
   test harness.

6. **Head plumbing.** The C++ wrapper handles one contiguous head tensor through
   column offsets; the Python `--head` check still uses one output tensor per
   chunk. Consolidate if useful.

7. **Coverage.** Extend the manifest if new E2B shapes appear. The current
   representative set passes; `weights_cols=256` is covered by two padded cases.

8. **NPU+iGPU split (parked).** The first llama.cpp attempt places a 75/25 layer
   split correctly but fails on cross-device views. Parked until the NPU path
   is closer to competitive, ideally at a 50/50 split on a much larger model.
   The per-call, per-tensor interface is kept for that.

## Chain length: TXN concatenation and overlays

Launch consolidation is no longer an open design question.
`tests/poc/txn_concat` builds multiple fixed-program matmuls into one TXN and
chains each output into the next input; `tests/poc/overlay.md` shows the
program-memory tail can be swapped or broadcast while the cores run. The
optimization metric is **ops per TXN and host boundaries per token**, not
launches per token.

- **Dispatch floor:** an empty TXN is ~44 us on XRT; N matmuls in one TXN pay it
  once.
- **Layout:** the native drain already emits `[N/8][vec][8]`, the next matmul's
  packed input, so an identity chain is bit-identical and needs no pack kernel.
- **Activation:** the GEGLU epilogue is the first non-identity link. It needs no
  patch-policy change: op0 uses the epilogue mode, op1 uses `input_from_output`.
- **Program memory:** the GEGLU epilogue brings production `.text` to 12544 B.
  Later stages belong in the overlay tail or a split image.

### What still breaks a token chain

A measured Gemma 4 E2B Q4_0 decode graph is 268 dispatches, 234 host spans and
1202 CPU nodes between them. The host ops that force a boundary:

| host op | count | status |
| :-- | --: | :-- |
| `GLU` (GEGLU) | 34 | now an epilogue link |
| `RMS_NORM` + `MUL` | 235 | feeds the next matmul |
| `FLASH_ATTN_EXT` | 34 | the remaining real compute island |
| `ROPE` | 48 | between Q/K norm and attention |
| `VIEW`/`PERMUTE`/`RESHAPE` | 367 | attention layout; needs a layout decision |
| `SET_ROWS`/`GET_ROWS` | 28/24 | KV append and per-layer embedding rows |
| Q6_K head | 11 links | already in-stream; throughput only |

Reproduce with `SPIRAL_PROFILE_CPU=1 SPIRAL_PROFILE_NODES=1` and `-v` on
`llama-completion`; the backend logs at INFO, which the default level hides.

### Links to add, in dependency order

- [ ] **RMS norm** with the norm weight folded in. Hidden width is 1536 and
      every compute needs all K for its own columns, so the reduction can be
      replicated per compute; no cross-group barrier. Pre-norm reads the
      residual; post-norm reads the distributed down output (32 x 3 KiB
      redundant reads per block) and adds the graph's second operand, which in
      Gemma is `pe_in`, not the residual.
- [ ] **RoPE** for Q/K, after the QK norm.
- [ ] **KV append + attention as one link.** Cache in host DDR, stable addresses
      in the token TXN; context length, GQA and sliding window as config words.
      Reuse the mlir-aie `llama_3_2_1b` patterns.
- [ ] **Hoist token-start gathers** into the input BO (token row, 24 per-layer
      rows, F16 `per_layer_model_proj`). Keep sampling host-side; it is the
      natural TXN boundary.
- [ ] **Head throughput only:** the 11 chunks are in-stream; the one-launch
      expansion is a streaming optimization.

### Milestones

| state | NPU commands/token | host boundaries | needs |
| :-- | --: | --: | :-- |
| today (per-call) | 268 | 234 | - |
| concat probe (measured) | 1 | 1 | concat, real weights, no glue |
| FFN block (gate+up+down) | ~233 | ~199 | GEGLU link (done), chain API |
| whole layer | ~35 | ~35 | norms, RoPE, attention |
| whole token | 1 | prologue + sampling | all links, gathers hoisted |

Record commands/token, ops per command, host boundaries, wall, kernel time,
instruction bytes and the per-stage numeric policy for every milestone, with
temperature/fan context.

## Open risks

- **Hangs** are almost always a mismatch between configured counts, host
  partition fills and drained padding. Keep them derived from one partition
  function.
- **Nested carried-memref loops** deadlocked on device. Keep the worker a single
  loop carrying all state, or validate a new structure in
  `tests/poc/iter_args.py` first.
- **Dynamic BD limits:** the shim repeat/iteration dimension is capped at 64 in
  our fills. Verify for every runtime-shaped fill/drain.
- **Dynamic host loops / BD pool:** straight-line runtime-sized fills with static
  BD assignment are validated; runtime loops plus the BD pool deadlocked.
- **JIT cache:** inline-kernel objects can be stale after source edits; clear
  `~/.npu/cache` entries.
- **Padding columns:** `weights_cols=256` pads to 512, roughly doubling those
  columns' work.
- **Per-stage numerics:** chained bf16 intermediates need an oracle and a
  tolerance; GEGLU shows ~5e-3 global relative from the bf16 vtanh alone.
- **Instruction/config growth:** 1.10 MB at 286 ops is in the good region;
  per-copy cost regresses above ~1 MB and each link adds 8 KiB of config.
- **Cross-compute ordering** comes from the TXN's TCT waits, not a lock; verify
  any stage that reads data written by other computes.
- **Attention** adds array compute that the 29.7 ms zero-data simulation did not
  include.
- **Head overhead:** 11 band chunks, up to 48 bands each (120 blocks max; the
  last chunk is 32 bands).
- **Program memory:** production `.text` 12544 B (77%), `program_output` 1360 B
  with no stack frame, leaving ~3.9 KiB resident headroom. Keep vector
  activation math in bf16; the f32 GELU was 4x the code for the same error.
- **Buffer sizing:** host weight BO is 32 MiB at 128 blocks; the canonical packed
  tensor can be ~333 MB for the Q6_K head.

## Definition of done

One loaded C++ program executes sequential Q4_0 and Q6_K calls covering the
Gemma 4 E2B matmul set:

- all Q4_0 attention, FFN (6144 and 12288), `inp_gate` and `proj` shapes;
- the Q6_K output head (`weights_rows=1536`, `weights_cols=262144`) via band
  chunking and one contiguous host tensor;
- `weights_cols=256` cases padded to 512;
- outputs matching a NumPy reference, with no hangs and no recompilation per
  shape.

This is met for the representative matrix in `build/golden/index.txt`.
Remaining work is performance and any additional shapes.
