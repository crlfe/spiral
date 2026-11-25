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
      Gemma is `pe_in`, not the residual. See "RMS norm link: detailed plan"
      below.
- [ ] **RoPE** for Q/K, after the QK norm.
- [ ] **KV append + attention as one link.** Cache in host DDR, stable addresses
      in the token TXN; context length, GQA and sliding window as config words.
      Reuse the mlir-aie `llama_3_2_1b` patterns.
- [ ] **Hoist token-start gathers** into the input BO (token row, 24 per-layer
      rows, F16 `per_layer_model_proj`). Keep sampling host-side; it is the
      natural TXN boundary.
- [ ] **Head throughput only:** the 11 chunks are in-stream; the one-launch
      expansion is a streaming optimization.

### RMS norm link: detailed plan

**Goal.** Fuse `RMS_NORM -> MUL(norm weight)` into the consuming matmul so the
CPU norm nodes and one CPU->NPU boundary per matmul disappear. The current x4
stage profile puts `RMS_NORM` at 1.24 ms/step over 242 nodes and `MUL` at
0.43 ms/step over 298 nodes; it is the largest remaining fallback op. A
successful link must pass the golden and llama.cpp A/B checks, remove those
nodes from the profile, fit program/L1 memory, and improve the 4-slot soak.

**What the failed prototypes established.**

- `hacking-prenorm` (weight-stream prelude) got the math and layout right
  (one-hot norm weights pass at K 0/7/256/1024/1280/1535; normal-only build
  passes the 32-case manifest) but put the norm slice in the per-compute
  weight stream and emitted the sum-pass fills on every call. Normal calls had
  to drain those fills, which intermittently timed out the full manifest, and
  the static BD-ID pass rejected a runtime `scf.if` around the prelude task
  group. The branch also reached `.text` 15808/16384 B and a 3840 B stack.
  Do not revive the weight-stream slice or always-present prelude fills.
- `hacking-prenorm-2` (standalone norm stage) was slower than the CPU: a
  separate norm dispatch per matmul (about 235/token) plus a full activation
  read/write and f32 accumulator costs more than the host op. Do not make the
  norm a separate launch.
- Both branches predate the `npu.py` / `sequence.py` / `worker.py` split. The
  current split is what makes a clean host-body variant and a config-driven
  worker possible.

**Design decision: config-carried norm weights, two host TXN bodies.**

1. **Norm weights in the config object.** The config object is 8 KiB and only
   the first 64 B are used. Store `hi[K]` then `lo[K]` as bf16 (K=1536:
   `2 * 1536 * 2 = 6144 B`), keeping the config words at offset 0. The host
   converts the f32 norm weight once in `prepare`, and the prenorm config fill
   writes the config words plus the hi/lo vector. The worker holds the acquired
   config object for the call and reads the weight from it; no 6 KiB L1 copy,
   no weight-stream slice, no per-compute duplication.
2. **Two host TXN bodies.** Generate a normal body and a prenorm body at build
   time and select one in `Program`/`PreparedWeights`. The prenorm body has
   the sum-pass input fills followed by the normal input fills; the normal body
   has only the normal fills. This avoids the branch's always-present fills and
   the static BD-ID rejection of runtime `scf.if`. Do not put a runtime branch
   in the host sequence. If two generated bodies prove impractical in
   `build.py`, the fallback is the dynamic EmitC runtime path, not a runtime
   `scf.if`.
3. **Worker prelude.** For `CFG_PRENORM` calls:
   - keep the config object acquired after `program_begin`;
   - sum pass: acquire each input group from the first fill set, accumulate
     four f32 `sum_sq` values (one per input vec), release; compute
     `r = rsqrt(sum_sq / K + eps)` and split `r` to bf16 hi/lo;
   - scale pass in the normal loop: acquire the second input group, scale it
     in place with `(r_hi + r_lo) * (w_hi + w_lo)` combined before applying to
     the activation, run the existing block kernel, release.
   The block kernel, weight stream and epilogue are unchanged. The activation
   is read twice (sum then scale) through the broadcast input FIFO; the input
   data is only 4 x 1536 bf16 per token, so this is cheap compared with the
   weight stream.
4. **State.** Add a small per-compute norm state (four sums, `r_hi`/`r_lo`) or
   extend the state object; do not reuse the block/carry words. An 8-word
   extension is enough.
5. **Config words.** Add `CFG_PRENORM` (0/1), `CFG_PRENORM_EPS` (f32 bits) and
   `CFG_PRENORM_INV_K` (f32 bits). Keep the norm-weight offset fixed in the
   config layout and assert it in `design.py`/`design_meta.h`.

**Host/API changes.**

- `MatmulShape` gains a `PrenormSpec` (`const float* weight`, `K`, `eps`) or an
  equivalent optional handle. `PreparedWeights` owns the converted bf16 hi/lo
  vector and a flag; `prepare` validates `K == shape.weights_rows` for the
  pre-norm consumer.
- `Program::prepare`/`prepare_chain` copy the norm vector into the config BO
  for prenorm handles. `run`/`submit` select the generated prenorm body and
  fill the config object with it; normal handles keep the current body and
  config.
- The Python side mirrors this: `design.py` constants, `npu.py` kernel list,
  `sequence.py` normal/prenorm builders, `worker.py` prenorm branch,
  `host/program.cc` config fill and body selection, `build.py` two transaction
  headers.
- The config object is held for prenorm calls. Verify the input FIFO depth with
  config + one group in flight. If depth 2 stalls, the fallback is a 6 KiB
  norm buffer plus input L1 depth 1, not a weight-stream slice.

**Backend matcher.**

- Match `RMS_NORM -> MUL(weight)` where the `MUL` result feeds a supported
  matmul. Use `RMS_NORM.src[0]` as the activation and `MUL.src[1]` as the norm
  weight; read `eps` from the norm op. Skip both nodes and prepare the matmul
  with `PrenormSpec`.
- For the FFN chain, the norm feeds the combined gate+up pair; put the prenorm
  spec on the first chain op and keep the chain input at the down node.
- For Q/K/V, fuse the norm feeding each projection independently first. If one
  normalized tensor feeds several matmuls, materialize it once only after the
  simple per-matmul path works; do not re-run a 32x-replicated reduction
  unnecessarily.
- Add `SPIRAL_NO_RMS_PRELUDE=1` for A/B and keep the CPU path as the reference.

**Scope order.**

1. Pre-FFN norm into gate+up: 35 calls, K=1536, largest CPU saving, chain path
   already exists.
2. Pre-attention Q norm, then K and V. Check whether the graph shares one norm
   result across projections before duplicating.
3. Pre-`proj`/post-FFN norm and the final norm before the head later: their
   inputs are distributed outputs and may need a materialized normalized
   activation or a mixed-input prelude, like the old post-FFN note.
4. Do not start with attention or RoPE; the norm is the prerequisite they
   consume.

**Numerics.**

- Input is bf16, the sum is f32, `r` and `w` are bf16 hi/lo, and the scaled
  activation is stored bf16 before the matmul. Combine `r*w` before applying
  it; the branch measured 4.82 -> 2.34 max error from that change alone.
- The branch's `kPrenormRel = 2.5e-2` is not automatically acceptable. Measure
  the error at the **matmul output**, not only the normalized activation, and
  either meet the existing `max(1e-3, 6e-3 * column_scale)` policy or add a
  named prenorm policy with its own oracle. Do not widen the existing golden
  tolerance.
- Keep one-hot and random-weight cases; include a case where the norm weight
  is zero/one in selected lanes to catch chunk mapping.

**Memory and program budget.**

- Current production `.text` is 12416 B; the branch's norm objects were
  `stage` 496 B + `scale` 1232 B + `finish` 32 B. The current headroom is
  about 4 KiB, so the prelude should fit, but measure the linked ELF and stack
  on both normal and prenorm builds. If it does not fit, use the overlay tail
  or split the image; do not silently drop the norm.
- Holding the config object avoids a 6 KiB L1 norm buffer. Verify the input
  FIFO depth and the worker's group loop; the branch used input L2 depth 8 and
  still saw flake, so measure this path directly with the full manifest.
- The prenorm body has twice the input fills of a normal body. They are 8 KiB
  broadcasts; count BD IDs and the input FIFO depth. The weight stream is
  unchanged, so the block kernel and its tiles are untouched.

**File map.**

- `src/funcs/rmsnorm/` (new): pure vector sum-of-squares, `invsqrt`, hi/lo
  combination and scale; `tests/` with a f32/bf16 reference and a cycle
  harness. No FIFO, config or state.
- `src/program/kernels/program_prenorm_*`: the program wrapper that reads the
  config object, runs the funcs kernel and advances the worker state.
- `src/program/design.py`: `CFG_PRENORM`, eps/inv-K words, norm-weight offset,
  state words; `build.py` emits them in `design_meta.h`.
- `src/program/npu.py`: the new external functions and their JIT source list.
- `src/program/sequence.py`: `normal` and `prenorm` sequence builders and the
  config fill that carries the norm vector.
- `src/program/worker.py`: hold/release the config object, sum pass, scale pass
  and the `CFG_PRENORM` branch.
- `src/program/run.py`: compose both builds and keep both JIT source lists.
- `src/program/build.py`: generate the normal and prenorm transaction headers
  from the same device design; build the xclbin once.
- `src/program/host/program.{h,cc}`: `PrenormSpec`, cached hi/lo vector, config
  fill and body selection.
- `integrations/llama.cpp/ggml-spiral.cpp`: matcher and A/B flag.
- `src/program/tests.py` / `host/test.cc` / `pack.py`: prenorm golden cases and
  references; keep the normal suite unchanged.

**Tests and acceptance.**

- Add `src/funcs/rmsnorm` standalone tests (see below): random and one-hot
  weight, K=1536, compare normalized bf16 output and cycles.
- Add a `prenorm` golden case (Q4_0 1536x512) and a Python reference that
  normalizes bf16 input in f32; then a full-suite run with prenorm off and a
  subset with it on. Add the two-body normal path to the 30-case manifest.
- Run the C++ `./test`, Python suite and `--head`; then llama.cpp A/B on E2B:
  greedy output, `SPIRAL_PROFILE_CPU=1` (RMS/MUL nodes gone),
  `SPIRAL_PROFILE_GRAPH=1` (wall/cpu/npu and graph count), and the 4-slot soak
  with Tctl/fans.
- Acceptance for the first milestone: normal suite unchanged; prenorm golden
  within its named policy; RMS/MUL nodes absent from the decode profile; no
  TDR; `.text`/stack fit; x4 step wall and Tctl/fan improve or at least do not
  regress.

**Risks and blockers.**

- Two-body build/selection is the main new machinery. Prototype it on a single
  shape before wiring the matcher.
- Holding the config object may reduce input buffering or deadlock the FIFO;
  the fallback is a 6 KiB buffer and input depth 1.
- Matcher safety: norm output may feed multiple consumers, or the residual add
  may sit between the norm and matmul. Require a single consumer and the
  expected op order, or defer those cases.
- Program memory and the overlay tail; the norm functions are linked into the
  normal image too.
- Numerics at the final output; do not trade golden tolerance for speed.

**Prelude status (2026-11-24).** The real two-body prelude works on the
production build. `build.py` emits `generate_txn.h` and
`generate_txn_prenorm.h`; `MatmulShape::prenorm` plus `PrenormSpec` select the
body and supply the f32 norm weight. The host converts the weight to
interleaved hi/lo bf16 and writes it into the config object at byte 64
(`CFG_PRENORM`, `CFG_PRENORM_EPS`, `CFG_PRENORM_INV_K`). The prenorm host body
sends the input fill set twice (sum set, then main set); the worker sums group
by group, finishes `r` hi/lo, and scales each main-loop group before the mmul.

Two constraints forced the storage design:

- The config object is a stream object. Holding it across the input stream
  makes the first sum acquire return the config slot, and releasing it lets
  the input DMA overwrite the weight. The worker therefore copies the weight
  into a stable per-compute L1 buffer (`program_prenorm_load`) and releases the
  config immediately.
- L1 data memory is exactly full at 64 KiB, so the weight buffer came from
  dropping the input FIFO to depth 1 (`inputs_l2.cons().forward(depth=1)`).

The real prelude is production-only in the device: the diagnostic build keeps
its probe kernels and cannot fit the four prenorm functions. `npu.py` aliases
the prenorm kernel handles to `program_zero` for the diagnostic build, and
`worker.py` guards every prenorm call with `production`.

Physical result (production, 2026-11-24): `.text` 16208 B of 16384 (98.9%),
stack 8192 B; max-shape TXN streams 973/1121 words (3892/4484 B);
`./test --prenorm-spike` is bit-identical to a host-normalized bf16 reference
through the same mmul (max error 0) on `q4_0_1536x512` and `q4_0_1536x2048`.
Interleaved 20-run totals are 92.1/94.9 us for 1536x512 and 126.4/129.9 us for
1536x2048, so the prelude costs ~3 us on the device. The normal suite and the
Python suite are unchanged.

Limits: the config object caps the norm weight at `K <= 2032` and the 8 KiB L1
buffer at `K <= 2048`; larger hidden sizes need a different weight container.
`.text` has ~176 B of headroom, so further worker work needs the overlay.

**Prelude as a fused-call input.** The FFN gate+up chain carries it through
`prepare_chain` op0; backend tasks and the attention gap are in
`integrations/llama.cpp/TODO.md`.

- [x] `PreparedWeights::Impl::weights_bo` is a `std::shared_ptr<xrt::bo>`.
      `Program::prepare_prenorm(base, spec)` clones the scalars, regenerates the
      TXN with `generate_txn_main_prenorm_sequence`, converts the gain and
      shares the BO. `--prenorm-spike` is bit-identical.
- [x] `prepare_chain` accepts the prelude on op0 from a prenorm handle,
      writes the gain into op0's config, and `txn_concat` handles the extra
      input fill. `--prenorm-spike` still bit-identical.
- [x] The backend standalone path uses `prepare_prenorm`; no duplicate BO.
- [ ] Extend `host/test.cc --prenorm-spike` with a prenorm chain case
      (`prepare_chain` with a prenorm op0), bit-exact against the
      host-normalized reference. The backend path is already exercised by
      E2B (`prenorm=595`, output identical).

Acceptance: a fused call carries the prelude once for its shared input, the
fused-call count is unchanged (`chains`/`ffn_chains`), and the standalone
suite still fits 16 KiB program memory.

### Kernel ownership: `src/funcs`

The norm work is a good moment to answer the ownership question, but only
partially. Recommendation:

- **Extract pure vector math, not dataflow.** Create `src/funcs/` for
  format-independent device functions: `rmsnorm/`, later `rope/`, `softmax/`,
  and `epilogue/` (GELU/GEGLU/softcap). `src/funcs/rmsnorm` should own the sum
  of squares, `invsqrt`, hi/lo combination and in-place scale, with its own
  `tests/` and cycle/error harness analogous to `src/quants/tests`.
- Keep `src/program/kernels` for the program shell: FIFO acquire/release,
  block assembly and carry, config/state, worker loop and tile dispatch. The
  prenorm wrapper (`program_prenorm_*`) stays there and includes the funcs
  header.
- Keep `src/quants` for quant formats, transport layout, repack and fused
  mmul. `src/funcs` must not know about Q4_0/Q6_K; it is activation/vector
  math.
- Start with `src/funcs/rmsnorm` now, because this plan needs an isolated
  oracle and cycle measurement before integration. Move the existing
  `program_output` softcap/GEGLU math to `src/funcs/epilogue` only after the
  norm link is stable; moving it during the norm work would put the verified
  path at risk.
- If `src/funcs` remains a single module after the norm link, that is still
  justified: the alternative is keeping reusable vector kernels inside a
  program-memory-constrained directory with no independent test surface.

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
