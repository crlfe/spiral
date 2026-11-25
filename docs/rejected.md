> Role: reference (negative results). Current as of 2026-11-24. History: `git log -S '<keyword>'`.

# Rejected experiments

Things that were tried and lost, with the reason. Read this before spending time
on an optimisation or a format variant: most of them look reasonable in
isolation.

Each line names the mechanism, so `git log -S '<mechanism>'` reaches the
original change. Numbers, where they matter, are in
[measurements.md](measurements.md) or were left in the commit.

## Kernel loops and Peano

- **Pointer increments in the q4_0 dequant+mmul loops.** Regressed 2-4%: the
  loop is recurrence-bound, not address-bound, and loop-carried pointer state
  disturbs the existing schedule. Measure both forms for every new
  dequant+mmul. The same rewrite wins on address-bound copy loops.
- **A fixed-shape rewrite of q4_0** (pointer form, no outer column loop,
  shift-based fp16 decode, manual record unroll) reached 2316 cycles against
  2204 for the existing schedule.
- **`aie-*` backend tuning flags.** No effect on copy kernels: varied neutral
  results from `-aie-unroll-max-count`, `-aie-unroll-only-pragma-loops`,
  `-aie-prefer-swp-over-unroll`, `-aie-pipeliner-track-regpressure`,
  `-aie-pipeliner-max-stagecount`, `-aie-postpipeliner-target-ii`,
  `-aie-postpipeliner-force-stagecount`, `-aie-aggressive-realloc`,
  `-aie-reg-rewrite-mode`, `-aie-stack-minimize`, `-aie-merge-congruent-ivs`,
  `-aie-loop-idiom-unrolling-threshold`, `-aie-unroll-max-cost`. Use flags only
  after the disassembly shows a specific problem.
- **`-O3`** produced byte-identical kernels and cycles. **`-Os`** grew `.text`
  and was slower on large shapes. Keep the default (`-O2`).
- **Hand-written assembly for simple copies** is competitive but not faster
  than the pointer loop and uses more text.
- **Full unrolling** of the program's attempt loop overflows program memory;
  `unroll_count(2)` fits but regresses. Keep it rolled.
- **Partial unrolling (`unroll_count`) or pragma changes applied generally.**
  Per-loop effects vary; sweep the actual loop and measure.
- **`noinline` on `next()` / the mmul helper**, **struct-based record views**,
  **four columns concurrently**, **multiple accumulators per column**,
  **extra accumulators hoisted across records**, **`__restrict` on the fused
  kernel pointers**, **copying the input tile before both mmuls**, and
  **dequantizing all tiles before issuing mmuls** all lost to the current
  two-column direct-pointer form.
- **Column-sequential ordering** (+1.0%) and **forcing the input tile into a
  register** (+1.1%) were slower on Q6_K.
- **Bitwise three-term Q6_K scale split** is exact but slower than the hardware
  conversion round-trip.
- **Integer two-term Q6_K scale path** is exact and slower than bf16
  three-term: the int32 -> bf16 conversion costs more than the bf16 multiply it
  replaces, and preparation would need to be under ~26 cycles per group.
- **Host-precomputed three Q6_K bf16 scale terms** was 1.375 bytes/weight,
  worse than the exact int8 layout at the 32-core memory budget.
- **Applying the scale after the mmul** was slower and outside tolerance.
- **Element-wise f32 `aie::mul`** (emulated) is 3-8x slower.
- **`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`** is ~25% faster and loses
  exactness (0.5-0.9% relative). Only usable as a named relaxed mode.
- **`AIE2_FP32_EMULATION_ACCURACY_FAST` / `_LOW`** did not change element-wise
  timings; they remap different helpers than the AIE API uses.
- **`aie::accumulate` and `aie::sliding_mul_ch` for per-column scales.**
  `accumulate` is uniform-scale only; `sliding_mul_ch` is integer-only on XDNA2
  and needs `CoeffStep = 1`.
- **Host pre-biasing float16 scales by `2^10`** removes the subnormal case but
  not the device work.
- **Scalar packed immediately before an aligned tile load** silently reads the
  wrong address; reserve aligned blocks.
- **`grow_replicate` / `concat` ladders / `.rodata` constants** for Q8_0 scale
  expansion all grew `.text`; `broadcast_vector` is the chosen form.
- **`broadcast_elem_128` register-lane `vextbcst.128` for Q4_0 scale
  expansion.** It regressed `q4_0` (7883 -> 7929 cycles) and `q4_0_packed16`
  (7397 -> 7573), so Q4_0 keeps `broadcast_vector`. The same edit does win for
  Q6_K's subgroup expansion (`q6_k_int8` 9101 -> 8440, `q6_k` 10180 -> 10115)
  because it drops the `vshift` that `broadcast_vector` emits before a non-zero
  lane; Q6_K uses it. Not related to the Q8_0 case above: there the index is
  zero, so there is no shift to remove.
- **`MEMORY_STALL` trace event** never fires for any current probe; revisit
  only for an unexplained cycle gap that looks like bank contention.

## Formats and variants

- **`q4_0_packed16` as a program variant.** It is near-exact (0.78% weight
  error, ~3.9% worst-case mmul error) with a named 5%-of-column policy, so it
  cannot serve as the unprefixed production Q4_0. After the exact path improved
  to a 6.2% kernel gap it was removed from `src/program`; the format and its
  tests stay in `src/quants`.
- **`q4_0_hilo` as production.** Exact and faster per cycle, but 320-byte
  records (5.0 bits/weight) lose the bandwidth advantage of the canonical
  288-byte layout.
- **A lossy Q8_K head** is slightly smaller than the int8 Q6_K layout but has
  coarser scales; Q8_0 is the better lossy target if one is ever wanted.
- **Two-term bf16 scale split for Q6_K** is ~3% faster and not exact for
  canonical scales up to 127.
- **Four column tiles** for Q8_0 was ~65% slower from register pressure.

## Program dataflow

- **Nested carried-memref loops** compiled but deadlocked on device. Use one
  loop carrying block, input group and group index.
- **Conditional acquire nested inside the compute branch.** The ObjectFifo
  transform rejects it; carry the acquired memref as an `scf.for` iter_arg and
  use a result-producing `scf.if`.
- **Output-drain trimming.** A drain offset is a destination offset; trimming
  needs per-compute runtime consumers. Keep drains contiguous.
- **A 4D `[computes][bands][64]` packed drain** wrote only the first compute.
- **Weight/output FIFO depth 4** exhausted the 24 BD IDs; depth 3 was neutral.
- **Larger L1/L2 blocks (16/64 KiB)** helped large shapes but regressed
  small ones; 8/32 KiB is the compromise.
- **Runtime loops plus `--aie-lower-dynamic-bd-pool`** deadlocked once live
  counts exceeded FIFO depth. Keep straight-line fills with static BD
  assignment.
- **`xrt::bo` weight flags:** `normal` is rejected, `cacheable` produces wrong
  results. Only `host_only` is usable.
- **Cross-node look-ahead** is correct but within noise on Gemma; opt-in only
  (`SPIRAL_LOOKAHEAD=1`).
- **PM QoS on by default:** helps prompt (+6% `pp8`), hurts decode (-7% `tg16`,
  -10% on a 300-token answer). Opt-in only.
- **The Python runner's fixed-max layout** is kept for tests; do not treat it
  as the runtime partition path.

## llama.cpp backend

- **Chaining `MUL_MAT`s across intervening ops.** Correct in isolation, wrong
  in the model: the scheduler aliases compute buffers by graph-order lifetime,
  so computing a later matmul early clobbers a gap tensor.
- **Q/K/V as one combined call** is blocked by the QK-norm/RoPE gaps for the
  same reason.
- **NPU + iGPU layer split (parked).** Layer assignment works, but Gemma
  permutes views across the boundary and the Spiral CPU fallback cannot consume
  ROCm buffers. `SPIRAL_MATMUL_ONLY=1` removes the errors but the scheduler
  then gives every Q4_0/Q6_K matmul to ROCm, so the run measures the iGPU.
- **More `SPIRAL_CPU_THREADS`.** Thread overhead across ~900 fallback calls per
  token outweighs the faster tanh; one thread is the default.
- **A local async mode** showed no gain and is not kept.
- **The old `fw/` bf16 packer and its 2048x2048 packing path** are gone; the
  backend calls `Program::prepare` and must not resurrect them.

## Measurement and tooling

- **Short generation runs** for thermal comparisons: shorter than the thermal
  time constant, so the iGPU and CPU look fanless. Soak until the plateau.
- **`bytes / total_time`** as a bandwidth figure: it hides the ~80 us launch
  floor and understates streaming by ~30%.
- **Cross-boot absolute cycles** are usually close but not guaranteed; prefer
  per-value ratios from one session.
- **pinning fclk/socclk** changes nothing measurable.
