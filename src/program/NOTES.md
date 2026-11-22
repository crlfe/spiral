# `src/program` notes

Reference material for the fixed NPU matmul program: terminology, design,
data flow, test layout, measurements, and lessons from failed experiments.
Active tasks and the current work plan live in `TODO.md`; usage lives in
`README.md`.

## Goal and scope

One loaded NPU program (one xclbin) executes Q4_0 and Q6_K matmuls at
runtime-varying dimensions.  It is built from Python and called through a
small C++ wrapper.  The first target is the Gemma 4 E2B Q4_0 matmul set plus
its Q6_K output head.

The objective is productive inference on the NPU: correct output at usable
interactive rates while the machine stays quiet and cool.  The NPU is chosen
for efficiency, not peak throughput, so the CPU and iGPU figures in this file
are comparison baselines rather than targets to beat.  Correctness and
data-flow stability come before any throughput tuning.  A later phase may
split work between the NPU and the iGPU in llama.cpp, but that is parked until
the NPU path is closer to competitive; the fixed program keeps a per-call,
per-tensor interface so a second backend can own a disjoint tensor set.

Out of scope: the `per_layer_token_embd` and `token_embd` row lookups, and the
F16 `per_layer_model_proj` CPU projection.  Only the `token_embd` output-head
matmul is covered.

## Terminology

Several layers used "fused" for different things, so avoid that word here.
Use `weights_rows`, `weights_cols`, `inputs_len`, `outputs_len`,
`inputs_vecs`, and `outputs_vecs`; avoid M/N/K.

- **transport tile** — one packed 256-`weights_rows` x 16-`weights_cols`
  weight tile.
- **tile compute** — the per-tile unpack + `aie::mmul<4,8,8>` call.
- **tile epilogue** — `program_output`: transpose the native mmul tile to
  `[vec][16]`, optionally apply a tanh softcap, and store bf16.
- **weight stream** — the concatenated transport tiles for one compute
  within one run, in `[rows_step][band]` order.
- **stream drain** — the worker loop that consumes L1 blocks and invokes the
  tile compute.
- **band** — 512 consecutive `weights_cols`, handled by all 32 computes
  (8 groups x 4 computes x 16 columns).
- **rows_step** — 256 consecutive `weights_rows`.
- **input group** — four consecutive rows_step row slices, held as one input
  object.
- **chunk** — one invocation of the program covering a subset of bands
  (band chunking).

`tests/poc/stream` remains the reference for the stream drain and carry
logic.  It is not a scaffold to extend; the real kernels live here.

## Locked design

| item | value |
| :-- | :-- |
| groups x computes | 8 x 4 = 32 compute tiles |
| inputs_vecs | 4, zero-padded when fewer rows |
| weights_rows tile | 256 rows |
| weights_cols tile | 16 columns (one compute) |
| band | 512 weights_cols (8 groups x 4 computes x 16) |
| L1 stream block | 8 KiB |
| L2 stream block | 32 KiB |
| carry buffer | 4 KiB (largest tile is Q6_K at 3392 bytes) |
| weight FIFO depth | 2 per channel |
| max weights_rows | 12288 (48 rows_steps) |
| max weights_cols, Q4_0 single tensor | 12288 (24 bands) |
| max weights_cols, Q6_K output head | 262144 (512 bands) |
| max host packed tensor | ~333 MB (Q6_K head; raw golden is ~315 MB) |
| host weight BO | reusable per-run chunk streams, 32 MiB at 128 blocks (`host_only`) |
| logical input data | 96 KiB (weights_rows=12288, inputs_vecs=4) |
| L1 input object | 8 KiB (4 row slices x 2 KiB) |
| config object | one, first in the input FIFO |
| input groups per call | `ceil(rows_steps / 4)`, at most 12 |
| fixed Python input buffer | 64 input objects (1 config + 63 input groups) |
| max weight blocks per call | 128 (C++ splits into two <=64 fills; static Python factorizes) |
| max total tiles per call | 2304 design cap; weight-block cap is lower (~455 Q4_0 / ~309 Q6_K) |
| max chunk bands | 48 bands / 24576 columns |
| L1 output object | 12 KiB f32 accumulator + 6 KiB bf16 output (48-band chunk) |
| L2 output object | 24 KiB (4 computes x 6 KiB bf16, 48-band chunk) |
| max L3 outputs | 4 MiB (weights_cols=262144, inputs_vecs=4) |
| formats | Q4_0 and Q6_K; one per call, config-selected |
| call ordering | sequential calls, one matmul per call |
| alignment | transport tiles are 64-byte multiples; stream starts are tile multiples |

`weights_rows` must be a positive multiple of 256.  `weights_cols` is rounded
up to a whole band (512 columns); host packing writes zero weights for padded
columns and the caller ignores their outputs.

Buffer maxima are sized from the actual target-tensor set, not from
`max weights_rows * max weights_cols`: the largest single tensor is the Q6_K
output head (`weights_rows=1536`, `weights_cols=262144`), while the largest
Q4_0 tensor is `weights_rows=12288` or `weights_cols=12288`.  Combined column
calls are capped at 48 bands (24576 columns); the weight-block limit still
applies.  Do not allocate the product of the independent maxima.

The host weight BO holds one chunk of per-compute streams; the host
canonical packed tensor can be much larger.  The reusable chunk buffer keeps
L3 usage at tens of MB instead of the full 333 MB head.

## Config words

The config object is the first input-FIFO object.  It is filled by its own
shim fill from a dedicated config BO, so the input BO contains only input
groups; both fills drive the same producer channel, so the config fill is
finished before the input fill is issued.  The config object is still 8 KiB
because the input FIFO has a fixed object type, but only the first 64 bytes
are used.  It is copied into a per-compute buffer and decoded by the worker.

| word | value |
| --: | :-- |
| 0 | format/mode (ggml_type: 2 = Q4_0, 14 = Q6_K; 0xF0 = FIFO, 0xF1 = raw checksum, 0xF2 = tile checksum) |
| 1 | `tile_bytes` |
| 2 | `rows_steps` (`weights_rows` / 256) |
| 3 | `cols_bands` (bands in this chunk) |
| 4 | `weights_blocks` (L1 blocks per compute stream) |
| 5 | `total_tiles` (`rows_steps * cols_bands`) |
| 6 | `input_pad` (extra input-group objects drained by the dynamic path) |
| 7 | `weight_pad` (extra weight blocks drained by the dynamic path) |
| 8 | softcap hi/lo: two bf16 bits (`cap_hi | cap_lo << 16`); 0 disables |
| 9 | softcap reciprocal hi/lo: two bf16 bits (`inv_hi | inv_lo << 16`) |

`weights_blocks` covers the whole continuous stream:
`ceil(total_tiles * tile_bytes / 8192)`.  Padding exists only for the final
partial block.

The host splits `cap` and `1/cap` into bf16 hi/lo pairs; the epilogue adds
`hi + lo` back in f32.  This removes the device division while keeping about
16 bits of reciprocal precision.

## Data flow

```text
host canonical weights (up to ~333 MB for the Q6_K head)
  -> src/quants repack
  -> per-run per-compute chunk streams: [rows_step][band][tile_bytes]
  -> 32 reusable L3 weight streams, host layout [group][block][compute][L1]

inputs [rows_step][inputs_vecs][8] bf16
  -> one 8 KiB L1 input group per four rows, broadcast to all computes
  -> per-compute f32 accumulator [band_chunk][outputs_vecs][16]
  -> common epilogue: optional softcap, bf16
  -> chain layout: [N/8][outputs_vecs][8] in L3
  -> host boundary: CPU utility converts blocks to [outputs_vecs][outputs_len]
```

The worker receives one L1 weight block per acquire (8 KiB of concatenated
tiles), one config object filled separately, one input group per four rows,
a private f32 accumulator per compute
(zeroed before accumulation), a bf16 output object, and the decoded config.
The epilogue produces the output object after the block loop; the accumulator
is what persists across `rows_steps`.

The L1 input object is four rows_steps, not the whole `inputs_len`:
`weights_rows=12288` would need 96 KiB, which does not fit in compute-tile
data memory.  The host keeps the full input buffer in L3.

### Worker loop

The worker is one `range_` loop over `weights_blocks + num_groups + 2` units.
It carries three values as iter args:

- the current weight block memref,
- the current input-group memref,
- the current group index.

Each iteration:

1. If `tile_index` reached the current group limit, release the old input
   group, acquire the next one, and advance the group index.  This uses a
   result-producing `scf.if`; the memref is selected by the `scf.if` result.
2. If the block is exhausted and more tiles remain, release the old block,
   reset `block_pos`/`block_done`, acquire a fresh block, and select it with
   another result-producing `scf.if`.
3. If the active group still has tiles, run the block kernel for it; the
   kernel accumulates the f32 tile into the private accumulator.
4. Yield the updated block, input group, and group index.

Extra units are no-ops because the group index or tile index is past the
limit.  After the loop the final block and input group are released, then the
bf16 output object is produced by `program_output` (transpose, optional
softcap, f32 -> bf16) and released to the group join.

A block may span group boundaries.  Carry length and block position persist
across groups; only `block_pos`, `block_done`, and `ready` reset when a new
block starts.

### Direct read and carry

Complete tiles are read in place from the acquired L1 block
(`kTileSource=1`); the carry buffer is used only for tiles crossing a block
boundary.  The direct read roughly halved large-shape kernel time.

`program_assemble` is the split-tile path used by the probe/stream
diagnostics.  Q4_0 and Q6_K use `program_q4_block` / `program_q6_block`,
which run assembly, direct/carry selection, and `unpack_mmul` in one call per
L1 block.  The shared body lives in `program_block_impl.h`
(`program_block::run<Tile>`); each block kernel is a thin adapter that passes
a tile policy whose `apply` calls its format's `unpack_mmul<256, 16>`.  A
runtime function pointer is not an option: every target would have to be
linked (the Q4_0 pair already overflows program memory by 224 B when linked
together), and an indirect call cannot inline the dequantize+mmul body.  Block position, carry length, block-done, and tile index travel in
the state buffer so a group boundary can pause and resume within a block.

`MAX_TILES_PER_BLOCK` is 5: an 8 KiB block holds three Q4_0 tiles or two Q6_K
tiles, then still needs an attempt for the next prefix.

### Output layout

Each compute keeps a 12 KiB f32 accumulator, because one output tile is
reduced across all `rows_steps`.  After the block loop, the common
`program_output` epilogue reads each 64-float accumulator tile and writes a
64-element bf16 tile in the native chain order:

```text
accumulator tile: [col_half][vec][8 cols] f32
output tile:      [col_half][vec][8 cols] bf16
```

`program_output` also applies the optional tanh softcap
(`cap * tanh(x / cap)`) using the native bf16 `vtanh`, then stores bf16 with
round-to-nearest-even.  Config words 8 and 9 carry `cap` and `1/cap` as bf16
hi/lo pairs; the epilogue reconstructs both in f32, so there is no device
division.  The drain writes the chain layout `[N/8][outputs_vecs][8]`: each
group's 3-D drain tap places its 64-column slice into the shared N-major
buffer, so the next call's input fill can read the same buffer without a CPU
transpose.  The C++ API still exposes `[inputs_vecs][outputs_cols]`;
`unpack_outputs` is the host-side CPU utility that converts the chain blocks
back to vec-major.  Diagnostic builds use the same layout; the Python runner
inverts the drain mapping for its raw and tile-checksum paths.  The C++ API is
bf16-only; a caller that needs f32 widens the result itself.

The softcap is a named
reduced-precision mode: the bf16 vtanh is approximate, so its verification
policy is `max(1e-3, 5e-2 * column_scale)`, not the normal bf16 policy.

## Golden and test layout

`build.py` emits `build/golden/index.txt`, a manifest of 30 cases.  `./test`
loads the xclbin once and runs every listed case.  Coverage:

- Q4_0 attention/FFN shapes in both orientations up to 12288 rows or 12288
  columns, including `256x512`, `256x1536`, and 2048/4096/6144/12288 rows.
- Two explicit 256-column padded cases (`q4_0_1536x256p512`,
  `q4_0_2048x256p512`).
- Q6_K `512x1536`, `1536x2048`, `1536x4096`, `1536x6144`, `1792x5632`, and
  `1792x11264`.
- The full 262144-column Q6_K head as 11 band chunks, concatenated into one
  host tensor to check `weights_col_offset` / `outputs_col_offset`.
- Partition boundaries at 1/2/3/4/63/64/65/112/114/154/225 tiles and
  36/63/64/65/77/127/128 weight blocks.

Each case has canonical raw weights, bfloat16 inputs, and a NumPy reference
output.  The C++ wrapper repacks with the real `src/quants` headers, builds
the per-compute block-major streams, packs the input groups, runs, and
unpacks into logical `[inputs_vecs][weights_cols]` outputs.

### Golden verification tolerance

Outputs are checked per column against

```
tolerance = max(1e-3, 6e-3 * column_scale)
```

where `column_scale` is the largest `|expected|` in that output column over
the four input vectors.  An element passes when its error is within its
column's tolerance; the reported `worst` ratio is `error / tolerance`.

The device stores the epilogue result as bf16, so bf16 round-to-nearest now
dominates the error: a value can move by one bf16 ulp relative to the f32
reference when the f32 accumulations differ slightly across the rounding
boundary, i.e. up to ~2^-8 relative.  The 6e-3 relative term leaves margin for
that plus the fp32 accumulation error; the worst observed ratio across the 30
golden cases is ~0.65.  The softcap case uses its own named policy,
`max(1e-3, 5e-2 * column_scale)`, because the bf16 `vtanh` is approximate.

The absolute error is not a shape-independent quantity.  The kernel
accumulates the K terms in fp32; the mean absolute error grows as
~`1.2e-7 * K`, and the max abs error is about 0.0026 at K=2048, 0.0049 at
4096, 0.0071 at 6144, and 0.0127 at 12288.  That is the envelope of an
unbiased round-to-nearest accumulation, not a rounding-mode bug: the signed
mean is ~1-4% of the absolute mean, the `dev > ref` sign split is 46-52%, and
emulating a truncating accumulation is about 100x worse at K=12288.  A
sequential fp32 RTNE sum of the same terms reproduces the measured mean
within ~10%.

`src/quants/tests/mmul.py` still uses its f32 per-column bound.  The program's
bf16 epilogue uses the `6e-3` relative term and the same `1e-3` floor;
`run.py` applies the identical policy and checks the same 30 cases.
Do not replace this with a fixed absolute error again: it cannot fit both
K=1536 and K=12288, and it makes the effective relative bound depend on the
column's magnitude (loose for small columns, tight for large ones).

Python checks:

```bash
~/Projects/mlir-aie/ironenv/bin/python src/program/run.py
~/Projects/mlir-aie/ironenv/bin/python src/program/run.py --head
```

The Python runner uses the fixed-max layout (64 input objects: one config plus
63 input groups, and 128 weight blocks) for its own suite.  The C++ path uses
runtime partition counts.

`tests/poc/iter_args.py` is the focused device regression for the carried
memref and conditional acquire/release pattern.  It sums four objects
containing `1, 2, 3, 4` and reports `sum 10, last 4, count 4`.

## C++ API and prepared weights

`host/program.h` / `host/program.cc` own the loaded design, the instruction
stream, and all host packing.  `host/test.cc` is a manifest-driven golden and
timing driver.

- `Program::prepare` repacks a tensor or head chunk once and caches a compact
  block-major host buffer per handle.  A `WeightSegment` overload concatenates
  several tensors along `weights_cols` for gate+up or Q+K+V.
- `Program::run` packs inputs and executes the prepared handle.  Each
  `PreparedWeights` owns its own resident host weight BO and compact group
  stride, so multiple handles can remain resident and repeated runs skip the
  weight upload.
- Weight and output column offsets support one contiguous tensor split into
  band chunks.
- The weight, input, and output BOs are `host_only`: prepared weights live in
  host memory and the NPU reads them over the interconnect.  "Resident" means
  resident in host memory, not in NPU SRAM.  The instruction BO is cacheable.
- Each prepared handle owns its weight BO; there is no most-recently-used
  eviction.  Destroying a handle releases its BO.

## Build and cache notes

- Build: `~/Projects/mlir-aie/ironenv/bin/python src/program/build.py`, then
  `cd src/program/build && ./test`; timings with `./test --timing`.
- `compile_external_kernels` skips an existing `.o` even when its source
  changed.  `build.py` deletes the active build directory's `*.o` first so
  kernel edits always reach the xclbin.
- The inline-kernel JIT cache can also reuse stale objects.  Clear stale
  `~/.npu/cache` entries after editing an inline kernel
  (`tests/poc/README.md` documents this for `iter_args_*`).
- `design.py` and `run.py` are in the JIT `source_files` list, so edits to
  fixed constants or module-level helpers invalidate the cache automatically.
- `build.py` emits `design_meta.h` from `design.py`; the host uses it for
  buffer sizes, tile geometry, output layout, and config indices, so the host
  cannot drift from the Python design.
- `build.py` prepends the Peano/AIE bin directories to `PATH` because `aiecc`
  needs `opt` and the other toolchain binaries.
- The shim BD repeat/iteration dimension is capped at 64 in our fills (the
  PoC documents a separate 1023-element contiguous access limit).  Larger
  counts factor into two dimensions when possible and fall back to <=64
  chunks for primes.  Contiguous output drains use the wide contiguous length
  and are exempt.
- The C++ dynamic path uses straight-line runtime-sized fills with static BD
  assignment (`--aie-assign-runtime-sequence-bd-ids`).  Runtime loops plus
  `--aie-lower-dynamic-bd-pool` deadlocked against the FIFOs once live counts
  exceeded the FIFO depth; do not reintroduce them.

## Performance history

`./test --timing` breaks a call into prepare, upload, txn, input, kernel, and
unpack.  It also runs a `cycle` row (FIFO mode, fills + acquire/release only)
and a `raw` row (per-block transport checksum) on the diagnostic build.

- Before direct reads: Q4_0 1536x12288 real 1527 us / probe 1447 us; Q6_K
  1536x6144 real 1227 us / probe 1130 us.
- After direct reads: Q4_0 1536x12288 real 917 us / probe 793 us; Q6_K
  1536x6144 real 978 us / probe 844 us; Q4_0 1536x512 real 180 us / probe
  92 us.
- Cycle-only baseline (including launch): Q4_0 1536x12288 288 us, Q6_K
  1536x6144 241 us, Q4_0 1536x512 78 us.  The large-shape total rate is
  ~35-39 GB/s including launch; removing the ~70 us launch gives ~54 GB/s.
- Word-sized carry copies instead of a scalar byte loop: Q4_0 1536x12288
  real/probe/cycle 424/300/284 us, Q6_K 1536x6144 331/251/225 us.
- 4 KiB/16 KiB -> 8 KiB/32 KiB L1/L2 blocks: Q4_0 1536x12288 ~369 us, Q6_K
  1536x6144 ~291 us.  Q4_0 1536x512 regressed (~165 us vs ~134 us), so the
  block size is a large-shape tradeoff.
- Weight FIFO depth 4 exhausted the 24 BD IDs per channel.  Depth 3 built and
  passed but showed no clear gain over depth 2, so depth stays 2.
- Block compute + 4-row group alignment: Q4_0 2048x2048 ~122 us, Q6_K
  ~146 us, Q4_0 1536x12288 ~340 us, Q6_K 1536x6144 ~263 us.
- Single-loop continuous streams before the loop-idiom changes: Q4_0
  2048x2048 ~123 us, Q6_K 2048x2048 ~148 us, Q4_0 1536x12288 ~315 us,
  Q6_K 1536x6144 ~247 us, Q4_0 1536x512 ~133 us.
- Shared block loop: the assembly loop is now one non-template
  `program_block::run` with a format function pointer instead of an inlined
  template copy per format, and the pointer is the `unpack_mmul` instantiation
  itself.  Production `.text` 12192 -> 11104 bytes; Q4_0 block 4368 -> 3232
  bytes (32-byte thunk + 3200-byte `unpack_mmul`), Q6_K 5168 -> 4032 bytes
  (32 + 4000).  The shared loop is 1184 bytes.  Interleaved A/B with five
  warm-up and twenty timed runs: Q4_0 1536x512 78.9 -> 79.8 us, Q4_0
  1536x12288 285.0 -> 278.2 us, Q6_K 1536x6144 228.4 -> 223.9 us, Q4_0
  2048x2048 118.2 -> 117.8 us, Q6_K 2048x2048 141.2 -> 144.1 us.  The
  pointer call is inside the run-to-run noise.
- bf16 tile epilogue (`program_output`: transpose + optional softcap + f32 ->
  bf16): one extra pass over the 12 KiB f32 accumulator per compute per call.
  Production `--timing` (twenty runs; kernel times stay in the historical
  run-to-run spread): Q4_0 1536x512 ~71-79 us, Q4_0 1536x12288 ~267-280 us,
  Q6_K 1536x6144 ~210-230 us, Q4_0 2048x2048 ~103-113 us, Q6_K 2048x2048
  ~127-151 us.  `--head-timing` is ~7.3-7.7 ms kernel / ~7.7-8.0 ms total for
  the 11-chunk head.  Production now drains the native chain blocks and the
  host utility copies 8-column blocks: ~0.3 us for 1536x512, ~8 us for
  1536x12288, ~230 us for the head.  The llama.cpp backend converts those
  bf16 rows to f32 in its gather.
  Production `.text` moved 11104 -> 11872 bytes; diagnostic 14048 -> 14880
  bytes, leaving ~1.5 KiB headroom.

### Real-query throughput

Gemma 4 E2B Q4_0, native Q6_K head, 30-token prompt and a 300-token greedy
answer (`llama-completion -no-cnv --ignore-eos --temp 0`, one CPU fallback
thread; PM QoS off, iGPU SCLK capped at 2400 MHz, NPU pmode `turbo`).

| device | pp8 | tg16 | 300-token generation | per token |
| :-- | --: | --: | --: | --: |
| iGPU (`ROCm0`, gfx1150) | 383.1 t/s | 55.30 t/s | 55.11 t/s | 18.1 ms |
| CPU (`-ngl 0`, 12 threads) | 224.1 t/s | 35.34 t/s | 35.78 t/s | 28.0 ms |
| Spiral NPU | 111.5 t/s | 18.47 t/s | 17.78 t/s | 56.2 ms |

The NPU is the slowest of the three in throughput on this model and the iGPU
the fastest; that is context, not the objective (see "Goal and scope").
Temperature and fan behaviour are the acceptance criteria; the measurement
recipe and acceptance bar are summarised under "Efficiency acceptance" below.

All three paths stream the same ~1.4 GB/token of active weights; the NPU path
adds the ~80 us per-call floor (see the cost model below) plus the PCIe hop.  Decode makes ~240 NPU calls/token, so ~19 ms/token
is fixed launch/fill overhead.

`SPIRAL_CPU_THREADS=1` is best for the NPU path (2/4/8 give 16.8/14.5/10.5
t/s `tg16` and 106/95/68 t/s `pp8`); the CPU path peaks at 12 threads
(36.1 t/s `tg16`, 225 t/s `pp8`).

### Efficiency acceptance

The program is judged on productive-but-quiet behaviour, not peak throughput.
For every device comparison run record:

| signal | source |
| :-- | :-- |
| CPU temperature | `k10temp` `temp1_input` (Tctl) |
| iGPU temperature | `amdgpu` hwmon `temp1_input` |
| fan speed | both `hp` fans (`fan1_input`, `fan2_input`) |
| host CPU utilisation | `/proc/stat` busy delta |
| NPU clocks | `DRM_IOCTL_AMDXDNA_GET_INFO` (MP-NPU and H) |
| throughput | `pp8`, `tg16`, a 300-token generation |

Power is not measured.  Package RAPL is root-only in the measurement container
and socket power has to be integrated from a coarse sensor, so fan RPM and Tctl
are the user-visible proxy with far fewer approximations.

Procedure: cool to the idle baseline (fans off, Tctl below ~45 C), then run
a long greedy generation per device (thousands of tokens) while sampling every
0.5 s, and keep going until Tctl and fan RPM have been flat for
20 s so the peaks are steady-state rather than an artifact of a short task.
Report peak Tctl/fan and throughput, then let the device cool back to the idle
baseline (fan 0, Tctl within 2 C) before starting the next device.  Proposed
acceptance: the run stays at a usable rate with fan RPM within ~150 of idle and
Tctl within ~10 C of the idle reading.

No current path meets this bar (the quietest, FastFlowLM, still reaches 54 C
and 2.8k RPM), so treat it as the target rather than a pass/fail gate.

Measured on Gemma E2B with steady-state soaks: 3000 tokens per device, one
submission thread, sampling at 0.5 s, peaks taken after the 20 s plateau
window, then cool-back to the idle baseline (fan 0, Tctl within 2 C).  CPU and
iGPU run through llama.cpp as the comparison harness; the NPU runs through
this program with one CPU fallback thread; PM QoS off.

Benchmark baseline: the machine's normal boot state, the iGPU SCLK capped at
2400 MHz (`manual` DPM, OD bounds 600-2400 against a 600-3100 MHz hardware
range), platform profile `balanced`, CPU governor `powersave`, PM QoS off, and
NPU pmode `turbo`.  pmode is the one
performance knob that matters: it is worth ~11% on NPU decode (16.4 vs
14.7 t/s) at identical clocks and temperatures, and it does not survive a
reboot, so a `spiral-npu-tune` systemd unit re-applies it.  fclk and socclk are
left on DVFS; pinning them was measured to change nothing (14.64 and 14.65 t/s
against 14.70).  `tests/bench/soak/hostconfig.py` checks the controlled settings
and aborts on drift.

| engine | tokens | t/s | peak Tctl | peak iGPU | fan1 | fan2 | plateau | cool-back |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| iGPU (`ROCm0`) | 3000 | 52.42 | 54.2 C | 54.0 C | 2793 | 2597 | none in 59 s | 123 s |
| FastFlowLM (`gemma4-it:e2b`) | 3000 | 25.75 | 54.0 C | 48.0 C | 2818 | 2626 | none in 117 s | 194 s |
| CPU (12 threads) | 3000 | 22.17 | 91.0 C | 56.0 C | 4845 | 4657 | none in 138 s | 126 s |
| Spiral NPU | 3000 | 16.15 | 73.6 C | 56.0 C | 4834 | 4657 | 65 s | 121 s |

FastFlowLM runs its own quantisation and packing, so that row is a model-class
comparison; the other three share the same Q4_0 GGUF with the native Q6_K
head.  Its 3000-token run was driven through its OpenAI-compatible endpoint
with `tests/bench/soak/endpoint.py`.

The short-run "no fan" result for the iGPU and CPU was an artifact of task
length: both ramp the fans once the soak is long enough, so steady-state peaks
are the numbers to use.  The CPU row varied between 19.6 and 22.8 t/s across
runs at the same settings.

Both fans are on the `hp` hwmon and always move together, roughly 150-200
RPM apart, so neither belongs to a single sensor; they follow one EC curve.
The curve has a knee just above 53 C: a FastFlowLM run that peaked at 53.1 C
finished with both fans at zero, while the repeat at 54.0 C sat at 2.8k.  Above
that the EC jumps to ~4.8k for CPU-heavy runs.

That separates the two kinds of NPU heat.  FastFlowLM and the Spiral backend
run the same NPU at the same MP (compute) and H (host-interface) clocks
(1408/2112 MHz) and a similar decode
shape, yet Spiral peaks at 73.6 C / 4.8k RPM against FastFlowLM's 54.0 C /
2.8k RPM.  The NPU-only loop below shows the NPU alone lands at the lower
level, so the rest is the ~9 ms/token CPU fallback and host submission:
removing the softcap fallback should bring the Spiral path close to
FastFlowLM's fan behaviour.

The iGPU is the fastest path (52.4 t/s) and sits at 2.8k RPM.  The CPU is the
hottest at 91.0 C and throttles.  The NPU path sustains 16.2 t/s but spins both
fans to ~4.8k RPM because of its host-side work, not the NPU itself.

That is a property of this workload - ~240 NPU calls/token plus ~9 ms/token of
CPU fallback on a memory-bound model - not of NPU work in general; reducing
calls and the softcap fallback should move fan behaviour as much as
throughput.

An NPU-only busy loop (`./test --head-soak 20000`: repeated 11-chunk head
passes, one submitting thread) separates NPU heat from host work.  Over 152 s
the CPU was 1% busy, yet the sensors read Tctl 60.6 C, iGPU 49.0 C and fans
2818/2626 RPM.  So the NPU's heat appears in both the CPU control temperature
and the iGPU/SoC sensor and spins the fans with no CPU activity.  The same loop
runs ~4% faster under pmode `turbo` (152.3 s vs 155-159 s) at the same
temperatures; the head loop is less sensitive than decode, so this is not
the ~11% decode gain.  During real decode the fans are higher still (4.8k RPM)
because that path also runs the CPU fallback and host submission.

Kernel optimization levels were measured on the production build before the
loop-idiom changes.  Peano compiles kernels with `-O2` by default; per-kernel
`compile_flags` append after that and override it.  At the 10672-byte
baseline, `-O3` produced byte-identical Q4_0/Q6_K objects, the same `.text`,
and no timing change.  `-Os` produced different objects, grew `.text` to
10864 bytes, and was slower on the large shapes (Q4_0 1536x12288
~434-464 us vs ~349-357; Q6_K 1536x6144 ~359-372 us vs ~281-295).  Keep
`-O2`.

Loop-idiom changes (`program_zero`, carry copy):
- `program_zero` now uses a 16-float `aie::store_v` loop; the disassembly
  shows `vst x0` with post-increment instead of one scalar indexed store per
  word.  It grew from 192 to 528 bytes but runs once per call.
- The block/assemble carry copy now uses pointer increments.  The word loop
  compiles to `lda r4, [p0], #8` / `st r4, [p1], #8` instead of indexed
  loads/stores.  Only boundary tiles use it.
- Production `.text` moved from 10672 to 11232 bytes; diagnostic from 13328
  to 13984.
- Interleaved A/B runs show Q6_K 2048x2048 ~150 -> ~142 us, Q4_0 2048x2048
  ~124 -> ~122 us, and 512-column runs noisy but not regressed.  Large shapes
  were unchanged within run-to-run noise.
- Attempt-loop unrolling was rejected.  Full `#pragma unroll` overflows
  program memory by 800 bytes; `unroll_count(2)` fits at 12800 bytes but
  regresses (1536x512 ~138-142 us vs ~127-130, large shapes no better).
  Keep the rolled attempt loop.

### Production medians (historical)

This is a single-session snapshot (5 runs, no competing benchmarks) taken
before the head/dynamic-zero changes and before the current measurement
baseline existed.  Use it as a device-kernel reference, not as a live
comparison; the current baseline and procedure are under "Efficiency
acceptance".


| case | kernel med (us) | range |
| :-- | --: | --: |
| Q4_0 1536x512 | 135.6 | 125.0-137.5 |
| Q4_0 2048x2048 | 121.5 | 121.0-124.1 |
| Q6_K 2048x2048 | 146.5 | 144.7-150.5 |
| Q4_0 1536x12288 | 318.3 | 310.9-346.2 |
| Q6_K 1536x6144 | 249.7 | 241.4-274.9 |

Diagnostic probe/cycle medians were nearly identical, e.g. Q4_0 1536x12288
probe 262.1 / cycle 266.2 us, Q6_K 1536x6144 probe 216.6 / cycle 214.8 us.
Fill and launch dominate; assembly overhead is negligible.  The probe row was
later removed (see below): it tracked the real kernel within run noise.

Head chunk sweep, same xclbin and prepared weights, 3 runs each (total per
token):

| bands/chunk | chunks | total median (us) |
| --: | --: | --: |
| 25 | 21 | 9462.6 |
| 32 | 16 | 8764.7 |
| 40 | 13 | 8461.2 |
| 48 | 11 | 7984.1 |

The first attempt at 48 bands regressed large Q4_0/Q6_K matmuls because the
fixed output object grew and `program_zero` still cleared the whole object.
Making the zero count dynamic (`cols_bands * OUTPUT_TILE_ELEMS`) removed that
penalty.  Final production medians (5 warm runs):

| case | kernel med (us) | range |
| :-- | --: | --: |
| Q4_0 1536x512 | 128.8 | 126.9-142.1 |
| Q4_0 2048x2048 | 123.0 | 120.5-133.3 |
| Q6_K 2048x2048 | 143.9 | 142.1-153.3 |
| Q4_0 1536x12288 | 327.5 | 295.3-335.2 |
| Q6_K 1536x6144 | 233.7 | 225.1-263.7 |

Head timing with 48 bands and dynamic zero is ~8.2 ms per token
(11 chunks), versus ~9.1-9.5 ms at 25 bands in the controlled sweep.  Keep
`MAX_BANDS_PER_CHUNK = 48` and use it for `HEAD_GOLDEN`; per compute the f32
accumulator is 12 KiB and the bf16 epilogue output is 6 KiB (24 KiB per
group).  Program `.text` is 11872 bytes production / 14880 diagnostic, worker
stack 8192 bytes.  The diagnostic build leaves about 1.4 KiB of
program-memory headroom.

### Streaming gap versus the PoC

The PoC was re-run at the same large shapes with `l1_stream_bytes=8192`,
`work_bytes=3392`, `max_row_steps=6`, `max_col_steps=24`, `skip_copy=1`,
`skip_mmul=1`, and `--reps 64`:

| case | per rep | average | best |
| :-- | --: | --: | --: |
| Q4_0 1536x12288 | 194.8 us | 56.2 GB/s | 58.6 |
| Q6_K 1536x6144 | 138.9 us | 58.0 GB/s | 58.6 |

At 2048x2048 the PoC reports 52.8 GB/s (Q4_0) and 54.6 GB/s (Q6_K); the large
shapes above are slightly faster because fixed overhead is amortized.

Our cycle path (fills + FIFO only) is Q4_0 1536x12288 ~266 us and Q6_K
1536x6144 ~215 us, but that includes the per-call launch.  Subtracting the
~60-70 us launch floor gives ~196 us and ~145 us, i.e. ~54 GB/s, at or near
the PoC.  The streaming path is not the gap; per-call launch/fixed overhead
is.

`xrt::run` is now cached in `Program::Impl` and reused across calls instead of
constructing a run per call.  Cycle medians moved by roughly 2-16 us in the
expected direction, but the difference is within run-to-run noise.  Keep it
as a low-risk cleanup.

The remaining lever is reducing the number of calls or amortizing launch
overhead: fewer head chunks (done), parallel column combining (done for
gate+up and QKV), sequential-operation fusion (inp_gate+proj, FFN
gate+up+down), or a persistent worker queue.  Re-tuning weight-fill BD shape
is not justified by the measurements.

### Combined column matmuls

`Program::prepare` now accepts a `WeightSegment` array and concatenates the
segments along `weights_cols`.  Each segment keeps its own raw pointer,
stride, and 512-column offset.  Segments must share `format` and
`weights_rows`.  The host repacks each segment and assembles the combined
packed tensor per 256-row tile, then builds the usual block-major stream.
This supports independent column blocks over the same input without any new
device dataflow.  Output columns appear in segment order.

Measured with `./test --combined-timing` (prepared weights, production):

| combined call | columns | kernel us | total us | separate total |
| :-- | --: | --: | --: | --: |
| gate + up, 6144 | 12288 | 318.8 | 339.3 | 2 x 218.0 = 436.0 |
| gate + up, 12288 | 24576 | 548.9 | 595.1 | 2 x 342.2 = 684.4 |
| Q + K + V, sliding | 3072 | 139.8 | 148.0 | 121.6 + 2 x 84.2 = 290.0 |
| Q + K + V, full | 5120 | 178.8 | 189.6 | 169.4 + 2 x 84.2 = 337.8 |

The gate+up combined calls verify both output column segments against the
golden references.  QKV now verifies all three segments too; this works
because the golden inputs depend only on `weights_rows`, which is shared by
the K=1536 cases.

Per token this saves about 3.24 ms of wall time for gate+up and 2.15 ms for
QKV, taking the launch-limited estimate from 55.83 ms / 17.9 tok/s to about
50.44 ms / **19.8 tok/s** (kernel-only about 21.2 tok/s).  Launches drop from
286 to 221.

### Cold weight streaming

The llama.cpp backend (`integrations/llama.cpp`) sees each layer's weights once per token, so the warm
standalone benchmark understates the real per-call cost.  `./test
--cold-timing` prepares several resident BOs for the same shape and cycles
through them to defeat the last-level cache.  It reports min/p50/p95/max, and
`SPIRAL_COLD_BOS` / `SPIRAL_COLD_ITERS` control the BO count and iteration
count.

The important result is that per-call time is **linear in weight bytes plus a
fixed launch cost**, not a constant "GB/s".  Using the llama.cpp decode
medians for the same N=1536 output over different K:

| case | weight bytes | p50 kernel |
| :-- | --: | --: |
| 4096x1536 | 3.54 MB | 153.3 us |
| 6144x1536 | 5.31 MB | 194.5 us |
| 12288x1536 | 10.60 MB | 299.7 us |

A least-squares fit gives

    kernel_us ≈ 80 us + bytes / 48 GB/s

and it predicts the rest of the decode set to within a few percent
(1536x256 86.1 us, 1536x2048 116.3 us, 2048x1536 115.2 us).  The cold
standalone medians land on the same line (12288x1536 p50 289 us,
1536x6144 p50 182 us), so the model and the standalone test are consistent.

So naive `bytes / total_time` understates the streaming rate badly:

- 12288x1536 median: 10.6 MB / 300 us = 35 GB/s, but removing the 80 us
  launch gives ~48 GB/s.
- Decode averages lower per call because the fixed floor is amortised over
  smaller transfers.
- The PoC's 54-58 GB/s was streaming-only (`skip_mmul`, `skip_copy`) after
  subtracting a launch floor, so 48 GB/s incremental is only ~10-20% below it.

The fixed ~80 us floor is now the dominant inefficiency for small calls:
decode makes roughly 240 NPU calls/token, so ~19 ms/token is launch/fill
overhead.  That is the case for fewer, wider calls (the column-combining and
graph-level chaining work), not for more DMA bandwidth.

**Weight over-read guard.**  The compact per-handle weights BO initially
ended exactly at the last L2 block.  The device prefetches one block ahead
while streaming, so the last group read past the mapping and raised AMD-Vi
`IO_PAGE_FAULT`s (about 10 per 4-token run, visible in `dmesg`).  The BO now
has one zeroed `kL2StreamBytes` guard block after the data; a 4-token run
produces zero faults.  This is a stability fix, not a throughput item.

**Wait stalls.**  `wait_us` stays in the microsecond range: a 64-token run
has no call above 15 ms, median wait ~95 us, and the device job lifecycle
(`job pushed` -> `sent to device` -> `signaled fence`) is at most ~2.6 ms.
Kernel tracing of an earlier run that did show 45-50 ms `wait_us` spikes found
the submitting thread switched out while runnable and resumed ~47 ms later on
an idle CPU, with no driver, PM or IOMMU activity in the window: the spikes
were host CPU-quota throttling, not the NPU.  No stall workarounds are needed.

**CPU DMA latency / PM QoS.**  PM QoS is opt-in.  When `SPIRAL_DMA_LATENCY`
is set to a non-empty value other than `0`, `Program` opens
`/dev/cpu_dma_latency` (or `SPIRAL_DMA_LATENCY_PATH`), writes `uint32_t(0)`,
and holds the fd until the Program is destroyed, asking the kernel to keep the
CPU out of deep idle states while it waits on the NPU.  If the request is
enabled but the node cannot be opened or written, the Program throws instead
of silently continuing without it.  The default leaves the node alone.

The effect is workload-dependent (Gemma E2B, one CPU fallback thread, three
runs each, alternating order):

| metric | PM QoS on | PM QoS off |
| :-- | --: | --: |
| `pp8` | 118.1 t/s | 111.4 t/s (+6% on) |
| `tg16` | 16.84 t/s | 18.11 t/s (-7% on) |
| 300-token generation | 15.2-15.3 t/s | 16.8-17.3 t/s (-10% on) |

Prompt processing gains because it is host-submission heavy and avoids CPU
idle-state exits; decode loses because holding every CPU out of deep idle
raises CPU power draw and lowers boost clocks while the NPU is the bottleneck.
That is why PM QoS is off by default; enable it for prompt-heavy workloads.

**Idle gaps are expensive.**  Inserting a sleep between calls (NPU idle,
CPU awake) slows the next call, and the penalty grows with the gap, so the
model's interleaving of NPU calls with CPU fallback costs more than a
back-to-back loop would.  Keeping the NPU busy (look-ahead submission or
overlapping CPU work) is worth more than raw bandwidth.

Experiments that did **not** help:

- `xrt::bo` weight flags: only `host_only` is usable on this platform.
  `normal` is rejected (`unsupported buffer type`), and `cacheable` produces
  wrong results (host/device coherency).
- Increasing the per-group `weights_l2` object FIFO depth from 2 to 3 compiles
  but was neutral to slightly slower within run-to-run noise; depth 4 fails to
  compile.
- Depth 3 also made no difference to the 2048x1536 case, so the small-shape
  cost is not L2 buffering.

The next device experiment should target outstanding weight DMA rather than
cache flags or FIFO depth, and should be judged with `--cold-timing`.

### Dropped: Q4_0_packed16 in the program

`src/quants` still provides `q4_0_packed16`, a near-exact 288-byte Q4_0
variant (host-packed 16-bit scale codes, 9-bit fraction, named 5%-of-column
verification policy).  It was integrated as a build-time alternative block
(the two blocks cannot be linked together; the pair overflows program memory
by 224 B) and removed again after the exact Q4_0 kernel improved to 7883
cycles (against 7397 for packed16, a 6.2% gap) with the shared exponent-add
split.  packed16 is
near-exact rather than bit-exact, so it would need its own quality policy and
a wider sweep before replacing the canonical path.

The format and its `src/quants` tests remain for future work.  `src/program`
builds one Q4_0 block kernel again.

### Pipelined submissions

`Program` now has `submit()` / `finish()` and a small pool of independent
{instructions, inputs, outputs, `xrt::run`} slots.  A caller can start the
next call before waiting on the previous one, which overlaps the launch and
pipeline fill with the previous kernel.  `SPIRAL_PIPELINE_SLOTS` selects the
pool size (default 2, clamped to 8); `--pipeline-timing` runs one full head
pass through the queue.

Q6_K head (11 chunks), production, one full pass per iteration:

| slots | wall | kernel sum |
| --: | --: | --: |
| 1 | 7.7-8.2 ms | 7.2-7.5 ms |
| 2 | 6.5-6.9 ms | 5.9-6.2 ms |
| 4 | 6.5 ms | 5.9 ms |
| 8 | 6.6 ms | 6.0 ms |

Two slots capture nearly all of the win (~16% wall, ~17% device time) because
the launches are short relative to a chunk; more slots only add memory and
queue pressure.  In the llama.cpp backend the head p50 kernel drops from
~7.65 ms (1 slot) to ~6.87 ms (2).  Prompt processing benefits too because
every matmul with `rows > 4` has multiple row chunks; `tg` decode is mostly
one call per node, so it only gains the head portion.

The pool does **not** make runs execute concurrently; the partition is fully
occupied by one call.  It removes the serial launch bubble, which is what the
old single-run path paid on every call.

`Program::max_outstanding()` is the high-water mark of submitted-but-not-
finished calls (printed as `max_slots=` by `SPIRAL_PROFILE`).  A 64-token run
with `SPIRAL_PIPELINE_SLOTS=1/2/4` gives `max_slots=1/2/4`, so the backend
never has more than the configured number in flight, confirming the
submissions stay within the configured depth.

In llama.cpp the backend pipelines every multi-chunk node.  The head is the
main decode beneficiary; prompt processing also gains because nodes with
`rows > 4` split into row chunks:

| slots | pp8 | tg16 |
| --: | --: | --: |
| 1 | 95.3 t/s | 18.0 t/s |
| 2 | 112.5 t/s | 18.3 t/s |
| 4 | 111.6 t/s | 16.4 t/s |

Prompt gains ~18%; decode is unchanged because only the head is multi-chunk
at `rows=1`.

A second experiment submits a later same-input matmul (for example K or V
while the host runs Q's norm/RoPE) and defers only its host unpack, so the NPU
stays busy during CPU fallback.  It is correct and available with
`SPIRAL_LOOKAHEAD=1`, but on Gemma it is within noise of the plain pipeline,
so it is off by default.  The offsetting cost is queue ordering: submitting
the next call before the current one delays the current result, while
submitting it after the current call only overlaps the CPU gap.

### What still needs a new design

- **inp_gate + proj:** sequential, not parallel columns.  `inp_gate` produces
  a 256-wide intermediate, then `proj` consumes it.  A single launch needs a
  two-stage worker that keeps the intermediate on device, applies the gate
  activation, and broadcasts the 256 values before `proj`.  The current
  single-matmul worker cannot express that.  Separate cost is ~158.9 us per
  layer, 35 layers = ~5.56 ms.
- **gate + up + down:** also sequential.  `down` needs the full 6144/12288
  intermediate, whose columns are spread across all groups.  A single launch
  requires a global barrier, intermediate broadcast, and device activation;
  this is a persistent FFN design, not a column concatenation.
- **Q6_K head as one launch:** needs `weights_blocks = 1272` for 512 bands
  (`ceil(6 * 512 * 3392 / 8192)`).  The current cap is 128.  Expanding means:
  `MAX_WEIGHT_BLOCKS = 1272`, `MAX_TOTAL_TILES >= 3072`, a weight BO of about
  333 MB (8 groups x 1272 x 32 KiB), and roughly 20 straight-line fill parts
  per group instead of 2.  That is a large XRT/BD/instruction-stream change;
  the current 11-chunk head is the safe point.  A smaller expansion to 256
  blocks / 4 parts with `MAX_BANDS_PER_CHUNK=64` would allow 64-band chunks
  and 8 head launches, at 64 MiB of weight BO.

### Launch-limited Gemma E2B estimate

One launch per matmul, prepared weights, steady-state `--all-timing`
latencies, non-matmul work ignored.  E2B has 35 blocks: 28 sliding-window
blocks (Q 2048, KV 256 padded to 512) and 7 full-attention blocks (Q 4096,
KV 512); only layers 0-14 own KV due to `shared_kv_layers=20`; FFN is 6144
for layers 0-14 and 12288 for layers 15-34.  The Q6_K head is 11 chunks at
the current 48-band limit.

| group | launches | kernel ms | wall ms |
| :-- | --: | --: | --: |
| attn_q 2048 / 4096 | 28 / 7 | 3.19 / 1.11 | 3.40 / 1.19 |
| attn_output 2048 / 4096 | 28 / 7 | 3.14 / 1.07 | 3.34 / 1.13 |
| attn_k + attn_v | 30 | 2.39 | 2.53 |
| inp_gate | 35 | 2.79 | 2.95 |
| proj | 35 | 2.43 | 2.61 |
| ffn_gate + ffn_up 6144 | 30 | 6.01 | 6.54 |
| ffn_gate + ffn_up 12288 | 40 | 12.61 | 13.69 |
| ffn_down 6144 / 12288 | 15 / 20 | 3.07 / 6.41 | 3.29 / 6.83 |
| Q6_K head (11 chunks) | 11 | 7.76 | 8.33 |
| **total** | **286** | **51.97** | **55.83** |

That is **19.2 tok/s kernel-only** or **17.9 tok/s** including txn/input/unpack
per call before column combining.  With gate+up and QKV combined it is about
19.8 tok/s (221 launches).  The memory-bound estimate is about 1.381 GB/token
at 55 GB/s = 25.1 ms/token, or 39.8 tok/s.  Launch/fixed overhead is therefore
the dominant limit: removing it would roughly double token throughput.
Batching or a persistent worker queue is the only way to recover that gap.

## Kernel loop idioms

A standalone 64-byte C++ copy experiment found that the source shape of a loop
matters more than backend flags.  In particular, walking
pointers (`src += 64`, `dst += 64`) lets Peano emit post-increment addressing
and discover a single-register load/store rotation; indexed loops generate
explicit index updates and are several times slower.  `unroll(enable)` and
`unroll_count(N)` can recover part of the gap, and `unroll(full)` is fastest
but grows `.text` substantially.  The common `aie-*` unroll, pipeliner, and
register flags had no effect on the copy kernels.  See
`src/quants/NOTES.md` under “Loop optimization and Peano idioms” for the
measured table, ZOL constraints, and VLIW latency references.

Counterexample from the same notes: the q4_0 dequant+mmul loop is
recurrence-bound, not address-bound.  Advancing the input or record pointers
regressed it 2–4%; only the payload pointer was neutral.  Do not rewrite
`unpack_mmul`'s record/input loops to pointer form without measuring both
forms.  In `src/program`, pointer increments were applied only to the
address-bound copy paths (`program_zero`, carry copy), where the disassembly
confirms post-increment addressing.

## Failed or rejected experiments

- **Nested carried-memref loops.**  The first no-alignment attempt carried
  the weight block through an outer row-group loop and an inner block loop.
  It compiled but deadlocked on device for every shape.  A single loop
  carrying block, input group, and group index works.  Treat nested
  carried-memref loops as unsafe with the ObjectFifo transform until proven
  otherwise.
- **Conditional acquire inside the ready branch.**  Per-row input was first
  written with an acquire nested in the compute branch; the ObjectFifo
  transform rejected it.  The supported form is to carry the acquired memref
  as an `scf.for` iter_arg and use a result-producing `scf.if` for the
  conditional release/acquire.  `tests/poc/iter_args.py` validates this on
  device.
- **Output-drain trimming.**  A `drain` offset is a destination offset; the
  source is always the whole group object, so per-compute drains repeated
  compute 0.  The 4D `[computes][bands][64]` packed drain also wrote only the
  first compute.  Trimming needs per-compute runtime consumers or a different
  source access pattern.  Keep drains contiguous (full group object).
- **Weight FIFO depth 4.**  Exhausted the BD ID budget.
- **Larger L1/L2 blocks.**  Helped large shapes but regressed small ones;
  8 KiB / 32 KiB with five attempts is the current compromise.

## Stream PoC design differences

`tests/poc/stream` remains useful as a cross-check, but the measured
comparison is in “Streaming gap versus the PoC” above.  Structural
differences:

- the PoC uses one block kernel per L1 block; the production path now does the
  same for Q4_0/Q6_K,
- the PoC holds one large input object for the call; we hold one 8 KiB input
  group per four rows,
- the PoC defaults to 2 KiB L1 / 8 KiB L2; our tuned size is 8 KiB / 32 KiB,
- the PoC benchmarks with `--reps 64`, which amortizes the launch that the
  real per-call path pays.

With launch removed, our large-shape cycle path is at or near the PoC.  The
remaining real-kernel difference is quant compute, not fill throughput.

## Implementation notes

- Measure compute-tile program memory from the linked ELF:
  `llvm-size -A <build-dir>/xclbin_build/elfs_main_core_*/*.elf`.
  Diagnostic builds report `.text` = 14880 bytes on all 32 cores; production
  builds report 11872 bytes (73% of the 16 KiB limit).  Diagnostic overhead is
  `program_assemble`, `program_probe`, `program_raw_check`, the split-path
  worker code, and diagnostic-only branches.
- `build.py --production` builds into `build-production/` from
  `production_dynamic_program`.  It links only `program_begin`,
  `program_zero`, `program_output`, `program_block_begin`,
  `program_q4_block`, and `program_q6_block`.  It rejects the diagnostic mode
  range and skips the cycle timing row.  The normal build keeps the diagnostic
  split path.
- Unused device functions and kernels should still be removed from `run.py`:
  they slow JIT/build, appear in the source-file cache set, and hide which
  kernels the worker actually calls.  The old split `program_q4_0` /
  `program_q6_k`, `program_row_begin`, and the unused `full_blocks` /
  `last_blocks` math were removed.  The block kernels own the production
  path; `assemble`/`probe` remain for diagnostics.  `program_stream`/format 3
  were removed: its timing tracked the real kernel within run noise, so the
  tile-checksum probe and the FIFO cycle floor are the useful diagnostics.
- `else_` is a context manager that takes the `if_` handle:
  `with if_(cond) as branch:` then `with else_(branch):`.
- Result-producing `scf.if` is needed when a memref must be selected; plain
  `if_`/`else_` cannot yield memory objects.
- XRT `xrt::bo::write` takes `(src, size, offset)`.
- bfloat16 inputs must be reinterpreted, not cast: `.view(np.uint16)`.  A
  `dtype=np.uint16` cast converts numerically and produces garbage bit
  patterns.
- `@iron.jit` rejects unannotated scalar parameters (no runtime-scalar
  plumbing), even though `Runtime`/`fill` support runtime values.
- Host weight padding is per group, not per block: each group stores
  `weights_blocks * COMPUTES_PER_GROUP * L1` bytes contiguously, then zeroes
  up to `MAX_WEIGHT_BLOCKS`.
- `pack.py` owns case construction: config object, input groups, block-major
  weights, and reference data.  `run.py` pads into the fixed host buffers and
  verifies.
- `Program::prepare` takes canonical quant bytes, a raw column stride, and a
  512-column weight offset.  `Program::run` takes bfloat16 bits in
  `[inputs_vecs][weights_rows]` order and returns bfloat16 bits in
  `[inputs_vecs][outputs_cols]` order.
