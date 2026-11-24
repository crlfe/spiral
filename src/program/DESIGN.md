> Role: reference. Current as of 2026-11-24. History: `git log -- src/program/DESIGN.md`.

# Fixed program design

Reference for `src/program`: one loaded NPU design (one xclbin) that runs Q4_0
and Q6_K matmuls at runtime-varying dimensions, built from Python and called
through a small C++ wrapper. Usage and the C++ API are in `README.md`; open work
is in `TODO.md`; numbers are in
[`../../docs/measurements.md`](../../docs/measurements.md).

## Goal and scope

One loaded program executes Q4_0 and Q6_K matmuls at runtime-varying dimensions.
The first target is the Gemma 4 E2B Q4_0 matmul set plus its Q6_K output head.

The objective is productive inference on the NPU: correct output at usable
interactive rates while the machine stays quiet and cool. The NPU is chosen for
efficiency, not peak throughput, so CPU and iGPU figures are baselines rather
than targets. Correctness and data-flow stability come before throughput.

Out of scope: the `per_layer_token_embd` and `token_embd` row lookups, and the
F16 `per_layer_model_proj` CPU projection. Only the `token_embd` output-head
matmul is covered.

A later phase may split work between NPU and iGPU in llama.cpp, parked until the
NPU path is closer to competitive. The fixed program keeps a per-call,
per-tensor interface so a second backend can own a disjoint tensor set.

## Terminology

Several layers used "fused" for different things, so avoid that word. Use
`weights_rows`, `weights_cols`, `inputs_len`, `outputs_len`, `inputs_vecs`,
`outputs_vecs`; avoid M/N/K.

- **transport tile** — one packed 256-`weights_rows` x 16-`weights_cols` weight
  tile.
- **tile compute** — the per-tile unpack + `aie::mmul<4,8,8>` call.
- **tile epilogue** — `program_output`: transpose the native mmul tile to
  `[vec][16]`, optionally apply a tanh softcap or GEGLU, store bf16.
- **weight stream** — concatenated transport tiles for one compute within one
  run, in `[rows_step][band]` order.
- **stream drain** — the worker loop that consumes L1 blocks and invokes tile
  compute.
- **band** — 512 consecutive `weights_cols`, handled by all 32 computes
  (8 groups x 4 computes x 16 columns).
- **rows_step** — 256 consecutive `weights_rows`.
- **input group** — four consecutive rows_step row slices, held as one input
  object.
- **chunk** — one invocation covering a subset of bands.

`tests/poc/stream` remains the reference for the stream drain and carry logic.
It is not a scaffold to extend; the real kernels live here.

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
| max host packed tensor | ~333 MB (Q6_K head; raw golden ~315 MB) |
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

`weights_rows` must be a positive multiple of 256. `weights_cols` is rounded up
to a whole band (512 columns); host packing writes zero weights for padded
columns and the caller ignores their outputs.

Buffer maxima are sized from the actual target-tensor set, not from
`max weights_rows * max weights_cols`: the largest single tensor is the Q6_K
head (`weights_rows=1536`, `weights_cols=262144`), while the largest Q4_0 tensor
is `weights_rows=12288` or `weights_cols=12288`. Combined column calls are
capped at 48 bands. **Do not allocate the product of the independent maxima.**

The host weight BO holds one chunk of per-compute streams; the canonical packed
tensor can be much larger. The reusable chunk buffer keeps L3 usage at tens of
MB instead of the full 333 MB head. It carries one zeroed `kL2StreamBytes` guard
block after the data because the device prefetches one block ahead; without it
the over-read raises AMD-Vi `IO_PAGE_FAULT`s.

## Config words

The config object is the first input-FIFO object, filled by its own shim fill
from a dedicated config BO so the input BO contains only input groups. Both
fills drive the same producer channel, so the config fill finishes before the
input fill is issued. The object is 8 KiB because the FIFO has a fixed object
type; only the first 64 bytes are used. It is copied into a per-compute buffer
and decoded by the worker.

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
| 10 | tile epilogue kind: 0 = none, 1 = softcap (words 8-9), 2 = GEGLU pair |

`weights_blocks` covers the whole continuous stream:
`ceil(total_tiles * tile_bytes / 8192)`. Padding exists only for the final
partial block.

The host splits `cap` and `1/cap` into bf16 hi/lo pairs; the epilogue adds
`hi + lo` back in f32, removing the device division while keeping about 16 bits
of reciprocal precision.

GEGLU is for a two-segment gate+up call: `cols_bands` covers the combined
width, the first half is gate and the second half is up. The epilogue reads both
bands from the same compute's accumulator, computes `gelu(gate) * up` in bf16
with the native bf16 `vtanh`, and stores only the gate bands. The drain still
writes `cols_bands` bands, so the gate half is the packed next-matmul input at
offset 0. The three epilogue kinds are exclusive; the softcap is an explicit
kind (word 10 = 1) rather than being inferred from a nonzero cap word.

## Data flow

```text
host canonical weights (up to ~333 MB for the Q6_K head)
  -> src/quants repack
  -> per-run per-compute chunk streams: [rows_step][band][tile_bytes]
  -> 32 reusable L3 weight streams, host layout [group][block][compute][L1]

inputs [rows_step][inputs_vecs][8] bf16
  -> one 8 KiB L1 input group per four rows, broadcast to all computes
  -> per-compute f32 accumulator [band_chunk][outputs_vecs][16]
  -> common epilogue: optional softcap or GEGLU, bf16
  -> chain layout: [N/8][outputs_vecs][8] in L3
  -> host boundary: CPU utility converts blocks to [outputs_vecs][outputs_len]
```

The worker receives one L1 weight block per acquire (8 KiB of concatenated
tiles), one config object filled separately, one input group per four rows, a
private f32 accumulator per compute (zeroed before accumulation), a bf16 output
object, and the decoded config. The epilogue produces the output object after
the block loop; the accumulator persists across `rows_steps`.

The L1 input object is four rows_steps, not the whole `inputs_len`:
`weights_rows=12288` would need 96 KiB, which does not fit compute-tile data
memory. The host keeps the full input buffer in L3.

### Worker loop

The worker is one `range_` loop over `weights_blocks + num_groups + 2` units,
carrying three iter args: the current weight block memref, the current
input-group memref, and the current group index. Each iteration:

1. If `tile_index` reached the current group limit, release the old input group,
   acquire the next one, and advance the group index. This uses a
   result-producing `scf.if`; the memref is selected by the `scf.if` result.
2. If the block is exhausted and more tiles remain, release the old block, reset
   `block_pos`/`block_done`, acquire a fresh block, and select it with another
   result-producing `scf.if`.
3. If the active group still has tiles, run the block kernel for it; the kernel
   accumulates the f32 tile into the private accumulator.
4. Yield the updated block, input group, and group index.

Extra units are no-ops because the group or tile index is past the limit. After
the loop the final block and input group are released, then `program_output`
produces and releases the bf16 output object to the group join.

A block may span group boundaries. Carry length and block position persist
across groups; only `block_pos`, `block_done` and `ready` reset when a new block
starts.

### Direct read and carry

Complete tiles are read in place from the acquired L1 block (`kTileSource=1`);
the carry buffer is used only for tiles crossing a block boundary. The direct
read roughly halved large-shape kernel time.

`program_assemble` is the split-tile path used by probe/stream diagnostics.
Q4_0 and Q6_K use `program_q4_block` / `program_q6_block`, which run assembly,
direct/carry selection and `unpack_mmul` in one call per L1 block. The shared
body is `program_block_impl.h` (`program_block::run<Tile>`); each block kernel is
a thin adapter passing a tile policy whose `apply` calls its format's
`unpack_mmul<256, 16>`. A runtime function pointer is not an option: every
target would have to be linked (the Q4_0 pair already overflows program memory
by 224 B when linked together), and an indirect call cannot inline the
dequantize+mmul body. Block position, carry length, block-done and tile index
travel in the state buffer so a group boundary can pause and resume within a
block.

`MAX_TILES_PER_BLOCK` is 5: an 8 KiB block holds three Q4_0 tiles or two Q6_K
tiles, then still needs an attempt for the next prefix.

### Output layout

Each compute keeps a 12 KiB f32 accumulator because one output tile is reduced
across all `rows_steps`. After the block loop, `program_output` reads each
64-float accumulator tile and writes a 64-element bf16 tile in native chain
order:

```text
accumulator tile: [col_half][vec][8 cols] f32
output tile:      [col_half][vec][8 cols] bf16
```

`program_output` applies the optional tanh softcap (`cap * tanh(x / cap)`) with
the native bf16 `vtanh`, then stores bf16 with round-to-nearest-even. The drain
writes the chain layout `[N/8][outputs_vecs][8]`: each group's 3-D drain tap
places its 64-column slice into the shared N-major buffer, so the next call's
input fill reads the same buffer without a CPU transpose. The C++ API exposes
`[inputs_vecs][outputs_cols]`; `unpack_outputs` converts the chain blocks back.
The C++ API is bf16-only; a caller that needs f32 widens the result itself.

The softcap is a named reduced-precision mode: bf16 vtanh is approximate, so
its verification policy is `max(1e-3, 5e-2 * column_scale)`.

### GEGLU epilogue

```text
accumulator: [gate bands][up bands] f32, gate_bands = cols_bands / 2
gate band b pairs with up band b + gate_bands in the same compute (same 16 columns)
narrow both to bf16; out band b = gelu_bf16(gate) * up, only gate bands stored
```

Both operands narrow to bf16 in the epilogue, matching the two-launch path,
which also rounds to bf16 before the host activation. The native bf16 `vtanh`
is the only approximation. Because the drain writes `cols_bands` bands in
band-major order, the gate half is the packed next-matmul input at output offset
0; the down op retargets its input patches to the output argument and the gate
half is exactly its K. No pack stage is needed for this link.

AIE2P emulates f32 vector arithmetic with multiple bf16 products and
conversions, so the GELU polynomial is computed in bf16 and widened only for the
native tanh argument. The output is bf16 anyway and the measured error is
unchanged, while the loop body is 4x smaller. The GEGLU body is a loop over the
two 32-element halves with unrolling disabled; the unrolled form added program
memory for no speed.

GEGLU bands are exact only for a gate-then-up combined call; the host validates
two equal-width segments and no softcap.

## Golden and test layout

`build.py` emits `build/golden/index.txt`, a manifest of 30 cases. `./test`
loads the xclbin once and runs every listed case. Coverage:

- Q4_0 attention/FFN shapes in both orientations up to 12288 rows or 12288
  columns, including `256x512`, `256x1536`, and 2048/4096/6144/12288 rows.
- Two explicit 256-column padded cases (`q4_0_1536x256p512`,
  `q4_0_2048x256p512`).
- Q6_K `512x1536`, `1536x2048`, `1536x4096`, `1536x6144`, `1792x5632`,
  `1792x11264`.
- The full 262144-column Q6_K head as 11 band chunks, concatenated into one
  host tensor to check `weights_col_offset` / `outputs_col_offset`.
- Partition boundaries at 1/2/3/4/63/64/65/112/114/154/225 tiles and
  36/63/64/65/77/127/128 weight blocks.

Each case has canonical raw weights, bfloat16 inputs and a NumPy reference. The
C++ wrapper repacks with the real `src/quants` headers, builds the per-compute
block-major streams, packs input groups, runs, and unpacks into logical
`[inputs_vecs][weights_cols]` outputs.

### Verification tolerance

Outputs are checked per column against
`tolerance = max(1e-3, 6e-3 * column_scale)`, where `column_scale` is the
largest `|expected|` in that column over the four input vectors. An element
passes when its error is within its column's tolerance; `worst` is
`error / tolerance`.

The device stores bf16, so bf16 round-to-nearest dominates: a value can move by
one bf16 ulp relative to the f32 reference when accumulations differ across a
rounding boundary, up to ~2^-8 relative. The 6e-3 term leaves margin for that
plus fp32 accumulation error. The softcap uses `5e-2`; `q4_0_packed16` uses 5%.

Absolute error is not shape-independent. The kernel accumulates K terms in
fp32; mean absolute error grows as ~`1.2e-7 * K`. That is an unbiased RTNE
envelope, not a rounding-mode bug. Do not replace this with a fixed absolute
error: it cannot fit both K=1536 and K=12288, and it makes the effective
relative bound depend on the column magnitude.

`src/quants/tests/mmul.py` uses its own per-column bound. `tests.py` applies the
same 30-case policy as the C++ test.

The Python runner uses the fixed-max layout (64 input objects, 128 weight
blocks) for its own suite; the C++ path uses runtime partition counts.
`tests/poc/iter_args.py` is the focused device regression for the carried
memref and conditional acquire/release pattern.

## Build and cache notes

- `compile_external_kernels` skips an existing `.o` even when its source
  changed. `build.py` deletes the active build directory's `*.o` first.
- The inline-kernel JIT cache can also reuse stale objects. Clear stale
  `~/.npu/cache` entries after editing an inline kernel.
- `design.py`, `run.py`, `npu.py`, `sequence.py` and `worker.py` are in the
  JIT `source_files` list, so edits to fixed constants, the static device
  configuration, the host sequence or the worker body invalidate the cache
  automatically.
- `build.py` emits `design_meta.h` from `design.py`; the host uses it for buffer
  sizes, tile geometry, output layout and config indices, so host and Python
  design cannot drift.
- `build.py` prepends the Peano/AIE bin directories to `PATH` because `aiecc`
  needs `opt` and the other toolchain binaries.
- The shim BD repeat/iteration dimension is capped at 64 in our fills (the PoC
  documents a separate 1023-element contiguous access limit). Larger counts
  factor into two dimensions when possible and fall back to <=64 chunks for
  primes. Contiguous output drains use the wide contiguous length and are
  exempt.
- The C++ dynamic path uses straight-line runtime-sized fills with static BD
  assignment (`--aie-assign-runtime-sequence-bd-ids`). Runtime loops plus
  `--aie-lower-dynamic-bd-pool` deadlocked against the FIFOs once live counts
  exceeded the FIFO depth; do not reintroduce them.

### Python API gotchas

- `else_` is a context manager that takes the `if_` handle:
  `with if_(cond) as branch:` then `with else_(branch):`.
- A result-producing `scf.if` is needed when a memref must be selected; plain
  `if_`/`else_` cannot yield memory objects.
- `@iron.jit` rejects unannotated scalar parameters (no runtime-scalar
  plumbing), even though `Runtime`/`fill` support runtime values.
- Runtime fills and drains take a `TensorAccessPattern` (`tap=...`), not
  separate `offset`/`sizes`/`strides` arguments; `tensor_dims` is the runtime
  buffer's shape. `sequence.py` is the only place that builds them.
- `range_` with `iter_args` supplies its own yield, so do not pass
  `insert_yield`.
- bfloat16 inputs must be reinterpreted, not cast: `.view(np.uint16)`. A
  `dtype=np.uint16` cast converts numerically and produces garbage bits.
- `pack.py` owns case construction: config object, input groups, block-major
  weights and reference data. `sequence.py` pads into the fixed host buffers
  and `tests.py` verifies.

### C++ host notes

- XRT `xrt::bo::write` takes `(src, size, offset)`.
- Host weight padding is per group, not per block: each group stores
  `weights_blocks * COMPUTES_PER_GROUP * L1` bytes contiguously, then zeroes up
  to `MAX_WEIGHT_BLOCKS`.
- `Program::prepare` takes canonical quant bytes, a raw column stride, and a
  512-column weight offset. `Program::run` takes bfloat16 bits in
  `[inputs_vecs][weights_rows]` order and returns bfloat16 bits in
  `[inputs_vecs][outputs_cols]` order.
- Prepared handles own their host BO; there is no eviction. Destroying a handle
  releases its BO.
- `xrt::run` is cached in `Program::Impl` and reused across calls.

## Kernel loop idioms

A standalone 64-byte copy experiment found the source shape matters more than
backend flags: pointer walking lets Peano emit post-increment addressing and
discover a single-register load/store rotation; indexed loops generate explicit
index updates and are several times slower. `unroll(enable)` and
`unroll_count(N)` recover part of the gap; `unroll(full)` is fastest but grows
`.text` substantially. The common `aie-*` flags had no effect. Tables are in
`docs/measurements.md`; the full playbook is in
[`../quants/DESIGN.md`](../quants/DESIGN.md).

Counterexample: the q4_0 dequant+mmul loop is recurrence-bound, not
address-bound. Advancing the input or record pointers regressed 2-4%; only the
payload pointer was neutral. Do not rewrite `unpack_mmul`'s record/input loops
to pointer form without measuring. In `src/program`, pointer increments were
applied only to address-bound copy paths (`program_zero`, carry copy), where the
disassembly confirms post-increment addressing.

`-O3` produced byte-identical kernels; `-Os` grew `.text` and was slower. Keep
`-O2`.

## Build artefacts and program memory

`build.py --production` builds into `build-production/` from
`production_dynamic_program`. It links only `program_begin`, `program_zero`,
`program_output`, `program_block_begin`, `program_q4_block` and
`program_q6_block`. It rejects the diagnostic mode range and skips the cycle
timing row. The normal build keeps the diagnostic split path.

Measure program memory from the linked ELF:
`llvm-size -A <build-dir>/xclbin_build/elfs_main_core_*/*.elf`. Diagnostic
builds are larger because they also link `program_assemble`, `program_probe`,
`program_raw_check`, the split-path worker and diagnostic-only branches.

Unused device functions and kernels should be removed from `npu.py`: they slow
JIT/build, appear in the source-file cache set, and hide which kernels the
worker actually calls. The block kernels own the production path;
`assemble`/`probe` remain for diagnostics.
