# `src/program`

One fixed NPU design that runs Q4_0 and Q6_K matmuls with runtime-varying
dimensions.  The design is built from Python and called through a small C++
wrapper.  The C++ path generates runtime-sized fills and transfers only the
live inputs and weight blocks; the Python runner keeps the simple fixed-max
fill pattern for its test suite.  The first target is the Gemma 4 E2B matmul
set plus its Q6_K output head; row lookups and the F16 per-layer CPU
projection are out of scope.  The design targets productive, quiet inference:
the NPU is used for efficiency, not to beat the CPU or iGPU on throughput.

```
design.py           fixed geometry, buffer types, config layout
pack.py             case construction, host packing, NumPy references
run.py              JIT program and Python test runner
build.py            xclbin + transaction + design_meta.h + golden data
kernels/            AIE tile assembly, probe diagnostics, quant compute
host/program.h/.cc  loaded design and host repack/pack/unpack API
host/test.cc        manifest-driven golden and timing driver
```

## Build and test

```bash
~/Projects/mlir-aie/ironenv/bin/python src/program/build.py
cd src/program/build && ./test
```

`build.py` emits `aie.xclbin`, `generate_txn.h`, `design_meta.h`, the golden
cases, and the C++ test.  `./test` loads the xclbin once and runs every case
listed in `build/golden/index.txt`.  The manifest has 30 cases: Q4_0
attention/FFN shapes in both orientations up to 12288 rows or 12288 columns,
two padded 256-column cases, a Q4_0 softcap case, Q6_K variants up to
`1792x11264`, and the full 262144-column Q6_K head as 11 chunks.  The golden set includes the raw head
(about 315 MB), so golden generation takes a few minutes.

Outputs are compared per column against `max(1e-3, 2e-5 * column_scale)`,
because the fp32 accumulation error grows with K; the rationale and the
measured error model are in `NOTES.md` under "Golden verification tolerance".

Kernel edits need a clean object build: `build.py` deletes the active build
directory's `*.o` because `compile_external_kernels` otherwise reuses stale
objects.  Inline kernels can also leave stale `~/.npu/cache` entries;
`tests/poc/README.md` shows how to clear them.

### Production build

```bash
~/Projects/mlir-aie/ironenv/bin/python src/program/build.py --production
cd src/program/build-production && ./test
```

The production variant uses `run.production_dynamic_program` and omits the
diagnostic path: `program_assemble`, `program_probe`, `program_raw_check`, the
`ready` dispatch, and the diagnostic modes.  Only Q4_0 (ggml_type 2) and Q6_K
(14) are accepted; the host wrapper rejects the diagnostic range
0xF0-0xFF.  It builds into
`build-production/`, reuses `build/golden` through a symlink when available,
and `./test --timing` skips the diagnostic timing rows.

Linked program memory is 11872 bytes per compute core (73% of 16 KiB),
versus 14880 bytes for the diagnostic build.

### Python checks

```bash
~/Projects/mlir-aie/ironenv/bin/python src/program/run.py
~/Projects/mlir-aie/ironenv/bin/python src/program/run.py --head
```

### C++ timings

```bash
cd src/program/build && ./test --timing
cd src/program/build && ./test --head-timing
cd src/program/build && ./test --combined-timing
cd src/program/build && ./test --all-timing
```

`--head-timing` measures the prepared-weight, steady-state Q6_K head over all
band chunks.  `--combined-timing` measures gate+up and Q+K+V as single wide
matmuls and verifies each segment.  `--cold-timing` cycles through several
resident BOs so the weights are not cache-warm and reports min/p50/p95/max
(`SPIRAL_COLD_BOS`, `SPIRAL_COLD_ITERS`).  `SPIRAL_COLD_GAP_US` inserts an
idle gap between calls to measure the power-state penalty.  `--pipeline-timing`
runs the head through the `submit`/`finish` queue; `SPIRAL_PIPELINE_SLOTS`
sets the in-flight slot count (default 2).  `SPIRAL_LOOKAHEAD=1` enables the
experimental cross-node look-ahead (off by default; neutral on Gemma).
`SPIRAL_DMA_LATENCY=1` opts into holding `/dev/cpu_dma_latency` at 0 (default
off; it helps prompt processing but costs decode), and
`SPIRAL_DMA_LATENCY_PATH` overrides the node path.  `--all-timing` measures every
golden case with prepared weights.

The diagnostic build's timing table also includes `cycle` (fills + FIFO
acquire/release only) and `raw` (per-block transport checksum, no assembly)
rows.  The production build omits them.

## C++ API

```cpp
#include "program.h"

spiral::Program program;

spiral::MatmulShape shape;
shape.format = 2;          // ggml_type: Q4_0 = 2, Q6_K = 14
shape.weights_rows = 1536; // multiple of 256
shape.weights_cols = 2048; // multiple of 512; max 48 bands per call
shape.inputs_vecs = 4;     // must match the fixed design
shape.soft_cap = 0.0f;    // optional tanh softcap; 0 disables it

std::vector<uint16_t> outputs(4 * shape.weights_cols);
program.run(shape,
            raw_weights, raw_stride, 0,  // canonical quant bytes
            inputs,                      // bfloat16 bits, [4][weights_rows]
            outputs.data(), shape.weights_cols, 0);
```

`raw_weights` uses the canonical quant block layout with `raw_stride` bytes
per weights column.  `outputs` is `uint16_t` bfloat16 bits in
`[inputs_vecs][outputs_cols]` order, the same layout as the next call's
input:

```cpp
std::vector<uint16_t> outputs(4 * shape.weights_cols);
program.run(shape, prepared, inputs, outputs.data(),
            shape.weights_cols, 0);
// outputs is valid bfloat16 bits in [4][shape.weights_cols] order.
```

The epilogue transposes the native mmul tile and writes bf16 with
round-to-nearest-even.  With `soft_cap != 0` it applies
`cap * tanh(x / cap)` using the native bf16 vtanh; the host passes `cap` and
`1/cap` as bf16 hi/lo pairs so the device has no division.  That is an
approximate, named reduced-precision mode.

Repack once and reuse the weights across calls:

```cpp
auto prepared = program.prepare(shape, raw_weights, raw_stride, 0);
for (int step = 0; step < steps; ++step) {
  program.run(shape, prepared, inputs, outputs.data(), shape.weights_cols, 0);
}
```

Each prepared handle owns its own resident host weight BO, so repeated runs
skip the weight upload and multiple handles can remain resident.  The weight,
input, and output BOs use `host_only`; the NPU reads them from host memory over
the interconnect.

For a tensor chunk, pass the first 512-column band as `weights_col_offset`
and the corresponding destination column as `outputs_col_offset`:

```cpp
program.run(chunk_shape, head_raw, head_stride, band_offset,
            inputs, head_outputs, head_cols, band_offset);
```

Independent column blocks over the same input can be prepared as one wider
matmul:

```cpp
spiral::WeightSegment segments[2] = {
    {gate_raw, gate_stride, 0, gate_cols},
    {up_raw,   up_stride,   0, up_cols},
};
shape.weights_cols = gate_cols + up_cols;
auto prepared = program.prepare(shape, segments, 2);
```

`prepare` repacks each segment and concatenates them along `weights_cols`, so
one launch computes both outputs.  Output columns appear in segment order.
This is intended for FFN gate+up and for Q+K+V in the KV layers; the caller
splits the output columns.  Segments must share `format` and `weights_rows`.

## Performance snapshot

Production medians (device kernel time).  This is a single-session
snapshot; `NOTES.md` records its provenance, and it predates the current
measurement baseline, so use it for kernel shapes, not as a live comparison:

| case | kernel µs |
| :-- | --: |
| Q4_0 2048x2048 | ~123 |
| Q6_K 2048x2048 | ~144 |
| Q4_0 1536x12288 | ~327 |
| Q6_K 1536x6144 | ~234 |
| Q6_K head, 11 chunks | ~8200 total |

The per-call launch/fixed floor is ~80 us including fill (~60-70 us of host
launch).  In the combined-column
benchmark, gate+up plus Q+K+V saves about 5.4 ms/token, taking the
launch-limited E2B estimate to ~19.8 tok/s (about 21 tok/s kernel-only).  The
memory-bound estimate is ~40 tok/s; the remaining gap is call/launch overhead.
See `NOTES.md` for the full baseline, head chunk sweep, PoC comparison, and
launch-limited model breakdown.

## Device contract

- 8 groups x 4 compute tiles, 4 input vectors, 256-row tiles,
  16-column tiles, 8 KiB L1 stream blocks, 32 KiB L2 stream blocks.
- Q4_0 and Q6_K use one block kernel per L1 block: assembly/carry and mmul run
  in the same call.  Probe diagnostics keep the split tile path.
- Tiles wholly contained in one L1 block are read in place; only tiles that
  straddle a block boundary are copied through the carry buffer.
- Weight streams are continuous and padded only at the end.  One input object
  holds four row slices.  The worker carries the current weight block through
  a single `range_` loop, advances the input group with a result-producing
  `scf.if`, and the block kernels select the local row.
- One call handles at most 128 weight blocks, 12 input groups (48 rows_steps),
  and 48 bands (24576 columns).
- Shapes are rounded up by the caller: `weights_rows` to 256 and
  `weights_cols` to 512.
- The C++ path partitions live inputs into four fills and each group's weight
  blocks into two fills, each with a 1..64 object repeat count.  Any padding
  needed to keep every part non-empty is at most three input objects and one
  weight block per group; the worker drains those via config words 6 and 7.
- The Python runner still fills the fixed maximum (64 input objects: one
  config plus 63 groups, and 128 blocks) for its no-op, Q4_0, and Q6_K
  checks.
- Notes for the stream/carry design, measurements, and failed experiments
  live in `NOTES.md`; `TODO.md` tracks past work, active tasks, and planned
  work.
