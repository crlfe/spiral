> Role: orientation. Current as of 2026-11-24. History: `git log -- src/program/README.md`.

# `src/program`

One fixed NPU design that runs Q4_0 and Q6_K matmuls with runtime-varying
dimensions. The design is built from Python and called through a small C++
wrapper. The C++ path generates runtime-sized fills and transfers only the live
inputs and weight blocks; the Python runner keeps the simple fixed-max fill
pattern for its test suite.

The first target is the Gemma 4 E2B matmul set plus its Q6_K output head; row
lookups and the F16 per-layer CPU projection are out of scope. The design
targets productive, quiet inference: the NPU is used for efficiency, not to
beat the CPU or iGPU on throughput.

```text
design.py           fixed geometry, buffer types, config layout
pack.py             case construction, host packing, NumPy references
run.py              JIT program and Python test runner
build.py            xclbin + transaction + design_meta.h + golden data
kernels/            AIE tile assembly, probe diagnostics, quant compute
host/program.h/.cc  loaded design and host repack/pack/unpack API
host/test.cc        manifest-driven golden and timing driver
```

Design, data flow, config words, epilogues and the golden tolerance are in
[`DESIGN.md`](DESIGN.md). Numbers are in
[`../../docs/measurements.md`](../../docs/measurements.md); open work is in
[`TODO.md`](TODO.md).

## Build and test

```sh
source tools/env.sh
python src/program/build.py
cd src/program/build && ./test
```

`build.py` emits `aie.xclbin`, `generate_txn.h`, `design_meta.h`, the golden
cases and the C++ test. `./test` loads the xclbin once and runs every case in
`build/golden/index.txt`. The manifest has 30 cases: Q4_0 attention/FFN shapes
in both orientations up to 12288 rows or columns, two padded 256-column cases,
a Q4_0 softcap case, Q6_K variants up to `1792x11264`, and the full
262144-column Q6_K head as 11 chunks. The golden set includes the raw head
(~315 MB), so generation takes a few minutes.

Kernel edits need a clean object build: `build.py` deletes the active build
directory's `*.o` because `compile_external_kernels` otherwise reuses stale
objects. Inline kernels can also leave stale `~/.npu/cache` entries; see
`tests/poc/README.md` for how to clear them.

### Production build

```sh
source tools/env.sh
python src/program/build.py --production
cd src/program/build-production && ./test
```

The production variant uses `run.production_dynamic_program` and omits the
diagnostic path. Only Q4_0 (ggml_type 2) and Q6_K (14) are accepted; the host
wrapper rejects the diagnostic range 0xF0-0xFF. It builds into
`build-production/`, links to `build/golden` when available, and
`./test --timing` skips the diagnostic rows.

### Python checks

```sh
python src/program/run.py
python src/program/run.py --head
```

### C++ timings

```sh
cd src/program/build && ./test --timing
cd src/program/build && ./test --head-timing
cd src/program/build && ./test --combined-timing
cd src/program/build && ./test --all-timing
```

`--head-timing` measures the prepared-weight, steady-state Q6_K head over all
band chunks. `--combined-timing` measures gate+up and Q+K+V as single wide
matmuls and verifies each segment. `--cold-timing` cycles through several
resident BOs so weights are not cache-warm and reports min/p50/p95/max
(`SPIRAL_COLD_BOS`, `SPIRAL_COLD_ITERS`). `SPIRAL_COLD_GAP_US` inserts an idle
gap between calls to measure the power-state penalty. `--pipeline-timing` runs
the head through the `submit`/`finish` queue; `SPIRAL_PIPELINE_SLOTS` sets the
in-flight count (default 2). `SPIRAL_LOOKAHEAD=1` enables the experimental
cross-node look-ahead (off by default; neutral on Gemma). `SPIRAL_DMA_LATENCY=1`
opts into holding `/dev/cpu_dma_latency` at 0 (off by default; helps prompt
processing, costs decode); `SPIRAL_DMA_LATENCY_PATH` overrides the node.
`--all-timing` measures every golden case with prepared weights.

The diagnostic build's timing table also includes `cycle` (fills + FIFO
acquire/release only) and `raw` (per-block transport checksum, no assembly)
rows. The production build omits them.

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
shape.epilogue = spiral::kEpilogueNone;  // or kEpilogueSoftcap / kEpilogueGeglu

std::vector<uint16_t> outputs(4 * shape.weights_cols);
program.run(shape,
            raw_weights, raw_stride, 0,  // canonical quant bytes
            inputs,                      // bfloat16 bits, [4][weights_rows]
            outputs.data(), shape.weights_cols, 0);
```

`raw_weights` uses the canonical quant block layout with `raw_stride` bytes per
weights column. `outputs` is `uint16_t` bfloat16 bits in
`[inputs_vecs][outputs_cols]` order, the same layout as the next call's input:

```cpp
std::vector<uint16_t> outputs(4 * shape.weights_cols);
program.run(shape, prepared, inputs, outputs.data(),
            shape.weights_cols, 0);
// outputs is valid bfloat16 bits in [4][shape.weights_cols] order.
```

The epilogue transposes the native mmul tile and writes bf16 with
round-to-nearest-even. `shape.epilogue` selects the kind: `kEpilogueNone`
(plain bf16), `kEpilogueSoftcap` (`cap * tanh(x / cap)` using `shape.soft_cap`
and the native bf16 vtanh), or `kEpilogueGeglu`. The softcap host passes `cap`
and `1/cap` as bf16 hi/lo pairs so the device has no division; it is an
approximate, named reduced-precision mode. The kinds are exclusive:
`kEpilogueSoftcap` requires a nonzero `soft_cap`, and the others require zero.

`shape.epilogue = spiral::kEpilogueGeglu` selects the gate+up activation for a
two-segment `prepare`: the first half of `weights_cols` is the gate and the
second half is up. The epilogue narrows both to bf16 and computes
`gelu(gate) * up` with the native bf16 vtanh, then stores the gate bands. The
output is `[4][weights_cols / 2]`; the call requires exactly two equal-width
segments, an even number of bands, and `soft_cap == 0`. In a chained TXN the
gate half is the down matmul's packed input at output offset 0, so no host pack
is needed. See `DESIGN.md` for the measured error and tolerance.

Repack once and reuse the weights across calls:

```cpp
auto prepared = program.prepare(shape, raw_weights, raw_stride, 0);
for (int step = 0; step < steps; ++step) {
  program.run(shape, prepared, inputs, outputs.data(), shape.weights_cols, 0);
}
```

Each prepared handle owns its own resident host weight BO, so repeated runs skip
the weight upload and multiple handles can remain resident. The weight, input
and output BOs use `host_only`; the NPU reads them from host memory over the
interconnect.

For a tensor chunk, pass the first 512-column band as `weights_col_offset` and
the corresponding destination column as `outputs_col_offset`:

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

`prepare` repacks each segment and concatenates along `weights_cols`, so one
launch computes both outputs. Output columns appear in segment order. This is
intended for FFN gate+up and for Q+K+V in the KV layers; the caller splits the
output columns. Segments must share `format` and `weights_rows`.

Dependent operations can be prepared as one chained transaction:

```cpp
spiral::ChainOp ops[2];
ops[0].shape = gate_shape;                 // weights_cols = gate + up
ops[0].shape.epilogue = spiral::kEpilogueGeglu;
ops[0].weights = &gate_up;                 // two-segment prepare
ops[1].shape = down_shape;                 // weights_rows = gate width
ops[1].shape.epilogue = spiral::kEpilogueNone;
ops[1].weights = &down;
auto chain = program.prepare_chain(ops, 2);
program.run_chain(chain, inputs, outputs.data(), down_shape.weights_cols, 0);
```

The first operation reads the host input; each later operation reads the
previous output in place, and only the final output is unpacked.
`prepare_chain` copies every weight stream into one BO, so the per-operation
handles only need to be alive during preparation. Up to four operations; the
chain runs synchronously and does not use the `submit`/`finish` pipeline.

## Device contract

- 8 groups x 4 compute tiles, 4 input vectors, 256-row tiles, 16-column tiles,
  8 KiB L1 stream blocks, 32 KiB L2 stream blocks.
- Q4_0 and Q6_K use one block kernel per L1 block: assembly/carry and mmul run in
  the same call. Probe diagnostics keep the split tile path.
- Tiles wholly contained in one L1 block are read in place; only tiles that
  straddle a block boundary are copied through the carry buffer.
- Weight streams are continuous and padded only at the end. One input object
  holds four row slices. The worker carries the current weight block through a
  single `range_` loop, advances the input group with a result-producing
  `scf.if`, and the block kernels select the local row.
- One call handles at most 128 weight blocks, 12 input groups (48 rows_steps),
  and 48 bands (24576 columns).
- Shapes are rounded up by the caller: `weights_rows` to 256 and `weights_cols`
  to 512.
- The C++ path partitions live inputs into four fills and each group's weight
  blocks into two fills, each with a 1..64 object repeat count. Padding needed
  to keep every part non-empty is at most three input objects and one weight
  block per group; the worker drains those via config words 6 and 7.
- The Python runner still fills the fixed maximum (64 input objects, 128 blocks)
  for its no-op, Q4_0 and Q6_K checks.
