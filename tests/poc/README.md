> Role: record (frozen). Written 2026-11; numbers are historical evidence and
> are not updated. Current numbers: [`../../docs/measurements.md`](../../docs/measurements.md).

# Proof of Concept

These tests use the AMD MLIR-AIE Python library to evaluate basic functionality
and performance of the AMD XDNA 2 NPU. Toolchain setup and pinned versions are
in [`../../docs/upstream.md`](../../docs/upstream.md). Run them from the
repository root after `source tools/env.sh`.

To run examples with inline kernels, you may need to add the Peano LLVM compiler
that MLIR-AIE installed to your PATH after following the MLIR-AIE instructions
to enter their `ironenv` virtual environment. In my case, the default location
was `${VIRTUAL_ENV}/../my_install/mlir/bin`.

## Bulk Memory Bandwidth

`memory.py` reports the bulk transfer speed from host memory to the NPU in GB/s,
testing a varying number of DMA channels.

```
# Ryzen AI 9 HX 475 Laptop

Tested  1 channels: avg 14.2 GB/s   (min 14.0 to max 14.4)
Tested  2 channels: avg 26.9 GB/s   (min 26.6 to max 27.2)
Tested  3 channels: avg 35.2 GB/s   (min 34.7 to max 35.5)
Tested  4 channels: avg 41.0 GB/s   (min 39.9 to max 42.3)
Tested  6 channels: avg 49.4 GB/s   (min 48.8 to max 49.9)
Tested  8 channels: avg 55.5 GB/s   (min 54.7 to max 56.2)
Tested 10 channels: avg 56.6 GB/s   (min 56.2 to max 56.9)
Tested 12 channels: avg 57.0 GB/s   (min 56.6 to max 57.4)
Tested 14 channels: avg 57.2 GB/s   (min 56.8 to max 57.5)
Tested 16 channels: avg 57.4 GB/s   (min 57.0 to max 57.7)
```

## Matrix Memory Bandwidth

`flow.py` reports the transfer speed through the NPU following a path similar
to a matrix multiply, but without performing the actual calculations. This
result is unoptimized and the actual output data has not been verified.

```
# Ryzen AI 9 HX 475 Laptop

Transfer: avg 56.5 GB/s   (min 56.3 to max 56.6)
```

## Matrix Multiply

`calc.py` measures a 2048x2048 matrix multiply in types easily handled by
the NPU: float32 inputs and outputs with bfloat16 weights. It checks the
output against numpy, and reports throughput based on elapsed times.

```
# Ryzen AI 9 HX 475 Laptop

reps: 64
npu: 10022.6 us (min 9612.4, max 10321.8), e2e 10807.5 us
bandwidth: avg 53.98 GB/s, best 56.29 GB/s
compute: 214.26 GFLOP/s
verified against numpy
```

## Quantized Byte Stream

`stream/stream.py` streams weight tiles with mode-specific packed sizes
through fixed DMA buffers, showing how different quantization formats
can share one static NPU network without padding every tile to a
common maximum.

```
# Ryzen AI 9 HX 475 Laptop, --reps 64 (best samples; these run noisy)

q4_0:  52.8 GB/s
q6_k:  54.6 GB/s   (about 96% of the 57 GB/s bulk ceiling)
iq1_s: 33.5 GB/s
```

See [`stream/README.md`](stream/README.md) for the stream layout, kernel
structure, buffer geometry, cache behavior, and tuning results.

## Iter args and held ObjectFifos

`iter_args.py` is a device-level regression for carrying an acquired
ObjectFifo buffer through `scf.for` iter args and conditionally replacing it
with a result-producing `scf.if`:

```python
block = cons.acquire(1)
for i, current, _result in range_(
    NUM_OBJECTS, iter_args=[block], insert_yield=False
):
    ...
    branch = IfOp(take_next, [current.type], has_else=True)
    with InsertionPoint(branch.then_block):
        cons.release(1)
        next_block = cons.acquire(1)
        scf_yield([next_block])
    with InsertionPoint(branch.else_block):
        scf_yield([current])
    iron_yield([branch.results[0]])
cons.release(1)
```

The worker sums the first word of four input objects (1+2+3+4) and writes the
sum plus the last value and a call count to an output object:

```bash
python tests/poc/iter_args.py
```

Expected:

```text
iter_args carried block: sum 10, last 4, count 4 (expected 10, 4, 4)
ok
```

This is the pattern needed to hold a weight block across row boundaries
without row-aligning the weight stream.  If the inline kernels are edited,
clear the stale kernel objects in the JIT cache (for example, remove the
`~/.npu/cache/*` directories containing `iter_args_*.cc`) because
`compile_external_kernels` skips existing object files.

## Program-Memory Overlays

Three scripts show that a loaded xclbin can replace compute-tile program memory
without a PDI reload, and how far that can be taken while the tiles keep
running.

- `overlay.py` - the cost baseline. It swaps two real kernels by writing their
  program memory from the command processor, and compares that with loading a
  different xclbin. It resets the cores around the write, so it is the
  reference, not the running case.
- `overlay_blocks.py` - the link side. It checks that the format blocks can move
  into a reloadable tail while the resident code keeps the same addresses and
  entry point, so a tail can be swapped without relinking or reloading the
  xclbin.
- `overlay_packets.py` - the runtime experiment. It streams control packets
  over the NoC into program memory while the core runs, with three modes:
  - `slot` writes a second kernel into a spare region of one tile's program
    memory and reads it back to verify.
  - `broadcast` sends one write stream to every compute tile at once and has
    the command processor verify it in place, so no read-back traffic is
    needed.
  - `switch` keeps the tile running on one stage while the next stage is
    streamed into the spare region, then transfers execution to the new stage
    and proves it ran. A/B rounds can repeat, either one dispatch per round or
    all unrolled into a single dispatch.

Commands, measurements, the packet and link-layout findings, limits, and the
next steps (real quant block images at fixed slots, whole-array format
switching) are in [`overlay.md`](overlay.md).

## Call Overheads

`invoke.py` reports the time required to call the NPU using different patterns,
from both Python and C++.

The EmitC runtime variants keep one compiled design and pass the step counts per
call:

```bash
python tests/poc/invoke.py --emitc       # runtime sizes, static body
python tests/poc/invoke.py --emitc-loop  # runtime sizes plus a host loop
python tests/poc/invoke.py --emitc-if    # runtime sizes plus a runtime if
```

What is dynamic there: tap offsets, sizes and strides, and runtime-bound host
loop/`if` control flow, all carried as `DispatchTime` scalars generated by the
EmitC dispatch bridge. What stays static: tensor shapes and dtypes in the
`Runtime` signature, Worker arguments, and anything used as a shape or truth
test at generation time. Runtime loops are the dynamic-BD-pool path; each
iteration needs an explicit `TaskGroup` and awaited transfers, otherwise BD ids
can be recycled while a DMA is in flight. The loop and `if` variants are probes,
not the fast path.

The C++ program generates NPU instructions at runtime, using code generated by
a semi-supported path in MLIR-AIE. This makes it practical to change input and
output sizes by avoiding the ~2.5ms needed to reconfigure the NPU.

The ~70us fixed per-call overhead shown here is a bit slow, but fast enough to
support an end-to-end demo. Newer Linux kernels include some changes that may
improve matters, and AMD has a wide variety of experimental projects.

```
# Ryzen AI 9 HX 475 Laptop

Python API, 128 calls (warmup 8):
  workload              avg us       (min-max)       e2e avg us
  repeat A               105.5    (  93.1- 208.7)         281.7
  repeat B               108.6    (  85.7- 137.4)         271.1
  reconfigure A/B       2748.5    (2366.9-3973.4)        3309.8
  reconfiguration per switch: npu 2641.4 us e2e 3033.4 us

C++ API, 128 calls (warmup 8)
(33/33)
  mode                  avg us       (min-max)
  repeat A                91.2    (  72.6- 109.1)
  repeat B                80.6    (  69.8- 138.4)
  alternate A/B           92.6    (  80.0- 102.1)
  reuse A/B               89.4    (  75.5- 103.0)
  pipeline A/B            31.8  (total 4073.6 us over 128 calls)

2x size:
  repeat A               130.4    ( 112.5- 241.6)
  repeat B               124.2    ( 100.7- 138.1)
  alternate A/B          124.3    (  91.5- 198.5)
  reuse A/B              122.1    ( 102.2- 142.4)
  pipeline A/B            56.0  (total 7170.6 us over 128 calls)

4x size:
  repeat A               221.3    ( 197.5- 270.5)
  repeat B               226.8    ( 200.0- 239.0)
  alternate A/B          226.8    ( 208.2- 244.4)
  reuse A/B              228.0    ( 206.7- 248.8)
  pipeline A/B           158.5  (total 20286.3 us over 128 calls)
```

## TXN Concatenation and Chained Matmuls

`txn_concat/` builds two fixed-program matmuls into one NPU command and chains
the first output directly into the second input, with no host conversion of the
intermediate activation.  The prototype also repeats the pair eight times to
show the per-dispatch floor being amortised.

```
# Build the production design first
python src/program/build.py --production
python tests/poc/txn_concat/run.py

q4_0_1536x2048 -> q4_0_2048x1536
  chain vs independent: bit_mismatches=0 max_abs=0.00000000
  2 ops: chain 156.5 us vs independent 217.1 us (1.39x)
  8 ops: chain 445.9 us (55.7 us/op) vs 866.8 us (108.4 us/op) (1.94x)

q6_k_2048x2048 -> q6_k_2048x2048
  chain vs independent: bit_mismatches=0 max_abs=0.00000000
  2 ops: chain 222.0 us vs independent 290.3 us (1.31x)
  8 ops: chain 695.7 us (87.0 us/op) vs 1095.1 us (136.9 us/op) (1.57x)
```

The durable design record, the current TXN argument mapping, the patch
recipe, and the measurement history are in
[`txn_concat/README.md`](txn_concat/README.md).
