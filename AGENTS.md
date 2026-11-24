# Agent guide

This repository builds and benchmarks AMD XDNA2 / AIE2P NPU kernels for
quantized model inference. Correctness and measured cycles matter more than
source-level cleverness.

**Framing.** The NPU path exists for productive inference without the machine's
fans ramping up. It is chosen for efficiency, not peak throughput. Do not reject
a design only because it is slower than the CPU or iGPU, and never report a
performance result without its temperature and fan context. Power is not
measured; `Tctl` and fan RPM are the proxy.

## Invariants

1. **Exactness.** `q4_0`, `q6_k`, and `q8_0` are exact over the canonical
   format's finite range and keep the canonical tile size. A format that grows
   the tile or relaxes precision takes a named variant with its own
   verification policy (`q4_0_hilo`, `q4_0_packed16`, `q6_k_int8`, softcap,
   bfp16). Never widen a tolerance to make a test pass.
2. **Transport contract.** Repack and transport operate on 256-row x 16-column
   tiles, packed to a 64-byte multiple, row-block-major then column-pair-major.
   Preserve it unless the change is explicitly a new format.
3. **Header boundary.** `layout.h` and `repack.h` must compile for the host
   loader without AIE headers. Device-only code belongs in `.aie.h`.
4. **Program memory.** A compute tile has 16 KiB. Production is ~12.5 KiB
   (77%). Check `.text` and stack frame for every kernel change.
5. **Measurement provenance.** A number without a build, a shape and a date is
   not evidence. Tables live in `docs/measurements.md`.

## Traps

Read these before optimising; each has cost someone a day.

- **The q4_0 mmul is recurrence-bound.** Pointer-increment rewrites regress
  2-4%. The copy-loop rule does not transfer. Measure both forms.
  See `docs/rejected.md`.
- **`aie-*` backend flags are not the lever.** The unroll, pipeliner and
  register flags were all neutral on copy kernels. Fix source shape first.
- **Nested carried-memref loops deadlock on device.** Keep the worker as one
  loop carrying all state, or validate a new structure in
  `tests/poc/iter_args.py` first.
- **Stale objects.** `build.py` deletes the active build directory's `*.o`, but
  inline kernels can still be cached in `~/.npu/cache`. Clear the relevant
  entries after editing an inline kernel.
- **Compare within one boot.** Clocks are not locked. Cross-boot absolute
  cycles usually match, but use per-value ratios from one session.
- **Thermal runs must be long.** Short generations look fanless on the iGPU and
  CPU. Soak until `Tctl` and fan RPM plateau; `tests/bench/soak/` does this and
  exits 2 when the measurement configuration drifts.
- **`bytes / time` is not a bandwidth.** A call costs ~80 us plus
  bytes/48 GB/s; the fixed floor dominates small calls.
- **Do not allocate the product of independent maxima.** Buffer sizes come from
  the target tensor set.

## Workflow for a kernel change

1. Measure the current kernel and read the relevant section of
   `src/quants/DESIGN.md`. Check `docs/rejected.md` first.
2. Try the pointer form only on address-bound loops. Sweep
   `unroll(disable|enable|count(N)|full)` on the actual loop.
3. Batch to the native 512-bit width (32 bf16 lanes) where the format is
   narrower. Prefer host precomputation when the repack contract allows it.
4. Inspect the disassembly before hand-writing assembly; the compiler often
   wins when the source shape is right.
5. Report cycles, `weights/cycle`, bytes/weight, `.text`, and stack frame.
   Update `docs/measurements.md`, not ten other files.
6. If it lost, add one line to `docs/rejected.md` with the reason.

## Navigation

| task | read |
| :-- | :-- |
| any session | this file, then `docs/index.md` if you need orientation |
| quant format, kernel loop, tile/API contract | `src/quants/DESIGN.md` |
| build/run/call the fixed program | `src/program/README.md` |
| worker, data flow, config, epilogues | `src/program/DESIGN.md` |
| attach/build the ggml backend | `integrations/llama.cpp/README.md` |
| matchers, fallback, preparation | `integrations/llama.cpp/DESIGN.md` |
| any performance figure | `docs/measurements.md` |
| before retrying an idea | `docs/rejected.md` |
| mlir-aie/llvm-aie sources, env setup | `docs/upstream.md` |

## Environment

```sh
source tools/env.sh     # resolves MLIR-AIE + Peano, sets SPIRAL_PY
python src/program/build.py
```

No document may hardcode an absolute `~/Projects/...` path. If a command needs
the toolchain, it goes through `tools/env.sh` or `$SPIRAL_PY`.

## Documentation rules

- One job per file: orientation, reference, tasks, or record.
- Reference files are edited in place and contain no history. `git log` is the
  history; do not add changelogs or "Done" lists.
- Link a number, never restate it.
- Keep the role line at the top of each document current.
