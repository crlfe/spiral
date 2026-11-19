# Agent guide

This repository builds and benchmarks AMD XDNA2 / AIE2P NPU kernels for
quantized model inference. Correctness and measured cycles matter more than
source-level cleverness. The point of the NPU path is productive inference
without the machine's fans ramping up: it is chosen for efficiency, not peak
throughput. Do not reject a design only because it is slower than the CPU or
iGPU, and report temperature and fan context.  The measurement harness does
not measure power; Tctl and fan RPM are the proxy.

## Documentation map

Read these before changing a kernel or its loop structure:

- `src/quants/NOTES.md` — canonical AIE2 technique and benchmark notes.
  Read **“Loop optimization and Peano idioms”** first when touching kernel
  loops; it covers pointer increments, unroll pragmas, ZOL constraints, the
  VLIW slot/latency model, and the copy benchmark.
- `src/program/NOTES.md` — fixed-program design, data flow, performance
  history, and rejected experiments.
- `tests/bench/README.md` — standalone experiment staging area and how to run
  probes on the NPU; `tests/bench/soak/` holds the power/fan soak harnesses and
  the measurement-configuration check.
- `integrations/llama.cpp/README.md` — the experimental llama.cpp GGML backend
  and how it is attached to a llama.cpp checkout (symlink plus a hook patch).
  `integrations/llama.cpp/TODO.md` holds its design notes, measurements and
  build recipes.
- `~/Projects/llvm-aie/PERF_OPT_GUIDE.md` — Peano/clang loop and optimization
  guide.
- `~/Projects/llvm-aie/llvm/lib/Target/AIE/AIE2Schedule.td`,
  `AIE2Slots.td`, and `~/Projects/llvm-aie/llvm/test/MC/AIE/` — instruction
  latencies, VLIW slots, bundle syntax, and assembler tests.

## Before optimizing a kernel

1. Try the **pointer-increment** form of the loop before heavier changes.
   Indexed loops often prevent post-increment addressing and can be several
   times slower. Measure both forms: pointer rewrites regress recurrence-bound
   kernels such as the q4_0 mmul (see the counterexample in
   `src/quants/NOTES.md`).
2. Sweep `unroll(enable)`, `unroll_count(N)`, and `unroll(full)` on the actual
   loop. Check both cycles and `.text`; full unrolling can be fast but large.
3. Measure with the existing harness. Use `src/quants/tests/mmul.py` and
   `unpack.py` for quant kernels, `tests/bench` for standalone loop probes, and
   `src/program` timing for the fixed program.
4. Inspect the disassembly before and after. The copy benchmark shows the
   compiler can beat hand assembly when the source shape is right, and the
   q4_0 recurrence shows the opposite when the pipeliner cannot schedule the
   loop.
5. Use backend flags only after the disassembly shows a specific problem. The
   loop experiments found the common `aie-*` unroll/pipeliner/register flags
   had no effect on the copy kernels.

## Constraints

- Keep production headers usable by the host loader; device-only code belongs
  in `.aie.h` files.
- Do not silently relax exactness. New fast/relaxed modes need a named variant
  and their own verification policy.
- Report cycles, `weights/cycle`, bytes/weight, `.text`, and stack frame when
  changing a kernel.
- Preserve the transport-tile and repack contracts unless the change is
  explicitly a new format.
