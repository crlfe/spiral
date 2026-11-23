# `ggml-spiral` backend tasks

This directory is the llama.cpp backend for the Spiral NPU program.  The files
live in the Spiral tree; `llama.cpp/ggml/src/ggml-spiral` is a symlink to this
directory and the llama.cpp-side seam is `llama-hooks.patch`.  See `README.md`
for the layout and build steps.

## Move TODO

- [x] Move `ggml-spiral.cpp`, `smoke.cpp`, `CMakeLists.txt` and this file out
      of llama.cpp into `integrations/llama.cpp/`.
- [x] Default `SPIRAL_PROGRAM_DIR` to the sibling `src/program` so no checkout
      path is hardcoded.
- [x] Capture the llama.cpp seam as `llama-hooks.patch`; that patch is the only
      tracked llama.cpp delta.
- [ ] Re-verify `llama-hooks.patch` on each llama.cpp rebase: the registry
      include/registration and the two CMake lines are the whole seam.
- [ ] Reconfigure `build-spiral`/`build-both` after a move or rebase; their
      CMake caches hold absolute paths.
- [ ] Decide whether to upstream the backend once the NPU path is competitive;
      until then this stays a patch on a local checkout, not an official fork.

Goal: a llama.cpp backend that runs `GGML_OP_MUL_MAT` on the AMD XDNA2 NPU
using the fixed Q4_0/Q6_K matmul program in `src/program`.
The first target is Gemma 4 E2B: Q4_0 attention/FFN tensors plus the native
Q6_K output head.

The point is productive inference without the laptop's fans ramping up: the
NPU is chosen for efficiency, not peak throughput, and CPU/iGPU numbers are
comparison baselines rather than targets to beat.  A later phase may split
work between the NPU and the iGPU, but that is parked until the NPU path is
closer to competitive (see "NPU + iGPU split").  Keep the backend's per-tensor
interface compatible with a second device.

The device design is already validated and documented in
`src/program/{README,NOTES,TODO}.md`.  This backend should
consume that program rather than recreate the old bf16 prototype design.

## Current state

- The backend is maintained in this directory and built through the symlink.
  It is no longer developed on a llama.cpp branch, and the `feat-ggml-spiral`
  and `hacking-spiral-1` branch names in the phase notes below are history.
- Backend registration, the Spiral weights buffer type, and a synchronous
  `MUL_MAT` path are in place.
- Native Q6_K is enabled.  The host Q6_K repack now pre-splits canonical fp16
  `d` into an exact bf16 hi/lo pair, so real subnormal head scales are handled.
  The real `token_embd` slice check passes, and the Q4_0 Gemma model output
  matches CPU with the native Q6_K head.
- The weights buffer prepares canonical Q4_0/Q6_K tensors through
  `spiral::Program` and keeps one resident per-handle weight BO.  N=256 is
  padded to 512; K must be a multiple of 256.
- `spiral-smoke` exercises the registry hook, the weights buft, Q4_0 and
  Q6_K buffer paths, N=256 padding, two resident prepared handles, a
  three-segment combined chain against CPU, and the full 11-chunk Q6_K head
  with `--head`.
- The backend now claims CPU-compatible ops and executes them through an
  internal `ggml_backend_cpu` graph view.  The scheduler sees **1 split** for
  the Gemma prompt graph instead of hundreds of backend transitions.  The
  internal CPU backend defaults to one thread because the fallback runs many
  small graph views without a persistent threadpool; `SPIRAL_CPU_THREADS`
  overrides it.
- Graph-level gate+up fusion is in place.  Consecutive `MUL_MAT`s with the same
  input data (pointer and layout) and one shared K are prepared as a single
  combined `WeightSegment` call and executed as one wide matmul; the chain
  supports up to four segments.  The buffer keeps canonical quant bytes so
  combined preparation does not need the original host tensor.  Nodes
  separated by other ops are not chained: the scheduler's compute-buffer
  liveness assumes graph order, and computing a later matmul early can clobber
  an intermediate tensor.
- `SPIRAL_PROGRAM_DIR` defaults to the sibling `src/program` (resolved from
  `CMAKE_CURRENT_LIST_DIR`), and `SPIRAL_PROGRAM_BUILD_DIR` to its
  `build-production` output.  Both remain `CACHE PATH` overrides for vendored
  or out-of-tree builds.
- The source of truth for the device program is
  `src/program`:
  - `build.py --production` emits `aie.xclbin`, `generate_txn.h`,
    `design_meta.h`, and the C++ test binary.
  - `host/program.h` / `host/program.cc` provide `spiral::Program` with
    `prepare` / `run`, prepared weights, column offsets, and combined
    `WeightSegment` preparation.
  - The production program supports Q4_0 and Q6_K, 8x4 compute geometry,
    256-row tiles, 512-column bands, and up to 48 bands per call.

### Bring-up commands

The backend needs the symlink and the hook patch in the llama.cpp checkout:

```bash
# from the llama.cpp checkout, with Spiral a sibling directory
ln -s ../../../spiral/integrations/llama.cpp ggml/src/ggml-spiral
git apply /path/to/spiral/integrations/llama.cpp/llama-hooks.patch
```

```bash
cd <llama.cpp checkout>
cmake -S . -B build-spiral -G Ninja \
    -DGGML_SPIRAL=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_OPENMP=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-spiral --target ggml-spiral spiral-smoke -j 8
cd build-spiral/bin && ./spiral-smoke
cd build-spiral/bin && ./spiral-smoke --head
```

The smoke test prints the ggml registry (Spiral0 + CPU), checks Q4_0/Q6_K
buffer paths, N=256 padding, and two resident handles.  `--head` additionally
runs the full 11-chunk Q6_K head.  `SPIRAL_AIE_INCLUDE_DIR` must point at the
`mlir_aie/include` directory containing `aie/Runtime/TxnEncoding.h`.

## Architecture decisions

1. **Use `spiral::Program` directly.**
   - One `Program` owns the xclbin, hw context, kernel, instruction BO, input
     BO, and output BO.
   - The backend must not duplicate the transaction generation or quant
     repack logic.

2. **Store canonical quant weights in the ggml weight buffer.**
   - The Spiral weights buffer type keeps the original Q4_0/Q6_K (or other
     supported) canonical bytes.
   - On first use, call `Program::prepare(shape, raw, raw_stride, 0)` and
     cache the returned `PreparedWeights` handle.
   - Release the canonical copy after preparation if the buffer can do so
     without violating ggml ownership rules.

3. **Prepared weights must have per-handle resident BOs.**
   - The current `spiral::Program` keeps one shared weight BO and only the
     most recently prepared handle is resident.
   - A token runs ~276 matmuls.  A shared most-recent BO would re-upload a
     weight stream on nearly every matmul.
   - Change `PreparedWeights` to own an `xrt::bo` allocated with
     `kernel.group_id(3)`; `Program::run` should use that BO directly.
   - The old prototype proved XRT can hold hundreds of resident BOs.  Native
     Q4_0/Q6_K E2B weights are about 1.38 GB, so resident per-tensor BOs are
     practical.

4. **Chunk rows by `inputs_vecs = 4`.**
   - Decode: one row plus three zero rows.
   - Prefill: chunks of four rows.
   - Inputs are packed as bf16 bit patterns in `[4][weights_rows]` order.
   - Outputs are F32 in `[4][weights_cols]` order.

5. **Shape contract.**
   - `weights_rows` must be a multiple of 256.
   - `weights_cols` must be a multiple of 512.
   - E2B K values are 1536/2048/4096/6144/12288 and already satisfy the row
     constraint.
   - N=256 tensors (`attn_k`, `attn_v`, `inp_gate`) need a canonical padded
     copy to 512 columns.  Reuse the padded reference cases from
     `src/program/build/golden` (`q4_0_*x256p512`).
   - Initially reject shapes that need K padding or unsupported types with a
     clear error.  Add generic K padding later.

6. **Keep the ggml device contract that worked in the prototype.**
   - Device type `IGPU` so `--device Spiral0 -ngl 999` selects the NPU.
   - `get_memory()` reports host RAM, not the NPU L2.
   - The weights buft returns `is_host = false`.
   - Default buffer type remains the CPU buffer; activations and outputs stay
     in CPU memory.

7. **No custom bf16 tile packer.**
   - Remove the old 2048x2048 bf16/64x64 packing path.
   - `Program::prepare` owns canonical repack, block-major stream assembly,
     and padding at the end of the stream.

## Phases

### Phase 0: repository and backend skeleton

- [x] Confirm the working branch is `feat-ggml-spiral`.
- [x] Remove the stale untracked `fw/` prototype output.
- [x] Add `ggml/src/ggml-spiral/CMakeLists.txt`.
- [x] Add `ggml/include/ggml-spiral.h` and the backend registration hook in
      `ggml/src/ggml-backend-reg.cpp`.
- [x] Add a minimal backend that registers as `Spiral0` and reports host
      memory.
- [x] Build with `-DGGML_SPIRAL=ON` and verify device enumeration.

### Phase 1: link and validate the new device program

- [x] Prototype linking model: compile
      `src/program/host/program.cc` into the smoke target
      and include its generated build directory.
- [x] Add CMake definitions for the xclbin path and generated header
      directory.
- [x] Add `spiral-smoke`; it loads the production xclbin, prepares one Q4_0
      and one Q6_K golden case, runs them, and checks the outputs.
- [x] Verify the production xclbin loads and both formats work before touching
      ggml `MUL_MAT`.
- [x] Decision: do not vendor the program.  `SPIRAL_PROGRAM_DIR` points at
      `src/program` (relative by default) and the generated headers are read
      from its build directory; the old `fw/` copy is not used.

### Phase 2: weight buffer type and prepared weights

- [x] Add a Spiral weights buffer type with one allocation per tensor
      (`alloc_buffer_n`).
- [x] Prepare weights in `set_tensor` using the canonical tensor bytes and
      `raw_stride`; the prepared stream is resident per handle.
- [x] Implement per-handle resident weight BOs in `spiral::PreparedWeights`
      and compact per-handle group strides.
- [x] Handle N=256 -> 512 padding by building a canonical padded copy and
      verify it through the buffer path.
- [x] Reject partial `set_tensor` writes and non-contiguous weights.
- [x] Test `set_tensor` for Q4_0 and Q6_K through the real ggml buffer path.
- [x] Keep two prepared handles alive and run them out of order (residency
      smoke test).

### Phase 3: `MUL_MAT` execution

- [x] Implement `supports_op` for dense, contiguous `GGML_OP_MUL_MAT` with
      F32 activations/output and Spiral weights.
- [x] Implement `supports_buft` and expose the weights buft through the
      backend proc-address hook.
- [x] Map ggml K/N to `weights_rows`/`weights_cols`.
- [x] Chunk rows into groups of four; zero-pad short chunks.
- [x] Convert F32 activations to bf16 bit patterns `[4][K]`.
- [x] Call `Program::run` and gather valid rows/columns into the F32 output.
- [x] Use one synchronous slot.
- [x] Verify Q4_0 M=4 and Q6_K M=4 through the ggml buffer path.
- [x] Verify N=256 padded and the full 11-chunk Q6_K head through the
      backend path.
- [ ] Verify M=1, M=17, and K=6144/12288 through the backend path.

### CPU fallback for scheduler transitions

- [x] Add an internal CPU backend to the Spiral backend.
- [x] Claim host-compatible ops in `supports_op` and execute consecutive runs
      with `ggml_graph_view`-style subgraphs on the CPU backend.
- [x] Keep Spiral `MUL_MAT` on the NPU and delegate other host ops to CPU.
- [x] Set the fallback CPU thread count (one thread by default).
- [x] Verify Gemma Q4_0 output still matches CPU.
- [x] Reduce the prompt graph to 1 scheduler split.
- [x] Make the CPU thread count configurable via `SPIRAL_CPU_THREADS`;
      default is 1 because the fallback runs many small graph views and has no
      persistent CPU threadpool.  More threads made it slower.
- [ ] Verify decode-graph split count and overlap behavior.
- [x] Fix native Q6_K real-weight accuracy (subnormal fp16 scales) and enable
      Q6_K in `supports_op`.
- [x] Real `token_embd` Q6_K slice check (`spiral-smoke --q6-slice <raw>`)
      passes with max_abs 0.026 on max_cpu 10.07.

### Phase 4: correctness on real models

- [x] Q4_0 Gemma 4 E2B matches CPU with native Q6_K
      (`The capital of France is Paris.`).
- [x] Test the Q4_0 llama.cpp Gemma 4 E2B checkpoint with native Q4_0.
- [x] Test the Q6_K output head buffer path across 11 chunks
      (`spiral-smoke --head`).
- [x] Re-enable native Q6_K after the subnormal-scale fix and re-check greedy
      output.
- [ ] Test `gemma-4-E2B-it-bf16.gguf` with greedy decoding against CPU.
- [ ] Confirm no late-token divergence caused by bf16 dequantization.
- [ ] Record max relative error and token-match lengths.

### Real-query throughput

Gemma 4 E2B Q4_0, native Q6_K head, 30-token prompt and a 300-token greedy
answer with `llama-completion -no-cnv --ignore-eos --temp 0`, one Spiral CPU
fallback thread, PM QoS off, iGPU SCLK capped at 2400 MHz, NPU pmode `turbo`.

| device | prompt eval | generation | per token |
| :-- | --: | --: | --: |
| iGPU (`ROCm0`, gfx1150) | 383.1 t/s (`pp8`) | 55.11 t/s | 18.1 ms |
| CPU (`-ngl 0`, 12 threads) | 224.1 t/s (`pp8`) | 35.78 t/s | 28.0 ms |
| Spiral NPU | 111.5 t/s (`pp8`) | 17.78 t/s | 56.2 ms |

The iGPU is fastest, the CPU next, and Spiral slowest; throughput is context,
not the acceptance criterion.  See "Efficiency baseline" below for the
temperature and fan criteria.

Both paths stream the same ~1.4 GB/token of active weights; the NPU path adds
~19 ms/token of fixed launch/fill (about 240 calls/token at an ~80 us floor)
plus the PCIe hop.

`SPIRAL_CPU_THREADS=1` is fastest for the NPU path (2/4/8 threads give
16.8/14.5/10.5 t/s `tg16` and 106/95/68 t/s `pp8`); CPU inference peaks at 12
threads (36.1 t/s `tg16`).  Those were measured before the pmode baseline, so
treat them as ratios rather than absolute values.

PM QoS is implemented in `src/program` and inherited by the backend.  It is
opt-in (`SPIRAL_DMA_LATENCY=1`, `SPIRAL_DMA_LATENCY_PATH` overrides the node)
and fails hard if enabled when the node is inaccessible.  With the node
writable, the effect is workload-dependent (three runs each, alternating
order): `pp8` 118.1 t/s with PM QoS held vs 111.4 t/s without (+6%), `tg16`
16.84 vs 18.11 (-7%), and a 300-token generation 15.2-15.3 vs 16.8-17.3 t/s
(-10%).  Holding the CPU out of deep idle helps prompt submission but hurts
decode by lowering boost clocks, so it stays off by default.

### Efficiency baseline

The acceptance criterion is productive inference that stays quiet, not peak
throughput.  Record these signals for every CPU/NPU comparison run:

| signal | source |
| :-- | :-- |
| CPU temperature | `k10temp` `Tctl` |
| iGPU temperature | `amdgpu` hwmon `temp1_input` |
| fan speed | `hp`/`acpi_fan` `fan*_input`, loudest |
| host CPU utilisation | `/proc/stat` busy delta and thread count |
| NPU clocks | `DRM_IOCTL_AMDXDNA_GET_INFO` (MP-NPU and H) |
| throughput | `pp8`, `tg16`, 300-token generation |

Power is not measured.  Package RAPL is root-only in the measurement container
and `SOCKET_POWER` has to be integrated from a coarse sensor, so fan RPM and
Tctl are the proxy: they are what a user notices and need far fewer
approximations.

Procedure: cool to idle (fans off, `Tctl` below ~45 C), then run a long greedy
generation per device (thousands of tokens) while sampling about once per
second, continuing until `Tctl` and fan RPM have been flat for 20 s so peaks
are steady-state.  Report peak `Tctl`/fan and throughput, then let the device
return to the idle baseline (fan 0, `Tctl` within 2 C) before the next device.
Proposed acceptance bar (adjust once measured): a usable rate with fan RPM
within roughly 150 RPM of idle and `Tctl` within roughly 10 C of the idle
reading.

No current path meets this bar (the quietest, FastFlowLM, still reaches 54 C
and 2.8k RPM), so treat it as the target rather than a pass/fail gate.

Steady-state soaks (Gemma E2B Q4_0, 3000 tokens per device, one CPU fallback
thread, peaks after a 20 s plateau window, cool-back to idle between devices):

| device | tokens | t/s | peak Tctl | peak iGPU | fan1 | fan2 | plateau | cool-back |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| iGPU (`ROCm0`) | 3000 | 52.42 | 54.2 C | 54.0 C | 2793 | 2597 | none in 59 s | 123 s |
| CPU (`-ngl 0`, 12 threads) | 3000 | 22.17 | 91.0 C | 56.0 C | 4845 | 4657 | none in 138 s | 126 s |
| Spiral NPU | 3000 | 16.15 | 73.6 C | 56.0 C | 4834 | 4657 | 65 s | 121 s |

Both `hp` fans always move together (150-190 RPM apart), so they follow one EC
curve with a knee just above 53 C.  FastFlowLM, which runs the same NPU at the
same MP/H clocks, peaks at 54.0 C with 2818/2626 RPM, while the Spiral backend
reaches 73.6 C and 4834/4657 RPM.  The NPU-only head-soak (1% CPU) sits at
60.6 C / 2818 RPM, so the extra heat in the Spiral decode path is the CPU
fallback and host submission, not the NPU.  The CPU is the hottest at 91.0 C
and throttles.  See `src/program/NOTES.md` for the FastFlowLM row and the full
per-fan analysis.

Short runs mislead: the iGPU and CPU "no fan" observations came from tasks
shorter than the thermal time constant, and both ramp the fans in a soak.
This describes this workload - memory-bound decode with ~240 NPU calls/token
and ~9 ms of CPU fallback - not NPU work in general.

A pure NPU busy loop (`src/program ./test --head-soak 20000`, 1% CPU busy)
reaches Tctl 60.5 C, iGPU 50.0 C and ~2.9k RPM over 152 s, so the NPU's heat
lands in both the CPU control temperature and the iGPU/SoC sensor and spins
the fans without CPU activity.  The same loop runs ~4% faster under pmode
`turbo` (152.3 s vs 155-159 s) at the same temperatures.

### NPU tuning: what actually matters

Same 3000-token NPU soak, machine otherwise in its normal boot state:

| configuration | t/s |
| :-- | --: |
| boot state (pmode `default`) | 14.70 |
| + fclk 7 (1960 MHz) | 14.64 |
| + fclk 7 + socclk 7 (1472 MHz) | 14.65 |
| pmode `turbo`, no clock pins | 16.38 |

`pmode` is the whole difference.  fclk and socclk are not worth pinning, so
they stay on DVFS.  MP/H clocks are identical (1408/2112 MHz) under both pmode
settings, so the gain is not a clock effect.  pmode does not survive a reboot;
a small `spiral-npu-tune` systemd unit re-applies it on boot and on resume.

The NPU clocks are **not** in sysfs.  The amdxdna driver exposes them through
`DRM_IOCTL_AMDXDNA_GET_INFO` (UAPI in `include/uapi/drm/amdxdna_accel.h`,
implementation in `aie2_pci.c`):

| param | returns |
| :-- | :-- |
| `DRM_AMDXDNA_QUERY_CLOCK_METADATA` | `mp_npu_clock` and `h_clock`, each `{name, freq_mhz}` |
| `DRM_AMDXDNA_QUERY_RESOURCE_INFO` | `npu_clk_max`, max/current TOPs, task counts |
| `DRM_AMDXDNA_GET_POWER_MODE` | power mode (4 = turbo) |

`tests/bench/soak/xdna.py` reads them through `/dev/accel/accel0`; the harness
samples them with temperatures, iGPU sclk and fan RPM.  Head-soak clocks are
constant at MP 1408 / H 2112 MHz, so NPU decode is not clock-throttled.  The
iGPU soak shows `sclk` 671-2400 MHz, mean 2323, against the 2400 MHz `manual`
bound: the cap is where the iGPU sits under load, not a lock near idle.

Measurement conditions for every run above: the machine's normal boot state,
the iGPU SCLK capped at 2400 MHz (DPM `manual`, OD bounds 600/2400 against a
3100 MHz hardware range), platform profile `balanced`, CPU governor
`powersave`, PM QoS off, NPU pmode `turbo`.  fclk/socclk are printed as
observations, not asserted, because pinning them was measured to change
nothing.

These are checked by `tests/bench/soak/hostconfig.py` at the start of every
harness run; it aborts on drift (or `--allow-drift` for an exploratory run that
must not be compared with the recorded table).  Sensor paths are resolved by
hwmon name rather than fixed index.

The iGPU SCLK cap is the one setting that can silently drift: a reboot or
driver reload restores the 3100 MHz hardware range, which raises temperature
and fan RPM for the same work.

### iGPU (ROCm) build

The iGPU is measured from a separate build so the Spiral backend does not
intercept the graph:

```bash
export HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang
export HIP_PATH=/opt/rocm/core-10.0
export HIP_DEVICE_LIB_PATH=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode
cmake -S . -B build-rocm -DGGML_HIP=ON -DGPU_TARGETS=gfx1150 \
      -DGGML_SPIRAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-bench llama-completion -j 12
```

The container needs `/dev/kfd` (234:0) and `/dev/dri/renderD128` (226:128).
The binaries also need
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib` because they link
`libhipblas.so.3` without an rpath.  `ROCm0` is `gfx1150` (Radeon 890M) with a
55296 MiB UMA aperture.

llama-bench, Gemma E2B Q4_0: `pp8` 383.1 t/s, `tg16` 55.30 t/s, `pp512`
3739 t/s.  `manual` sets clock bounds, not a fixed clock (recorded bounds
600-2400 MHz, hardware range to 3100 MHz), so record the force level, bounds
and peak observed sclk with every number.  Sustained throughput, temperature
and fan behaviour depend heavily on this (see "Efficiency baseline").  The
bounds are the only iGPU setting that is pinned; fclk/socclk stay on DVFS.

### NPU + iGPU split (parked)

`build-both` compiles both backends into one binary:

```bash
HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang HIP_PATH=/opt/rocm/core-10.0 \
  HIP_DEVICE_LIB_PATH=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode \
  cmake -S . -B build-both -DGGML_HIP=ON -DGGML_SPIRAL=ON \
        -DGPU_TARGETS=gfx1150 -DCMAKE_BUILD_TYPE=Release
cmake --build build-both --target llama-bench llama-completion -j 12
```

`llama-bench --list-devices` shows `ROCm0` and `Spiral0`, and
`-dev ROCm0,Spiral0 -ngl 999 -ts 0.75,0.25` is honoured by the loader:
54 layer assignments to ROCm0, 18 to Spiral0 (layers 27-35).

Execution is not usable yet:

- With the normal backend (MUL_MAT on the NPU, CPU fallback for everything
  else) the graph fails on cross-device views:
  `spiral: no CPU fallback for VIEW (cache_k_l13 (view))`, and a second run
  moved the failure to `Vcur_normed-0 (view)`; `-nkvo` does not help.  Gemma
  reuses KV across layers and permutes tensors across the split boundary, and
  the Spiral CPU fallback cannot consume ROCm buffers.
- Restricting the backend to MUL_MAT (`SPIRAL_MATMUL_ONLY=1`, diagnostic)
  removes the errors but runs **zero** NPU matmuls: ROCm also supports the
  Q4_0/Q6_K MUL_MATs, so the scheduler gives the Spiral-assigned layers'
  matmuls to ROCm and the run measures the iGPU alone (~52 t/s `tg16`).

A real split needs one of: cross-device view/input handling in the Spiral
backend (materialised boundary tensors), MUL_MAT-only support plus a way to
force those ops onto Spiral (device priority or graph rewrite) with
split-input copies, or a graph-level split by tensor rather than by layer.
No split performance number is recorded yet.

The path is parked until the NPU offers convincing performance: close enough
that a 50/50 split is attractive, and on a much larger model where the split
can relieve a real shared bottleneck.  The per-tensor interface is kept for
that.

### Q4_0_packed16 (trialled, dropped)

`src/quants` still has a near-exact `q4_0_packed16` layout.  It was integrated
as a build-time Q4 block choice because both blocks cannot be linked into the
same 16 KiB program (overflow by 224 B).  Kernel 256x64: Q4_0 8328 cycles,
packed16 7397 (-11.2%).

The variant was then removed from `src/program`, so the backend always uses
canonical Q4_0 and needs no environment variable.  The exact Q4_0 kernel
improved to 7883 cycles, which shrinks the gap to 6.2%, and packed16 is
near-exact rather than bit-exact, so replacing the canonical path is not
justified.  The format stays in `src/quants` for future work.

### Graph-level gate+up fusion

- [x] Keep the canonical padded quant bytes in the weight buffer so weights can
      be repacked again for a combined call.
- [x] Prepare per-tensor handles lazily from those bytes (`ensure_prepared`).
- [x] Detect consecutive supported `MUL_MAT`s with the same input data (pointer
      and full layout), same K, same format, and a combined column count
      within one band chunk.  Up to four segments per chain.
- [x] Do **not** chain across intervening ops.  Attempting it produced correct
      device output in isolation but wrong model output because the scheduler's
      compute-buffer allocator aliases tensors whose graph-order lifetimes do
      not overlap; running a later matmul early clobbered a gap tensor.
- [x] Prepare and cache one `WeightSegment[N]` handle per chain.
- [x] Run the chain as one wide matmul and scatter each segment's column range
      into the matching node output.
- [x] Fall back to individual `MUL_MAT` when detection or combined preparation
      fails (`SPIRAL_NO_PAIR=1` disables chaining for comparison).
- [x] Add self-checks (`spiral-smoke` `3-segment chain`) that run three Q6_K
      matmuls through the Spiral and CPU backends, and verify all three
      combined segments in `src/program ./test --combined-timing`.
- [x] Measure the effect: Gemma Q4_0 with native Q6_K, one CPU fallback
      thread, after the Q4_0 scale rebuild.

Pairing mostly removes launches; prompt processing benefits more:

| mode | pp8 | tg1 | tg16 |
| :-- | --: | --: | --: |
| individual (`SPIRAL_NO_PAIR=1`) | 106.1 t/s | 16.2 t/s | 17.3 t/s |
| gate+up chain | 122.2 t/s | 17.5 t/s | 18.4 t/s |

These were measured before the pmode baseline, so their absolute values are
not comparable with the throughput table above; the relative win is the point.

`pp8` and `tg16` carry roughly +-3-9 t/s and +-0.7 t/s run variance.  The
6144+6144 shapes gain the most; 12288+12288 is close to neutral, so the win is
smaller than the per-call combined-kernel microbenchmark suggests.  The
combined handles duplicate the paired weights in NPU BOs and cost roughly 1 s
of one-time preparation on the first graph.

Gemma's Q/K/V projections are separated by QK-norm and RoPE nodes, so they
are not chained; the multi-segment path currently benefits gate+up only.  The
three-segment smoke check exists to keep that path working for models where
the projections are contiguous.

### Host-mediated GEGLU fusion (no TXN chaining)

The FFN activation now runs in the device epilogue while the gate+up and down
matmuls remain separate launches.  A matcher recognizes
`gate+up -> GLU(GEGLU, split) -> down` with a single consumer per intermediate,
an F32 contiguous GLU output, and a supported down matmul.  The gate+up nodes
are skipped; the combined gate+up call runs at the GLU node's graph position
with `kEpilogueGeglu`, so the activation is written when its lifetime starts
(no early-write aliasing).  The down matmul then reads the GLU output as a
normal host tensor.

- [x] Matcher, execution at the GLU position, and the `SPIRAL_NO_GEGLU_FUSE=1`
      fallback to the CPU GLU.
- [x] `SPIRAL_GEGLU_CHECK=1` diagnostic: runs the same pair without the
      epilogue and compares the device GEGLU against a host GELU.  Real Gemma
      weights: 5e-4 to 3e-3 global relative per layer.
- [ ] In-TXN chain: retarget the down input to the GEGLU output instead of
      crossing the host (needs the production chain API).
- [ ] Pipeline the GEGLU row chunks; decode has one chunk, and the current
      path runs one chunk at a time because the output staging is shared.

64-token greedy A/B, 3 alternating runs, `-n 64`, `temp 0`:

| mode | output | cpu fallback | eval (median) | prompt eval |
| :-- | :-- | --: | --: | --: |
| CPU GLU (`SPIRAL_NO_GEGLU_FUSE=1`) | `Paris.` | 93.7 ms / 32 graphs | 51.65 ms/token | 76.7 ms |
| device GEGLU | identical | 63.8 ms / 32 graphs | 51.31 ms/token | 73.3 ms |

The CPU fallback drops by ~0.9 ms/token (the GLU); wall time gains ~0.3 ms
because part of the old CPU work overlapped with other host work.  `fused_geglu`
counts one per FFN block (35 per graph).  No NPU chaining, no dispatch saving,
and the intermediate still crosses the host.

### Pipelined NPU submissions

`Program` exposes `submit()` / `finish()` with a small pool of independent
instruction/input/output BOs and `xrt::run` objects.  Starting the next call
before waiting on the previous hides the launch/pipeline-fill bubble.
`SPIRAL_PIPELINE_SLOTS` sets the depth (default 2; 2 is enough, 4/8 do not
improve the head).

- Q6_K head pass: ~7.7-8.2 ms (1 slot) -> ~6.5-6.9 ms (2 slots), ~16% wall.
- Prompt: every node with rows > 4 has multiple chunks.  `pp8` improves from
  95.3 to 112.5 t/s at depth 2; depth 4 is 111.6 t/s.
- Decode `tg16`: 18.0 -> 18.3 t/s; only the head is multi-chunk at `rows=1`.
- Cross-node look-ahead (submit the next same-input matmul early, defer only
  the host unpack) is implemented and correct but within noise on Gemma
  (17.5 vs 18.1 t/s `tg16`, 110.7 vs 114.8 `pp8`).  Queue order matters:
  submitting the next call before the current one delays the current result,
  so it is opt-in with `SPIRAL_LOOKAHEAD=1`.

### Execution profile

Profiling is available through:

- `SPIRAL_PROFILE=1`: totals at backend free (includes `fused_softcap=`).
- `SPIRAL_PROFILE_GRAPH=1`: per-graph wall/CPU/NPU/pack/gather.
- `SPIRAL_PROFILE_NODES=1`: one line per MUL_MAT with shape and time.

64-token decode profile, one CPU fallback thread, 9.4 ms/token of CPU
fallback in total:

| component | per token |
| :-- | --: |
| clean NPU calls | ~48 ms |
| CPU fallback | 9.4 ms |
| host/sampling | ~10 ms |
| **total** | **~56 ms (17.7 t/s)** |

The CPU fallback is led by the logit softcap: `UNARY` (tanh over 262144
values) 5.1 ms/token, then `GLU` 1.2, `FLASH_ATTN_EXT` 1.2, `RMS_NORM` 0.8,
`MUL` 0.6, `ROPE` 0.2.  More `SPIRAL_CPU_THREADS` speeds the tanh itself but
slows the token (thread overhead across ~900 fallback calls/token), so one
thread stays the default.

Per-call NPU time follows `~80 us + weight_bytes / 48 GB/s`, so bytes/time is
not a bandwidth measurement: the fixed launch floor dominates small matmuls
and decode makes ~240 calls/token (~19 ms/token of fixed overhead).  The
compact weights BO has one `kL2StreamBytes` guard block because the device
prefetches past the last block; without it the over-read raised AMD-Vi
IO_PAGE_FAULTs.  `src/program ./test --cold-timing` reports min/p50/p95/max
for the streaming path; use medians/p95, not means, when comparing builds.

The largest remaining wins are therefore:

1. **Fewer calls** (~19 ms/token of fixed cost): head chunk reduction, and
   cross-gap schemes only with compute-buffer alias safety.
2. **The softcap on the NPU** (5.1 ms/token): **resolved** — the backend
   matches `MUL_MAT -> SCALE(1/cap) -> TANH -> SCALE(cap)` and runs the
   softcap in the bf16 tile epilogue.  `SPIRAL_NO_SOFTCAP_FUSE=1` restores the
   CPU fallback.  Measured on Gemma 4 E2B Q4_0, 29 decode runs: ~50.4 ms/token
   fused vs ~53.7 ms unfused (about 3.3 ms/token, 6%); the same 40-token greedy
   text is produced.  The fused value uses the approximate bf16 vtanh, so the
   `q4_0_1536x512_sc30` golden keeps its own `5e-2` policy (worst 0.725).
3. **Streaming rate and the head** (~3 ms at 48 -> 55 GB/s; head ~7 ms).
4. CPU fallback is now smaller: softcap and GEGLU are on the device.  The
   remaining fallback is norms, RoPE, attention, and small elementwise ops;
   do not optimize it before the chain/norm/attention links.

Gate+up chaining removes one call per layer and helps prompt processing; the
12288+12288 case is close to neutral, so wider is not always faster.

> **Resolved (2026-11-17 18:45):** the Q4_0 host repack/device unpack now
> pre-bias scales by `2^10`.  After rebuilding `build.py --production`, all
> `spiral-smoke` cases (Q4_0, Q6_K, N=256 padding, residency, `--head`, and
> `3-segment chain`) pass, and the Gemma Q4_0 model returns
> `The capital of France is Paris.` again.

### Wait stalls (resolved)

The 45-50 ms `wait_us` spikes were host CPU-quota throttling, not the driver.
Kernel tracing of a run with the spikes showed every device job completing in
<=2.6 ms and every ioctl in <=3 ms, while the submitting thread was switched
out runnable (`prev_state=R+` -> `swapper`) and resumed ~47 ms later on an
idle CPU, with no driver, PM or IOMMU activity in the window and no cgroup
throttling visible from inside the throttled cgroup.  With the quota removed,
a 64-token run has zero calls above 15 ms and median `wait_us` ~95 us.
PM QoS, cmdlist and pipeline-depth experiments never addressed this because
the delay was never in the device.

### Phase 5: performance baseline

- [x] Record initial pp8/tg1 for CPU and Spiral Q4_0 with native Q6_K.
- [x] Confirm the prompt graph uses 1 scheduler split with CPU fallback.
- [ ] Record resident weight bytes, per-op NPU time, pack/dispatch/sync, and
      decode-graph splits per token.
- [ ] Compare with the old `hacking-spiral-1` bf16/Q4_0 baseline.
- [ ] Confirm resident weights are ~1.38 GB rather than 6.34 GiB.
- [x] Identify whether the remaining cost is matmul, launch, scheduler
      transitions, or CPU work between segments: ~48 ms clean NPU calls
      (`~80 us + bytes/48 GB/s`), ~19 ms of per-call fixed cost, 9.4 ms of CPU
      fallback, and ~10 ms host/sampling per token.

### Phase 6: pipelining and call amortization

- [x] Two-slot input/output pipelining.  `Program::submit`/`finish` own a pool
      of instruction/input/output BOs and `xrt::run` objects; the backend uses
      it for every multi-chunk node.  `SPIRAL_PIPELINE_SLOTS` sets the depth
      (default 2).  Head pass ~16% faster; `pp8` 95.3 -> 112.5 t/s and
      `tg16` 18.0 -> 18.3 t/s.
- [x] Combined column matmuls at the backend level for gate+up.  Q/K/V is
      blocked by compute-buffer aliasing across the QK-norm/RoPE gaps.
- [x] Cross-node look-ahead (submit the next same-input matmul early, defer
      only the host unpack).  Correct but within noise on Gemma; opt-in with
      `SPIRAL_LOOKAHEAD=1`.
- [x] Hold `/dev/cpu_dma_latency` at 0 for the Program lifetime.
      Opt-in via `SPIRAL_DMA_LATENCY=1`; it helps prompt (+6% `pp8`) and
      hurts decode (-7% `tg16`, -10% on a 300-token answer), so it is off by
      default and fails hard when enabled with an inaccessible node.
- [ ] Keep the old local async mode out unless the scheduler can actually
      overlap with one copy; the prototype showed no gain.

### Phase 7: graph-level fusion

- [ ] Add a graph pass or fused ggml op that recognizes FFN gate+up sharing
      the same input.
- [ ] Use `Program::prepare(shape, WeightSegment[], count)` to run gate+up as
      one wide matmul.
- [ ] Add Q/K/V fusion for the 15 KV layers.
- [ ] Keep the per-shape fallback for shared-KV layers (Q only).
- [ ] Do not attempt FFN gate+up+down or device GEGLU until the simpler
      combined calls are integrated and measured.

### Phase 8: broader decode path

- [ ] Measure backend splits/token and the CPU segments between them.
- [ ] Decide whether to support more ggml ops on Spiral (RMS_NORM, GEGLU,
      ROPE, etc.) or rely on the scheduler.
- [ ] Only revisit device GEGLU/FFN fusion after the per-op path and combined
      calls are stable.

## Open questions

1. Vendor the host wrapper and xclbin into `ggml-spiral/fw`, or include
   `src/program` directly for the prototype?
2. Should per-handle BO ownership live in `PreparedWeights` (recommended) or
   in a backend-side `PreparedWeight` map?
3. Do we require generic K padding, or is K-multiple-of-256 sufficient for the
   E2B target?
4. How should the backend expose combined gate+up/QKV to the ggml scheduler:
   a graph rewrite, a custom op, or a backend graph-plan callback?
5. How do we bound "productive but quiet"?  `Tctl`, iGPU temperature, fan RPM
   and CPU utilisation now have a repeatable harness; the acceptance bar is
   still proposed rather than agreed (see "Efficiency baseline").

## Risks

- **Weight BO residency:** resolved with one resident BO per prepared handle;
  combined chains duplicate their segment weights in NPU BOs.
- **Padding:** canonical-padded N=256 copies must be byte-identical to the
  reference padded golden cases.
- **Views/aliases:** weights must be dense and contiguous; view handling is
  deferred.
- **Scheduler isolation:** supporting only `MUL_MAT` creates many backend
  splits per token.  The new program improves per-matmul time but not the
  split overhead.
- **XRT resources:** hundreds of resident BOs and one exclusive hw context
  must be validated for the native-quant weight set.
- **Numerics:** native Q4_0/Q6_K should be closer to CPU than bf16-dequantized
  weights, but still validate with greedy generation.
- **Head chunking:** the Q6_K head is 11 launches per token at 48 bands per
  chunk; a one-launch expansion would need a much larger weight BO and
  additional fill parts.
- **Stale artifacts:** the old `hacking-spiral-1` design and its `fw/` output
  are not part of this backend.  Do not resurrect or link them accidentally.

## Definition of done

- `ggml-spiral` builds cleanly with `-DGGML_SPIRAL=ON`.
- llama.cpp offloads all Gemma 4 E2B matmuls with `--device Spiral0 -ngl 999`.
- Q4_0 and Q6_K weights stay in native canonical form on the host; resident
  prepared BOs are reused across token steps.
- Greedy token output matches the CPU for a representative completion.
- pp/tg and backend splits are recorded and compared with the old prototype.
- A 300-token generation runs at a usable rate with both fan tachometers and
  `Tctl` staying near the idle baseline, recorded for CPU and NPU runs.
- The combined gate+up/QKV path has either a measured graph-level integration
  or a documented reason it remains benchmark-only.
