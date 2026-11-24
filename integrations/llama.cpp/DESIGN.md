> Role: reference. Current as of 2026-11-24. History: `git log -- integrations/llama.cpp/DESIGN.md`.

# `ggml-spiral` backend design

The backend runs `GGML_OP_MUL_MAT` on the NPU through `spiral::Program`, and
executes every other graph op on an internal CPU backend. Build and attach
steps are in `README.md`; milestones are in `TODO.md`; numbers are in
[`../../docs/measurements.md`](../../docs/measurements.md).

## Goal and framing

A llama.cpp backend that runs `GGML_OP_MUL_MAT` on the AMD XDNA2 NPU using the
fixed Q4_0/Q6_K matmul program. The first target is Gemma 4 E2B: Q4_0
attention/FFN tensors plus the native Q6_K output head.

The point is productive inference without the laptop's fans ramping up. The NPU
is chosen for efficiency, not peak throughput; CPU/iGPU numbers are comparison
baselines. A later phase may split work between the NPU and iGPU, but that is
parked. The backend's per-tensor interface stays compatible with a second
device.

The device design is documented in `src/program`. This backend consumes it and
must not recreate the old bf16 prototype.

## Architecture

1. **Use `spiral::Program` directly.** One `Program` owns the xclbin, hw
   context, kernel, instruction BO, input BO and output BO. The backend must
   not duplicate transaction generation or quant repack.
2. **Store canonical quant weights in the ggml weight buffer.** The Spiral
   weights buffer keeps the original Q4_0/Q6_K bytes. On first use it calls
   `Program::prepare(shape, raw, raw_stride, 0)` and caches the handle. Keep the
   canonical copy because combined calls need to repack again.
3. **One resident BO per prepared handle.** A token runs ~276 matmuls; a shared
   most-recent BO would re-upload nearly every call. `PreparedWeights` owns an
   `xrt::bo` allocated with `kernel.group_id(3)` and `Program::run` uses it
   directly. E2B native weights are ~1.38 GB, so hundreds of resident BOs are
   practical.
4. **Chunk rows by `inputs_vecs = 4`.** Decode is one row plus three zero rows;
   prefill is chunks of four rows. Inputs are bf16 bits in
   `[4][weights_rows]`; outputs are f32 in `[4][weights_cols]`.
5. **Shape contract.** `weights_rows` must be a multiple of 256;
   `weights_cols` must be a multiple of 512. E2B K values (1536/2048/4096/6144/
   12288) satisfy the row constraint. N=256 tensors (`attn_k`, `attn_v`,
   `inp_gate`) need a canonical padded copy to 512 columns, built from the
   `q4_0_*x256p512` golden cases. Reject shapes needing K padding.
6. **ggml device contract.** Device type `IGPU` so `--device Spiral0 -ngl 999`
   selects the NPU; `get_memory()` reports host RAM, not NPU L2; the weights
   buft returns `is_host = false`; the default buffer type stays CPU.
7. **No custom bf16 tile packer.** `Program::prepare` owns canonical repack,
   block-major stream assembly and end padding.
8. **Internal CPU fallback.** The backend claims host-compatible ops in
   `supports_op` and executes consecutive runs through `ggml_backend_cpu` graph
   views. This reduces the Gemma prompt graph to one scheduler split. The
   fallback defaults to one thread because it runs many small graph views
   without a persistent threadpool; `SPIRAL_CPU_THREADS` overrides it.

## Matchers

### Softcap

Recognizes the final-logit chain `MUL_MAT -> SCALE(1/cap) -> TANH ->
SCALE(cap)` and applies it in the bf16 tile epilogue instead of the CPU. The
matcher requires reciprocal scales, zero bias, a single consumer for each
intermediate, and a contiguous F32 final node. The fused result uses the
approximate bf16 vtanh and keeps a `5e-2` verification policy
(`q4_0_1536x512_sc30`). `SPIRAL_NO_SOFTCAP_FUSE=1` disables it.

### Parallel column fusion (gate+up, QKV)

Consecutive supported `MUL_MAT`s with the same input data (pointer and full
layout), the same K and format, and a combined width within one band chunk are
prepared as one `WeightSegment[]` call and run as one wide matmul; up to four
segments. Each segment's output columns are scattered into the matching node.

Do **not** chain across intervening ops. The scheduler's compute-buffer
allocator aliases tensors whose graph-order lifetimes do not overlap; computing
a later matmul early clobbers a gap tensor. For Gemma this blocks Q/K/V fusion
because QK-norm and RoPE sit between the projections. `SPIRAL_NO_PAIR=1`
disables chaining for A/B.

Combined handles duplicate the paired weights in NPU BOs and cost about a second
of first-graph preparation.

### GEGLU and the FFN chain

The matcher recognizes `gate+up -> GLU(GEGLU, split) -> down` with a single
consumer for gate, up and GLU, a contiguous F32 GLU output, and a supported down
matmul. Three execution paths share it:

- **chained TXN (default):** skip gate, up and GLU, and run
  `gate+up -> GEGLU -> down` as one `Program::run_chain` at the down node's
  graph position. The activation never reaches the host and the block pays one
  dispatch instead of two.
- **host-mediated (`SPIRAL_NO_FFN_CHAIN=1`):** run the gate+up pair with the
  GEGLU epilogue at the GLU node, then the down matmul as a separate launch.
- **CPU GLU (`SPIRAL_NO_GEGLU_FUSE=1`):** the original path.

The chain duplicates every operation's weight stream into one BO (~32 MB per
12288 FFN block) and costs ~30 ms of first-graph preparation. The chain is
synchronous and capped at four operations; cross-node look-ahead is not
compatible with it yet. `SPIRAL_GEGLU_CHECK=1` runs the pair without the
epilogue and logs the device-vs-host GELU error.

## Execution model

- Every multi-chunk node uses the `submit`/`finish` pipeline
  (`SPIRAL_PIPELINE_SLOTS`, default 2). The pipeline does not run calls
  concurrently — the array is fully occupied by one call — it removes the
  serial launch bubble.
- Rows are chunked by `kSpiralInputsVecs = 4`, and the NPU mmul is four rows
  wide. A decode batch of up to four sequences therefore shares one weight
  sweep; `llama-batched-bench` npl=1/2/3/4 measured 15.4/30.0/44.3/57.2 t/s
  aggregate with per-slot throughput falling only 7% (see
  `docs/measurements.md`). Splitting the array across calls does not help:
  there is no concurrent execution.
- A graph with zero output rows (for example the no-logits warm-up batch in
  `llama-batched-bench`) still visits the matmul nodes. They are a no-op, not a
  shape error.
- Cross-node look-ahead (`SPIRAL_LOOKAHEAD=1`) submits a later same-input matmul
  before the current result is consumed and defers only its host unpack. It is
  correct but within noise on Gemma; queue ordering costs as much as it saves.
- PM QoS (`SPIRAL_DMA_LATENCY=1`) holds `/dev/cpu_dma_latency` at 0 for the
  Program lifetime. It helps prompt submission and hurts decode, so it is off by
  default and throws if enabled with an inaccessible node.
- The weight BO over-read guard and the cold-timing model live in
  `src/program`; a call costs ~80 us plus bytes/48 GB/s, so the launch floor
  dominates small matmuls.

## Numerics

- Native Q4_0 and Q6_K dequantization is exact over the canonical finite range;
  the Q6_K host repack pre-splits `d` into an exact bf16 hi/lo pair so real
  subnormal head scales are handled.
- Device intermediates are bf16, matching the two-launch path. The softcap and
  GEGLU use the native approximate bf16 vtanh; their tolerances are named
  (`5e-2` of the column scale).
- Chained bf16 intermediates need an oracle and a tolerance per stage; GEGLU
  shows ~5e-3 global relative from vtanh alone.

## Environment variables

| variable | effect |
| :-- | :-- |
| `SPIRAL_PROGRAM_DIR` | source of the device design; defaults to the sibling `src/program` |
| `SPIRAL_PROGRAM_BUILD_DIR` | built artefacts; defaults to its `build-production` |
| `SPIRAL_GOLDEN_DIR` | smoke-test golden cases; defaults to `src/program/build/golden` |
| `SPIRAL_AIE_INCLUDE_DIR` | `mlir_aie/include`; needed for `aie/Runtime/TxnEncoding.h` |
| `SPIRAL_NO_SOFTCAP_FUSE=1` | disable the softcap epilogue match |
| `SPIRAL_NO_FFN_CHAIN=1` | host-mediated GEGLU instead of the chained TXN |
| `SPIRAL_NO_GEGLU_FUSE=1` | restore the CPU GLU |
| `SPIRAL_GEGLU_CHECK=1` | log device-vs-host GELU error |
| `SPIRAL_NO_PAIR=1` | disable parallel-column chaining (A/B) |
| `SPIRAL_CPU_THREADS=N` | fallback thread count (default 1) |
| `SPIRAL_PIPELINE_SLOTS=N` | submit/finish depth (default 2) |
| `SPIRAL_LOOKAHEAD=1` | experimental cross-node look-ahead |
| `SPIRAL_DMA_LATENCY=1` | hold `/dev/cpu_dma_latency` at 0 |
| `SPIRAL_DMA_LATENCY_PATH` | override the latency node |
| `SPIRAL_PROFILE=1` | totals at backend free |
| `SPIRAL_PROFILE_GRAPH=1` | per-graph wall/CPU/NPU/pack/gather |
| `SPIRAL_PROFILE_NODES=1` | one line per `MUL_MAT` |
| `SPIRAL_MATMUL_ONLY=1` | diagnostic: restrict the backend to `MUL_MAT` |

## Layout and seam

```text
integrations/llama.cpp/
  ggml-spiral.cpp       the backend: weights buffer, MUL_MAT path, CPU fallback
  smoke.cpp             standalone registry/weights/Q6_K-head smoke test
  CMakeLists.txt        ggml backend target; defaults to ../../src/program
  llama-hooks.patch     the four llama.cpp-side edits, and nothing else
```

The backend source lives here, not in llama.cpp. Inside a llama.cpp checkout it
is reached through a symlink at `llama.cpp/ggml/src/ggml-spiral` pointing at
`../../../spiral/integrations/llama.cpp`. Keep the checkouts siblings, or use an
absolute path. Never commit the symlink from a llama.cpp checkout.

The hook patch is the complete llama.cpp delta: one option line, one
`ggml_add_backend` line, the registry include plus registration, and the 22-line
`ggml/include/ggml-spiral.h` public declaration. If it stops applying, the seam
moved and the same small edit is needed nearby.
