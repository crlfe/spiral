> Role: orientation. Current as of 2026-11-25. History: `git log -- integrations/llama.cpp/README.md`.

# llama.cpp integration

An experimental ggml backend that runs `GGML_OP_MUL_MAT` on the AMD XDNA2
(AIE2P) NPU through the fixed Spiral program in [`../../src/program`](../../src/program).
It is a work-in-progress prototype developed against a local llama.cpp
checkout. **This is not an official llama.cpp fork**: the backend will not build
until you create the symlink and apply the hook patch.

Matchers, fallback and buffer design: [`DESIGN.md`](DESIGN.md). Milestones and
open questions: [`TODO.md`](TODO.md). Throughput and thermal results:
[`../../docs/measurements.md`](../../docs/measurements.md).

## What works

- Backend registration as `Spiral0`, a Spiral weights buffer type, and a
  synchronous `MUL_MAT` path.
- Native Q4_0 and Q6_K, including the real `token_embd` head over 11 chunks
  and the N=256 -> 512 padded cases.
- One resident prepared weight BO per tensor; combined gate+up as one wide
  call.
- Softcap fused into the bf16 epilogue.
- FFN `gate+up -> GEGLU -> down` as one chained TXN by default.
- Internal CPU fallback for every other op, which reduces the Gemma prompt
  graph to one scheduler split.

Gemma 4 E2B Q4_0 with the native Q6_K head tracks the CPU greedy text and
diverges only late from native-quant rounding (measured match points are in
[`../../docs/measurements.md`](../../docs/measurements.md)). Current best decode
is the chained-TXN path; the backend's CPU fallback is known to miscompute
granular graphs that contain unsupported matmuls, so select it only for models
whose weights it can offload.

## Attach and build

From a llama.cpp checkout with Spiral as a sibling:

```sh
ln -s ../../../spiral/integrations/llama.cpp ggml/src/ggml-spiral
git apply /path/to/spiral/integrations/llama.cpp/llama-hooks.patch
```

Keep the symlink out of version control with the checkout's
`.git/info/exclude`:

```text
ggml/src/ggml-spiral
```

Build the device program first, then the backend:

```sh
source tools/env.sh
python src/program/build.py --production
```

```sh
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

`spiral-smoke` prints the registry (`Spiral0` + CPU), checks the Q4_0/Q6_K
buffer paths, N=256 padding and two resident handles; `--head` adds the full
11-chunk Q6_K head. `SPIRAL_AIE_INCLUDE_DIR` must point at the `mlir_aie/include`
directory containing `aie/Runtime/TxnEncoding.h`; it defaults to the local
mlir-aie install and is a `CACHE PATH` override. `SPIRAL_PROGRAM_DIR` and
`SPIRAL_PROGRAM_BUILD_DIR` default to the sibling `src/program` and its
`build-production`.

## iGPU comparison build

The iGPU is measured from a separate build so the Spiral backend does not
intercept the graph:

```sh
export HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang
export HIP_PATH=/opt/rocm/core-10.0
export HIP_DEVICE_LIB_PATH=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode
cmake -S . -B build-rocm -DGGML_HIP=ON -DGPU_TARGETS=gfx1150 \
      -DGGML_SPIRAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-bench llama-completion -j 12
```

The container needs `/dev/kfd` (234:0) and `/dev/dri/renderD128` (226:128).
Binaries also need `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib`
because they link `libhipblas.so.3` without an rpath.

## Both backends in one binary

```sh
HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang HIP_PATH=/opt/rocm/core-10.0 \
  HIP_DEVICE_LIB_PATH=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode \
  cmake -S . -B build-both -DGGML_HIP=ON -DGGML_SPIRAL=ON \
        -DGPU_TARGETS=gfx1150 -DCMAKE_BUILD_TYPE=Release
cmake --build build-both --target llama-bench llama-completion -j 12
```

`llama-bench --list-devices` then shows `ROCm0` and `Spiral0`. The NPU+iGPU
split is parked because the backend cannot consume ROCm buffers at the split
boundary; details in `DESIGN.md` and `docs/rejected.md`.

## Running

```sh
./build-spiral/bin/llama-completion -m <model.gguf> \
    --device Spiral0 -ngl 999 -no-cnv --ignore-eos --temp 0 -n 64
```

The first graph prepares weights and can take seconds. Use
`SPIRAL_PROFILE=1` / `SPIRAL_PROFILE_GRAPH=1` / `SPIRAL_PROFILE_NODES=1` for a
per-call or per-node profile.
