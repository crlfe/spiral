# llama.cpp integration

An experimental ggml backend that runs `GGML_OP_MUL_MAT` on the AMD XDNA2
(AIE2P) NPU through the fixed Spiral program in [`../../src/program`](../../src/program).
It is a work-in-progress prototype, developed against a local llama.cpp
checkout.  **This is not an official llama.cpp fork** and the backend will not
build until you create the symlink and apply the hook patch below.

Read [`TODO.md`](TODO.md) for the backend's design notes, measurements, build
recipes and rejected experiments; [`../../src/program/NOTES.md`](../../src/program/NOTES.md)
covers the device program itself.

## Softcap fusion

The backend recognizes the final-logit chain
`MUL_MAT -> SCALE(1/cap) -> TANH -> SCALE(cap)` and applies it in the bf16 tile
epilogue instead of the CPU fallback.  The matcher requires reciprocal scales,
zero bias, a single consumer for each intermediate, and a contiguous F32 final
node.  `SPIRAL_NO_SOFTCAP_FUSE=1` disables it for A/B or fallback.  The fused
result uses the approximate bf16 vtanh; `spiral-smoke` includes the
`q4_0_1536x512_sc30` case with the `5e-2` softcap verification policy.

## Layout

```
integrations/llama.cpp/
  ggml-spiral.cpp       the backend: weights buffer, MUL_MAT path, CPU fallback
  smoke.cpp             standalone registry/weights/Q6_K-head smoke test
  CMakeLists.txt        ggml backend target; defaults to ../../src/program
  llama-hooks.patch      the four llama.cpp-side edits, and nothing else
  TODO.md               backend notes
```

The backend source lives here, not in llama.cpp.  Inside a llama.cpp checkout
it is reached through a symlink, so both spellings refer to the same file:

```
llama.cpp/ggml/src/ggml-spiral   ->  ../../../spiral/integrations/llama.cpp
```

Keep the two checkouts as siblings, or replace the link with an absolute path.

## Building

From a llama.cpp checkout, with Spiral next to it:

```bash
ln -s ../../../spiral/integrations/llama.cpp ggml/src/ggml-spiral
git apply /path/to/spiral/integrations/llama.cpp/llama-hooks.patch
```

Then configure with `-DGGML_SPIRAL=ON`.  The device program must have been
built first (`SPIRAL_Q4_VARIANT=q4_0 python src/program/build.py --production`).
`SPIRAL_AIE_INCLUDE_DIR` still defaults to a local mlir-aie install and is a
`CACHE PATH` override for other machines.

To keep the symlink out of version control, add it to the llama.cpp checkout's
`.git/info/exclude`:

```
ggml/src/ggml-spiral
```

## The hook patch

`llama-hooks.patch` is the complete llama.cpp delta — one option line, one
`ggml_add_backend` line, the registry include plus registration, and the
22-line `ggml/include/ggml-spiral.h` public declaration.  Re-verify it when
rebasing; if it stops applying, the seam moved and the backend needs the same
small edit somewhere nearby.

Nothing else in llama.cpp is modified.  Do not commit the symlink from a
llama.cpp checkout: upstream should see the patch, never a dangling link.
