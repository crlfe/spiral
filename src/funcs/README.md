> Role: orientation. Current as of 2026-11-24. History: `git log -- src/funcs/README.md`.

# `src/funcs`

Format-independent compute kernels extracted from the fixed program. `src/quants`
owns quant formats, transport layout, repack and fused mmul; `src/funcs` owns
vector math that does not depend on a quant format.

```
rmsnorm/            RMS norm group sum, r hi/lo, in-place scale
  layout.h          group/row/chunk geometry (host-compilable)
  rmsnorm.aie.h     pure device functions; no FIFO or config
tests/              standalone device harness and oracles
  common.py         trace, program-size and build helpers
  rmsnorm.aie.cc    test wrapper (input -> output, trace around each pass)
  rmsnorm.py        reference check, cycles and program memory
```

## Test

```sh
source tools/env.sh
python src/funcs/tests/rmsnorm.py
```

The test uses the program's input-group layout: four K-row steps of 256 values,
with four input vectors interleaved per 8-value chunk. K defaults to 1536 (two
groups, the second with two padded row steps). `FUNCS_NO_NPU=1` skips the device
run; `FUNCS_KERNEL_FLAGS` appends Peano flags.

## Current result

`rmsnorm.py` passes its reference at `max abs error 0.0156`, `worst tolerance
ratio 0.464` across seeds 1-5 under the existing `max(1e-3, 6e-3 * row_scale)`
policy. The full-K kernel costs about **5.3k cycles** on one compute tile
(sum 0.77k, scale 3.9k), `.text` 3600 B and a 384 B stack frame. Numbers and
the rows sweep live in `docs/measurements.md` § 3.

That is below the CPU's per-norm cost, but the scale pass is now issue-bound and
the program would still replicate the norm on every compute tile. See
`DESIGN.md` for the layout and the remaining structural options.
