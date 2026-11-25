> Role: tasks. Current as of 2026-11-24. History: `git log -- src/funcs/TODO.md`.

# `src/funcs` tasks

Open work only. Design and layout are in [`DESIGN.md`](DESIGN.md), commands in
[`README.md`](README.md).

## RMS norm

- [x] Standalone group layout and pure vector functions.
- [x] Device test with full K=1536, partial final group, per-row tolerance.
- [x] Native 32-lane scale and a single 32-lane `accfloat` partial in the sum.
- [x] Profile the sum and scale separately: now sum 765, scale 3904 cycles
      (was 1.1k / 7.5k). Two-chunk interleave, a 512-bit two-chunk weight load,
      one-instruction 128-bit lane broadcasts and four f32 sum partials did it.

### Program integration contract

The program prelude acquires **one 4096-value input group at a time**; the
current full-K functions take a contiguous `input` base plus a global `group`,
which cannot be called on a single acquired group without invalid pointer
arithmetic. The program side already has the two-static-body host selection
working with a dummy extra fill.

- [x] Add one-group/base-row entry points:
      `sum_squares_group_at(group_input, base_row, rows_steps, sums)` and
      `scale_group_at(group_input, base_row, rows_steps, weight, r_hi, r_lo)`.
      `base_row` selects the weight slice; `group_input` is exactly one group.
- [x] Keep `sum_squares_group`, `scale_group` and `normalize` as full-K
      wrappers over the `*_at` functions so the standalone test and callers that
      already hold the whole activation stay unchanged.
- [ ] Add a test that drives groups one at a time from separately allocated
      buffers: sum group 0, sum group 1, `finish_r`, scale group 0, scale group
      1. The current test drives the `*_at` entry points per group with
      per-group pointers into one packed buffer; a true two-buffer version
      still needs a group-object FIFO test.
- [x] Keep the weight contract full-K, interleaved per 8-value chunk as
      `[chunk][hi8][lo8]` (2 * K bf16, 6144 B at K=1536). The program config
      will carry that layout; do not switch to `hi[K]` then `lo[K]`.
- [ ] Report the one-group functions' linked `.text` and stack, not only the
      test wrapper's. The program has ~3.5 KiB of headroom before the wrapper
      and state at the current 12480 B production image.

- [x] Optimize the scale pass without dropping the hi/lo precision: 5315 total
      cycles (~3.8 us on one tile) versus 9197 before, same tolerance. The
      remaining gate is structural, not a loop rewrite; see `DESIGN.md` Cost.
- [ ] Try a dataflow where one compute/group produces `r` and broadcasts
      it, so the reduction is not replicated 32x.
- [ ] Alternatively test host computation of `r` with a device-only
      scale; measure CPU removal against device scale cost.
- [ ] Decide on folding `gamma` into the quantised weights on the host
      (`(x*g)W = x*(gW)`), which would leave the device applying only the
      per-row scalar `r`; needs a final-matmul numeric sweep.
- [ ] Reject or accept the per-compute prelude with measured end-to-end
      numbers before wiring the matcher or the full norm prelude.
- [ ] Add a one-group (K=256) case and a non-zero `eps`/weight edge case.
- [ ] Make `tests/common.py` locate the core ELF reliably when the trace path
      has no physical MLIR file (program memory currently prints unavailable
      on some runs).

## Later functions

- [ ] RoPE for Q/K, after the norm link.
- [ ] Softmax / attention pieces, only with a joint dataflow plan.
- [ ] Move the existing `src/program` epilogue GEGLU/softcap math here once the
      program's interface is stable; do not destabilize the verified path for
      the norm work.

## Ownership

Keep FIFO acquire/release, block assembly/carry, config/state and worker
dispatch in `src/program/kernels`. `src/funcs` contains only pure vector math
with its own oracle and cycle harness. `src/quants` remains the quant format,
repack, transport and fused-mmul tree.
