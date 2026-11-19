# Bench TODO

## Low priority

- **`MEMORY_STALL` trace event.** The default core trace already records
  `CoreEvent.MEMORY_STALL` (AIE2P value 23), but it has not fired for any
  current probe: u16/bf16 load-store, f32 conversion, packed16, int8 scale, or
  Q6_K. The event is distinct from the memory-tile `CONFLICT_DM_BANK_*` events.
  It should be the right signal for a same-bank load/store stall, but it is not
  useful for our present tests. Revisit only if a kernel shows an unexplained
  cycle gap that looks like bank contention. The experiment would be a matched
  pair of same-bank and separate-bank loads with identical instruction counts,
  parsing `MEMORY_STALL` intervals alongside `event0()` / `event1()`.

## Next

- **Extend the 32-lane Q6_K preparation into a full application demo.** The
  preparation is now implemented and measured at 107.3 cycles per 32 terms
  (~26.8 per group). The existing `q6k_3term_apply_x8` / `q6k_3term_full_x8`
  still apply one eight-column group at a time. The next step is a full
  4-subgroup apply that consumes the 32-lane terms, if the Q4_0/Q6_K work needs
  it.
- **Decide between the Q4_0 scale paths using the fused path only.** Exact
  exponent-add: fused 8328, 4160 B program. Packed16: fused 7397, 3808 B
  program, but near-exact (~0.78% weight error, ~3.9% worst-case mmul error).
  Standalone unpack numbers are diagnostic and no longer part of the decision.
  Packed16 is close to `q4_0_hilo` at 288 bytes; it has a named 5%-of-column
  mmul verification policy but still needs the Gemma model check before it can
  replace the exact path.
- **Phase out the `_x8` / `_x16` probes once no consumer needs them.** The
  default probe names are now the 32-wide batches. The narrow variants remain
  for history and for format-shaped cases such as a Q4_0 block's eight column
  scales. Audit consumers after the Q4_0 decision; if nothing needs them,
  remove the narrow probes rather than keeping them as permanent benchmarks.
- **Keep clocks locked for cross-boot comparisons.** The latest two-term run
  was on a rebooted kernel with unlocked clocks. Absolute cycles were stable
  against the earlier run, but future comparisons should still be made within
  one boot or with locked clocks.
