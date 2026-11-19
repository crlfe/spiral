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
- **Packed16 is a format experiment, not a program variant.** The fused
  comparison (exact exponent-add 8328 cycles / 4160 B against packed16 7397 /
  3808 B, near-exact with ~0.78% weight error) was settled in favour of the
  exact path: it improved to 7883 cycles, leaving packed16 a 6.2% kernel gap
  with no frequency win. The program variant was removed; the format stays in
  `src/quants` as an optional representation with its own named verification
  policy. Do not add it back to `src/program` without a new reason.
- **Phase out the `_x8` / `_x16` probes once no consumer needs them.** The
  default probe names are now the 32-wide batches. The narrow variants remain
  for history and for format-shaped cases such as a Q4_0 block's eight column
  scales. Audit consumers after the Q4_0 decision; if nothing needs them,
  remove the narrow probes rather than keeping them as permanent benchmarks.
- **Compare within one boot.** Clocks are not locked: the measurement policy
  deliberately avoids pmode/clock pins except the iGPU SCLK cap. The latest
  two-term run was on a rebooted kernel and its absolute cycles matched the
  earlier run, so cross-boot numbers are usable, but prefer per-value ratios
  measured in one session.
