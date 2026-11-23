# TXN concatenation and chained matmuls

This prototype builds two fixed-program matmuls into one TXN, retargets the
second matmul's input patches to the first matmul's output buffer, and compares
the chained result against the current two-launch path.  The aligned Q4_0 pair
and the square Q6_K pair are also repeated eight times to show the dispatch
floor being amortised.

This document is the durable record for the work.  It absorbs the still
applicable facts from the earlier scratch design notes (now removed) and
updates them for the current design: the four-entry runtime argument order, the
native output layout, the separate config BO, and the bf16 epilogue.

## Result

Every identity case is **bit-identical** to the independent two-launch path;
the GEGLU case differs only by the bf16 vtanh approximation (see item 6).

| case | chain | independent | speedup |
| :-- | --: | --: | --: |
| `q4_0_1536x2048 -> q4_0_2048x1536` (2 ops) | 156.5 us | 217.1 us | 1.39x |
| same pair, 8 ops | 445.9 us (55.7 us/op) | 866.8 us (108.4 us/op) | 1.94x |
| `q4_0_12288x1536 -> q4_0_1536x512` (2 ops) | 296.1 us | 350.4 us | 1.18x |
| `q6_k_2048x2048 -> q6_k_2048x2048` (2 ops) | 222.0 us | 290.3 us | 1.31x |
| same square pair, 8 ops | 695.7 us (87.0 us/op) | 1095.1 us (136.9 us/op) | 1.57x |
| `q6_k_512x1536 -> q6_k_1536x2048` (2 ops) | 144.7 us | 193.8 us | 1.34x |
| `q4_0_1536x2048 -> q6_k_2048x2048` (mixed, 2 ops) | 193.4 us | 260.1 us | 1.34x |
| `q4_0_1536x12288` gate+up (GEGLU) -> `q4_0_6144x1536` | 385.8 us | 454.7 us | 1.18x |

Host context: Tctl 43.5-49.2 C over the run, chassis fan 0 RPM, `amdgpu`
edge 41.0-43.0 C.  These are medians over 20 iterations from a representative
run; 2-op timings vary a few percent between runs.  They are not a thermal
soak.  The timings include the instruction-BO upload and output sync; TXN
assembly is prebuilt and excluded.

The Q4_0 8-op pair alternates the two shapes, so op1's output width (1536)
equals op0's input width.  The Q6_K square pair repeats one shape.  Both reuse
the same weight/config objects in one command.  The per-op time is higher than
the earlier 1536x512 figure because these shapes are larger; the point is that
the floor is paid once.

## What changed from the earlier prototype

1. **Runtime argument order is now four entries.**
   The earlier notes describe `arg 0 = weights, arg 1 = inputs, arg 2 =
   outputs`.  The separate config BO changed that to:

   ```text
   TXN arg 0 = weights   -> XRT kernel arg 3
   TXN arg 1 = inputs    -> XRT kernel arg 4
   TXN arg 2 = config    -> XRT kernel arg 5
   TXN arg 3 = outputs   -> XRT kernel arg 6
   ```

   The current body has 973 words total and 969 words of body.  The DDR patches
   are:

   ```text
   config:  arg 2, plus 0
   inputs:  arg 1, plus 0, 8192, 16384, 24576 bytes
   weights: arg 0, 16 patches (8 groups x 2 parts)
   outputs: arg 3, 8 patches at plus 0, 512, ..., 3584 bytes
   ```

   For a chain, retarget input patches from arg 1 to **arg 3**, not arg 2.
   The config patch is the arg-2 patch, not the first arg-1 patch.

2. **The native output layout fixes the missing pack stage for identity
   chains.**
   The earlier chain fed an output object laid out as
   `[group][compute][band][vec][16]` into an input expecting packed
   `[row][vec][8]`, so the second matmul read the wrong layout.

   The production drain now writes plain `[N/8][vec][8]` blocks.  The eight
   group drains use bases `0, 512, ..., 3584` bytes plus the 3-D tap, which
   together cover the packed layout from offset 0.  The next matmul's input
   patches read `0, 8192, ...` from that same buffer.  The two layouts match
   exactly, which is why the chain is bit-identical here.

3. **The config-in-input-buffer caveat is obsolete.**
   Config has its own BO.  Use one 8 KiB config object per operation at
   `k * 8192`; do not shift output patches by 8192.  If config is ever moved
   back to the front of the output BO, shift every output patch by 8192 and
   grow the BO by 8192, or the last group's drain runs off the end and the
   command times out.

4. **The worker already processes multiple config epochs.**
   The lowered MLIR wraps each core in a persistent outer loop that acquires
   an input object, calls `program_begin`, runs the per-op loop, and branches
   back.  A concatenated TXN with one config + input sequence per operation is
   therefore compatible with the current worker.

5. **bf16 output is already in the common epilogue.**
   The earlier `PROGRAM_OUTPUT_BF16` block-kernel changes are not needed.  The
   production `program_output` converts the f32 accumulator to bf16 after the
   optional softcap.  Do not reapply the per-record bf16 block changes.

6. **The GEGLU activation is now an epilogue kind.**
   Config word 10 is an exclusive kind: 0 = none, 1 = softcap, 2 = GEGLU.
   For a two-segment gate+up call GEGLU pairs band `b` with
   `b + cols_bands/2` in the same compute, computes `gelu(gate) * up` in bf16
   with the native bf16 `vtanh`, and stores only the gate bands.  The drain
   still writes `cols_bands` bands, so the packed gate half is the down
   matmul's input at offset 0.  No pack kernel is involved and the chain patch
   policy is unchanged: op0 uses kind 2, op1 uses `input_from_output`.

   bf16 matters here: AIE2P emulates f32 vector arithmetic with multiple bf16
   products and conversions.  The f32-operand version had a 1152 B loop body
   and cost a few microseconds; the bf16 version is 288 B, has no stack frame,
   and the epilogue delta is below the run noise.  The measured error is
   unchanged because the output and the two-launch reference are bf16 anyway.

   Measured (`q4_0_1536x12288` gate+up -> `q4_0_6144x1536`, 40 iterations,
   plain/GEGLU interleaved):

   | item | value |
   | :-- | --: |
   | epilogue delta (none vs GEGLU) | < 1 us (noise) |
   | chained vs host-GELU two-launch | 385.8 vs 454.7 us (1.18x) |
   | GEGLU intermediate vs host f32 | 5.3e-3 global relative |
   | final down output vs two-launch | 5.5e-3 global relative |

   The errors are bf16-level and dominated by the vtanh approximation.  The
   prototype now runs this as the `geglu` case (`./txn_concat ... geglu`), and
   the production `Program` path as the `program` case: `prepare(segments)` with
   two equal halves, `MatmulShape::epilogue = kEpilogueGeglu`, and a half-width
   output check/unpack.  Both produce the same device result
   (12,116 bit mismatches, 5.3e-3 relative).

7. **The production chain API is exercised.**  The `chain` case builds one
   `Program::prepare_chain` from the gate+up (GEGLU) and down prepared handles,
   runs it with `run_chain`, and compares it with two separate launches that
   use the same device GEGLU intermediate.  The result is bit-identical.  On
   this 6144 FFN the chain is 401-437 us against 447-458 us for two launches;
   the remaining difference is dispatch and host hand-off.  The chain copies
   both weight streams into one BO, so the per-op handles only need to be alive
   during preparation.

## Why concatenate

The ~44-50 us on every NPU call is **dispatch latency**, not this design's
fill/drain.  It is a latency, not a throughput limit.

| mode | XRT | HRX (C++) |
| :-- | --: | --: |
| one command, serial | 44.3 us | 50.3 us |
| 2 commands in flight | 17.1 us/call | - |
| 4 commands in flight | 15.2 us/call | - |
| N operations, one TXN | one dispatch | - |

Three different batching mechanisms must not be confused:

- **XRT concat:** N operations inside one TXN, one command.
- **HRX `dispatch_chain`:** N separate dispatches with an execution and memory
  barrier between each, i.e. N commands.
- **XRT pipeline:** N commands in flight on separate `xrt::run` objects.

Only the first removes the per-operation floor.  That is why the prototype
builds one concatenated TXN rather than chaining host launches.  The earlier
HRX `dispatch_chain` probe ran 64 operations in 772 us (N commands, not one
TXN).

Measured with the earlier production body:

| shape (rows x cols) | format | separate | concat steady | speedup |
| :-- | :-- | --: | --: | --: |
| 1536x512 | Q4_0 | 69 us | 22 us | 3.1x |
| 1536x2048 | Q4_0 | 102 us | 47 us | 2.2x |
| 1536x12288 | Q4_0 | 268 us | 218 us | 1.23x |
| 1536x2048 | Q6_K | 119 us | 65 us | 1.8x |

Model: `per-copy = steady + bubble / N`, bubble ~50 us, independent of shape.

The earlier chain prototype (old output layout) measured 45.3 us/op for two
1536x512 ops, 33.4 us/op for four, and 26.5 us/op for eight.  The values were
wrong because of the layout mismatch, but the timings show the floor being
amortised.  The current native layout is bit-exact; the larger shapes in this
prototype show 55.7 us/op for Q4_0 and 87.0 us/op for Q6_K over eight ops.

## TXN format reference

The compiler emits a flat `uint32_t` transaction.  The current production base
TXNs are 973 words (3,892 bytes); the body is 969 words.  The body size is
constant across shapes and formats; only the runtime scalars and the config
object differ.

**Header (4 words):**

```text
word0 = (numRows << 24) | (devGen << 16) | (minor << 8) | major
word1 = (numMemTileRows << 8) | numCols
word2 = op_count
word3 = total byte length
```

For the fixed NPU2 design the header is `[0x06040100, 0x00000108, op_count,
byte_length]` (devGen 4, 6 rows, 8 columns, 1 memtile row).  Concatenation
sums the `op_count`s and fixes the byte length.

**Operation sizes (words):**

- write32: 6 - `[0, 0, addr, 0, value, 24]`
- blockwrite: 4 + payload - `[1, col | row << 8, addr, (4+n)*4, data...]`
- maskwrite32 / maskpoll32: 7 - `[op, 0, addr, 0, value, mask, 28]`
- sync / TCT: 4 - `[128, 16, dir | row << 8 | col << 16,
  nrow << 8 | ncol << 16 | chan << 24]`
- loadpdi: 4 - `[8 | id << 16, size, addr_lo, addr_hi]`
- preempt: 1 - `[6 | level << 8]`
- DDR patch (custom op 129): 12 words -
  `[129, 48, 0, 0, 0, 0, reg_addr, 0, arg_idx, 0, arg_plus_lo, arg_plus_hi]`

The DDR patch is the one an implementation edits.  `reg_addr` is the BD
register, `arg_idx` selects the bound buffer, and `arg_plus` is a **byte**
offset added to that buffer's address.  A reliable way to find patches is the
`129, 48` opcode/size pair; fields are at +6 (`reg_addr`), +8 (`arg_idx`), and
+10/+11 (`arg_plus`).  Prefer walking operations by opcode/size over scanning,
because a blockwrite payload could contain that pair by accident.

**Per-operation body:** generated by
`generate_txn_main_sequence(input_part0..3, weight_part0..1,
group_stride_words)`.  Inputs:

```text
rows_steps      = weights_rows / 256
cols_bands      = weights_cols / 512
total_tiles     = rows_steps * cols_bands
tile_bytes      = 2304 (Q4_0) or 3392 (Q6_K)
weights_blocks  = ceil(total_tiles * tile_bytes / 8192), at most 128
input_groups    = ceil(rows_steps / 4)
input_parts     = split_parts(input_groups, 4)
weight_parts    = split_parts(weights_blocks, 2)
group_stride    = weights_blocks * 8192          (words)
input_pad       = padded_input_groups - input_groups
weight_pad      = padded_weight_blocks - weights_blocks
```

`split_parts(total, parts)` pads to `max(total, parts)` and distributes the
remainder:

```text
padded = max(total, parts)
base   = padded / parts
extra  = padded % parts
part_i = base + (i < extra ? 1 : 0)
```

The generated body contains the weight/input/output DDR patches, shim queue
pushes, and TCT waits.  It does not contain the 8 KiB config object.

**Config object (one 8 KiB input-group-sized object per operation):**

```text
[0] format (2 = Q4_0, 14 = Q6_K)
[1] tile_bytes
[2] rows_steps
[3] cols_bands
[4] weights_blocks
[5] total_tiles
[6] input_pad
[7] weight_pad
[8] softcap hi/lo: two bf16 bits (cap_hi | cap_lo << 16); 0 disables
[9] softcap reciprocal hi/lo: two bf16 bits
```

The worker's `program_begin` copies the config into per-compute state.  One
config object per operation is what lets one TXN mix shapes and formats.

## Patch recipe used here

The prototype keeps `generate_txn_main_sequence` as the source of truth for
BD IDs, queue assignment, and syncs.  `txn_concat::concat` copies bodies after
the 4-word header and walks each operation by opcode/size:

| field | change |
| :-- | :-- |
| header word 2 | sum of the base `op_count`s |
| header word 3 | total words x 4 |
| arg 2 (config) | `arg_plus += config_offset` |
| arg 0 (weights) | `arg_plus += weight_offset` |
| arg 1 (inputs), op 0 | unchanged |
| arg 1 (inputs), op k>0 | `arg_idx = 3` (outputs) |
| arg 3 (outputs) | `arg_plus += output_offset` |

The result is one command.  The first op reads the host-filled input BO and
writes the output BO; every later op reads and writes the output BO in place.
The input fill of op k+1 is ordered after op k's output drain by the generated
TCT waits, so the overwrite is safe.  Do not move the data patches ahead of
those waits.

**Output buffer sizing.**  The bf16 output object is
`8 * GROUP_OUTPUTS * 2` = 196,608 bytes.  With the native layout it is also
the next matmul's packed input layout.  In-place chaining uses one output BO
for both roles.  The first op may still read the host input BO.

**Weights guard.**  The device prefetches the next L2 block while streaming,
including past the final block of the last group.  Leave one zeroed
`kL2StreamBytes` guard block after the combined weight stream; without it the
over-read raises AMD-Vi IO_PAGE_FAULTs.  In a combined per-op weight BO, only
the end of the last stream needs the guard; an over-read into the next op's
stream is valid data.

## Weight, config, and instruction staging

- **Config:** one 8 KiB object per operation at `k * 8192`, uploaded once.
  The TXN config patch adds `k * 8192` to the generated `arg_plus`.
- **Weights:** one combined weights BO.  Give each operation a byte offset,
  add it to every arg-0 patch, and place a guard after the last stream.  The
  existing compact per-handle stream layout can be copied in at its offset.
- **Instruction BO:** size it to the largest chain TXN.  A ~286-operation
  token TXN with the current 973-word body is ~278k words / ~1.11 MB (the
  earlier simulation used 273,992 words / 1.10 MB).  An earlier stress test
  reached about 7.85 MB with no firmware cap, but per-copy cost regressed from
  ~22 to ~31 us above ~1 MB because the host instruction upload runs at
  ~4-5 GB/s.  The 1.10 MB token stream is in the good region; its upload is
  ~0.2 ms once per token.
- **Pipelining:** prebuild the token TXN once, upload ahead, and double-buffer
  across tokens.  Keep 2-4 commands in flight when the stream is split; every
  host/CPU boundary adds another dispatch floor.

## Token target

An earlier zero-data simulation of one command with the approximate Gemma 4
E2B decode stream gives the scale:

```text
ops=286  words=273992  bytes=1095968
avg 29,719 us  min 29,274  median 29,797  max 29,968  (103.9 us/op, 7 reps)
```

| | per token |
| :-- | --: |
| current per-call path | 55.8 ms wall / 52.0 ms kernel |
| one concatenated TXN | 29.7 ms |
| memory-bound estimate (1.381 GB at 55 GB/s) | 25.1 ms |

That is ~1.88x faster wall, ~1.75x on kernel time, ~4.6 ms above the memory
bound.  It excludes CPU glue, sampling, KV management, and the ~0.2 ms
instruction upload, and it used zero data, so it is a ceiling estimate rather
than an end-to-end result.

Approximate operation mix (K x N, count):

| group | K x N | count |
| :-- | :-- | --: |
| Q sliding / full | 1536x2048 / 1536x4096 | 28 / 7 |
| K + V | 1536x512 | 30 |
| O sliding / full | 2048x1536 / 4096x1536 | 28 / 7 |
| inp_gate | 1536x512 (256 padded) | 35 |
| proj | 256x1536 | 35 |
| FFN gate+up | 1536x6144 / 1536x12288 | 30 / 40 |
| FFN down | 6144x1536 / 12288x1536 | 15 / 20 |
| Q6_K head chunks | 1536x24576 | 11 |

## Rejected alternatives

**Host-side wakeup of a running control stream.**  Do not retry blindly.  A
running control stream cannot be woken by a host BO write in a non-root setup:

- a host write to a mapped/device-only BO is just memory; it sets no lock, TCT,
  or queue state, and device-only BOs are not host-mapped;
- `aiex.npu.maskpoll` resolves a 32-bit AIE-array address, so it cannot name a
  DDR BO in the 64-bit `host_addr + 0x80000000` aperture;
- `write_aie_reg/mem` needs a root-only ioctl, and the installed driver returns
  `EOPNOTSUPP` for the AIE-write parameters;
- a free-running shim DMA polling a host flag is fragile: shim DMA has no reset
  bit, a self-loop cannot be stopped, and repeat is bounded.

Conclusion: a host-timed stall needs a new dispatch.  Reduce or overlap
dispatches instead.

**HRX as the first implementation.**  The technique ports, but single-dispatch
HRX is slightly slower (50.3 vs 44.3 us), and executable creation costs
~0.43-0.61 ms per TXN unless the bytes are reused.  `dispatch_chain` is N
commands with barriers, not one TXN.  HRX remains a packaging alternative, not
the first path.  XRT concat matches HRX steady state at larger N:

| N | XRT total / op | HRX total / op |
| --: | --: | --: |
| 8 | 223.4 / 27.9 us | 257.8 / 32.2 us |
| 16 | 391.7 / 24.5 us | 401.1 / 25.1 us |
| 32 | 729.0 / 22.8 us | 734.7 / 23.0 us |

## Correctness and tolerance

The earlier f32 policy used a per-column tolerance
`max(1e-3, 2e-5 * column_scale)`: fp32 accumulation error grows with K, is
unbiased round-to-nearest, and is about 100x better than truncation.  The
current bf16 epilogue uses `6e-3` in the goldens because the bf16 rounding
dominates the fp32 accumulation error.  Do not apply the f32 `2e-5` policy to
bf16 output.

A chained stream needs a per-stage oracle in addition to the final check: the
intermediate activation is bf16-rounded and the next matmul's column scales
differ.  This prototype uses the strongest available check for identity
chains: the chained output must match the independent two-launch path
bit-for-bit.  The GEGLU case cannot be bit-exact because the device uses the
bf16 vtanh; it reports the global relative error against the host f32 GELU
(5.3e-3 intermediate, 5.5e-3 final) and fails above 2e-2 / 1e-2.

## Full implementation checklist

1. **Pack / activation glue.**  The identity-chain layout is solved by the
   native drain, and GEGLU is now an epilogue mode (see item 6 above).  What
   remains for the full decode path: RMSNorm (the hidden-width reduction is
   replicated per compute, so no cross-group barrier is needed), RoPE, and the
   KV append/attention link.  Decide the intermediate contract for each (bf16
   is the short path) and add a tolerance before writing the kernel.
2. **TXN generation and addressing.**  Use one arg slot per distinct buffer or
   per-op `arg_plus` offsets into larger BOs.  Keep per-op config objects.
   Bind one weights BO and select each operation's stream by offset.  Decide
   granularity: whole token (one TXN), per dependency group, or per layer.
   The 29.7 ms assumes one TXN; every host/CPU boundary adds a dispatch floor.
3. **Host sequencing and pipelining.**  Prebuild the token TXN once per token,
   or prebuild the static shape sequence and patch only the buffer addresses.
   Upload the instruction and config BOs ahead of time; double-buffer across
   tokens.  Keep 2-4 dispatches in flight when the stream is split.
4. **Backend integration.**  `integrations/llama.cpp` currently issues one
   synchronous `MUL_MAT` per op.  Use either one token TXN per decode step
   (requires the glue on the NPU) or dependency-group TXNs between CPU ops.
   Keep weight streams and the KV cache resident; reuse the instruction BO.
   Measure with the thermal/fan context recorded, not just device time.
5. **Open questions.**  Exact intermediate contract; whether the pack can run
   in the same compute tile between matmuls or needs a memtile/extra DMA
   stage; whether per-op config adds measurable control overhead; HRX's
   ~0.5 ms create cost; instruction size if the stream grows beyond ~1 MB.

## Files

- `txn_concat.h` - header check, op walker, patch policy, concatenation.
- `txn_concat.cc` - golden loading, weight repack, low-level XRT runner,
  independent vs chained comparison, the GEGLU chain case, and the production
  `Program` GEGLU case.
- `run.py` - compiles the prototype against `src/program/build-production`
  (including `host/program.cc` for the production case) and runs the cases.

## Running

Build the production design first, then run the prototype:

```bash
~/Projects/mlir-aie/ironenv/bin/python src/program/build.py --production
~/Projects/mlir-aie/ironenv/bin/python tests/poc/txn_concat/run.py
```

The prototype does not rebuild the xclbin or regenerate `generate_txn.h`.
It reuses `aie.xclbin`, `design_meta.h`, `generate_txn.h`, and `golden/` from
`src/program/build-production`.

## Limits and next steps

- Q4_0 and Q6_K are both covered.  The TXN body is format-independent; only
  the config and tile bytes differ.
- The chain is an identity matmul chain, plus the GEGLU activation link.
  RMSNorm, RoPE, softmax, and per-token activation quantisation still need
  stages; the hidden-width RMSNorm can be replicated per compute, while
  attention needs a real KV-cache/softmax link.
- The repeated chain reuses the same two weight offsets and config objects.
  A real token TXN needs one weight offset and one config object per
  operation, sized for the whole stream.
- The next measurable step is the token-shaped stream: ~286 operations in one
  TXN, combined weights/config BOs, instruction upload amortised across tokens.
- If the operation walker meets a new opcode, add its size to
  `txn_concat::op_words`; do not fall back to scanning for `129, 48`, since a
  blockwrite payload could contain that pair.
- The host packing helpers (`fill_weights`, `fill_inputs`, `unpack_outputs`)
  mirror `src/program/host/program.cc` for the prototype.  A production chain
  path should share one copy with the wrapper.

## Provenance

The dispatch-floor, E0 concatenation, token-simulation, HRX, and old chain
timings came from earlier discarded probes.  They are recorded here as design
evidence, not as runnable tests.  The Q4_0/Q6_K/mixed chain and the current
timings are reproduced by this directory.

## References

- `include/aie/Runtime/TxnEncoding.h` in mlir-aie: the TXN encoding.
- `src/program/build-production/generate_txn.h`: the current per-shape base
  body.
- `src/program/NOTES.md`: fixed-program data flow, config words, cost model,
  and the efficiency acceptance procedure.
- `tests/poc/invoke.py`: the dispatch-floor probe.
- `src/quants/NOTES.md`: quant kernel and loop notes.
- HRX release: `utils/fetch-hrx-release.sh` in mlir-aie.
