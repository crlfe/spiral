# Program-memory overlays

Can a loaded xclbin swap compute-tile program memory quickly, without a PDI
reconfiguration, while the tiles keep running? And can one host stream do it
for every tile at once?

Current answer: yes. A 12 KiB slot is rewritten in one BD while the core keeps
running, one broadcast stream updates all 32 tiles (verified by CP polls), and a
streamed slot executes; A/B rounds repeat either one dispatch per round or
unrolled into a single dispatch. The production shape (real Q4_0/Q6_K tails at
two fixed slots, a format dispatcher, whole-array switch) is not wired yet; see
[Next](#next).

Three scripts cover the pieces:

| script | phase | proves | status |
|---|---|---|---|
| `overlay.py` | cost baseline | blockwrite reload cost vs PDI; A/B images with two real kernels | keep for now; retire once its variant builder moves |
| `overlay_blocks.py` | link layout | real Q4_0/Q6_K tails, stable `overlay_entry`, identical resident symbols | keep |
| `overlay_packets.py` | transport + switch | one BD carries a whole control-packet stream; slot write while running; 32-tile broadcast; executing a streamed slot | keep |

## Cost baseline

Measured on the Ryzen AI 9 HX 475 (blockwrite and PDI from `overlay.py`;
control packets from the earlier per-tile experiment, which included the
overlay load per configuration):

| path | cost |
|---|---|
| full PDI reload | ~2.6 ms |
| blockwrite, per tile | ~1.3-1.5 us/KiB |
| blockwrite, 32 tiles x ~11 KiB | ~0.45-0.5 ms |
| control packets, per tile (1/4 cores) | 68.8 / 138.6 us vs 50.7 / 96.3 us for blockwrite |

The cost is the command processor's block-write throughput, not host
bandwidth. A blockwrite is one `TXN_OPC_BLOCKWRITE` executed by the CP; its
payload is inline in the instruction buffer and the destination is the flat AIE
address space (`col << 25 | row << 20 | offset` on AIE2P). No switchbox route
is involved, which is why it beats control packets for bulk program memory but
still costs one CP write per word.

`overlay.py` builds two variants that differ only in a compile-time addend,
extracts each core's `.text`, and prepends a disable/reset + blockwrite +
release/reset sequence to the generated transaction. That establishes that
swapping works and what it costs. It resets the cores around the write, so it
is not the "while running" case.

## Link layout

`overlay_blocks.py` is link-only and uses the real production block sources:

```text
program memory (16 KiB)
0x0000  crt0 / worker loop / config / init / zero / block_begin
...     resident main (same bytes in every variant)
0xe0    overlay_entry
...     format block, dispatcher, compiler-rt helpers it pulls in
0x4000  end
```

It checks the property the host loader depends on: the resident object is
identical across variants, so every symbol below `overlay_entry` has the same
address, and the host can replace the bytes from `overlay_entry` on. Current
tails: Q4_0 4848 B, Q6_K 5648 B; with the shared block loop the payload is the
`unpack_mmul` instantiation, 3200 B and 4000 B.

AIE2 calls are absolute (`j #0x3000`), so a second fixed slot needs either a
linker-script region (`AIETargetLdScript.cpp`) or a custom module link at that
base, with a `slot0_entry`/`slot1_entry` pair as the callable bases. The switch
test uses one fixed entry at `0x21000`; a production design would use such a
pair behind a per-call dispatcher.

### Runtime load

The production worker blocks on its config FIFO between calls, which is the
quiescent point to replace its tail: it is executing only resident code then.
The load must happen after the previous call has drained and before the next
config object is released.  Relying on the output drain alone is racy (the core
releases the output before it reaches the loop tail), so the robust sequence
adds a worker-to-host "ready for next config" lock.

`overlay_packets.py --design slot` is the non-destructive form: it writes a
*different* PM region (`0x21000`) while the core runs from `0x20000`, so no
quiescent point is needed.  Replacing the slot the core is executing still
requires the dispatcher and the switch.

### Bundles

There is one tail slot.  To keep several formats resident, link a dispatcher
(`overlay_entry(fmt) { switch (fmt) { ... } }`) and load the bundle once; the
host reloads only when the set of formats changes.  Multiple simultaneous fixed
slots would need a linker-script region per slot (an `AIETargetLdScript.cpp`
change) or the custom module link above.

### Caveats

- Program memory has no hardware A/B banks: "slots" are fixed address ranges
  inside the single 16 KiB region.
- Only `.text` is swapped.  Static data, stack and DM initialization stay
  resident, so variants must agree on their data layout; format-specific LUTs
  or tables need their own region or must remain resident.
- Resetting a core blocked mid-stream on a depth-2 FIFO can leave stale lock
  state.  Prefer the quiescent point, or reinitialize the affected locks and
  DMA as the `--expand-load-pdis` path does.

## Transport: control packets

A control packet's stream header carries the routing id; its 20-bit address is
**tile-relative**. One packet copied by the switchboxes to N `TileControl` ports
performs the same write in every tile's window, which is what makes a broadcast
possible.

`overlay_packets.py` hand-builds the stream:

```text
stream header | control header (stream_id, opcode, beats, address) | 0..4 data
```

Findings, each one a bug that cost time:

- **One BD can carry the whole stream.** A shim BD is one TLAST-delimited
  stream packet and can hold hundreds of inline control packets: 768 packets
  (12 KiB of PM, 18 KiB on the wire) in one BD works.
- **Control headers need odd parity.** The TileControl silently drops a header
  whose 32-bit popcount is even. `0x1F000` (five bits) works with bit 31 clear;
  `0x1F020` (six bits) is ignored unless bit 31 is set. `create_ctrl_pkt` in
  `xrt_test_wrapper.h` and the encoder in `AIETargetNPU.cpp` set it the same
  way.
- **BD ids are per channel and limited to 16.** Shim and core tiles have 16 BDs
  per channel (mem tiles 48). `id = 100` runs past the table and the transfer
  never happens.
- **Reads are expensive.** A read request carries at most four words and must be
  its own BD (upstream `aie-ctrl-packet-to-dma` only batches opcode 0). Two and
  four reads per pass work, eight is flaky, fifteen fails on the first pass.

### Sizing the write: `--packets` and `--batch`

A control packet carries four 32-bit data words (16 bytes); the header's
`beats` field is two bits. On the wire it is six words: stream header, control
header, four data words (24 bytes). Each packet's address advances by 16 bytes,
so `--packets N` writes `16 * N` bytes per destination tile and `24 * N` bytes
cross the wire.

- `broadcast` defaults to 768 packets: 12 KiB per tile, the free PM window
  `0x21000..0x23fff` (the resident `.text` at `0x20000` is untouched).
- `slot` defaults to 14 packets: 224 bytes, the most that fits in the
  per-packet form.
- `switch` takes the count from the compiled slot image (six packets for the
  0x60-byte function).

`--batch` is a `slot` option and changes how the stream is sent, not what it
contains. Without it, `slot` programs one BD per control packet (a BD is one
TLAST-delimited stream packet, and the TileControl consumes one control packet
per stream packet). A shim channel has 16 BDs and the two read requests need
two, so the per-packet form tops out at 14. `--batch` sends the whole stream in
one BD, so `--packets` is limited only by the host buffer and the BD length;
`--verify-all` forces it. `broadcast` and `switch` always use one BD.

## Single tile: write while running, then read back

`--design slot` free-runs a core in slot A (`0x20000`) writing a counter, and
streams a second kernel's `.text` into slot B (`0x21000`) with no reset. By
default the image comes from `overlay.py`'s two-variant builder (`--image
variant`), so it is a genuinely different kernel; `--image resident` repeats the
resident core's `.text`. Control-packet reads return the words; `--verify-all`
reads every 4-word chunk.

```text
12288 bytes into slot B in 304.5 us        # first dispatch
core counter: first snapshot 0x1, second 0x2
verified 768 of 768 chunks (192 passes of 4 reads)   # --verify-all --read-batch 4
```

A 14-packet (224-byte) write costs ~65-82 us after the first dispatch (which
pays the xclbin/overlay setup); 12 KiB in one BD is not measurably more. The counter
advancing across dispatches shows the core never reset.

## Broadcast: one stream, 32 tiles

`--design broadcast` (default) sends one 12 KiB write stream into one packet
flow with 32 `TileControl` destinations, then verifies the result with CP
`maskpoll`s at the start, middle and end of every tile:

```text
all maskpolls matched in 430.6 us
all maskpolls matched in 118.6 us
ok: CP polls verified start/middle/end of 32 tiles
```

No read-back flow, no response channel, no per-tile copy of the image.  For
comparison, 32 x 11 KiB through the blockwrite path is ~0.45-0.5 ms.

## Verifying without a read-back packet

The instruction stream has no data-returning `read32`: `write32`, `blockwrite`
and `maskwrite32` are writes, and `maskpoll` reads a word only to compare it.
`TXN_OPC_READ_REGS` is a legacy opcode in `include/aie/Runtime/TxnEncoding.h`;
nothing emits it, there is no op/test/layout for it, and current aie-rt has
dropped it. Host-side `XAie_Read32` is the host backend, not the CP.

With the expected image known, `maskpoll(addr, value, mask = 0xFFFFFFFF)` makes
the CP read the word and fail the dispatch unless it matches. The installed
September `aiecc` predates `aiex.npu.maskpoll`, so `append_polls()` appends the
txn instruction (`opcode 4, 0, address, 0, value, mask, byte size`) and bumps
the header's opcount/size.

Flipping one data word in the blob (at the end, and separately in the middle)
each turned the broadcast dispatch into a poll timeout, so the poll reads PM
rather than the blob. A failure is a dispatch timeout with no address, so
bisecting the poll set is the only diagnostic; it is a comparator, not a reader.

## Switch: execute the streamed slot

`--design switch` closes the loop, and repeats it. The resident core loops:
run stage A (writes `0xA000 + (round&1)*0x100`), publish it through the output
DMA, block on a hardware lock, then call `0x21000` (stage B) and publish that.
The host does one dispatch per round: stream that round's slot image, release
the lock, drain both buffers.

```text
$ ... --design switch --count 6
  round 0: A 0xa000 B 0xb000  ok  (353.1 us)
  round 1: A 0xa100 B 0xb100  ok  (81.8 us)
  round 2: A 0xa000 B 0xb000  ok  (63.3 us)
  round 3: A 0xa100 B 0xb100  ok  (68.0 us)
  round 4: A 0xa000 B 0xb000  ok  (65.1 us)
  round 5: A 0xa100 B 0xb100  ok  (64.0 us)
ok: 6 rounds of A,B executed with the streamed slot
```

A1,B1,A2,B2,A1,B1... with the slot image rewritten between every round: B1
writes `0xB000..`, B2 writes `0xB100..`. The first round pays the setup; later
rounds are ~50-80 us each with one dispatch per round.

All rounds can also run in **one dispatch**:

```text
$ ... --design switch --count 32 --one-run
  round 31: A 0xa100 B 0xb100  ok
  32 rounds in one dispatch: 404.5 us (12.6 us/round)
ok: one xrt::run executed 32 A,B rounds
```

The runtime sequence is unrolled at build time: both slot images are staged in
the packet buffer (`[B1][B2][lock]`), each round gets its own 8-word drain
offsets in one host buffer, and the CP walks the rounds itself. Per round this
is ~13-50 us instead of ~55-95 us, because the host launch is paid once; the
laptop's clock state moves both numbers, so compare within a run. The trade is
that the round count and both images are compile-time fixed; the per-round mode
is still the one to use when the image is chosen at runtime.

How the pieces fit:

- Each slot image is compiled by Peano and linked with `ld.lld -T` at
  `0x21000` (`.text` is 0x60 bytes, six 16-byte control packets). In the
  per-round mode the host rewrites that code section of the packet buffer each
  round, keeping the lock packet in place; in one-run both sections are staged
  and the sequence selects them by offset.
- The resident calls `((void (*)(int32_t *))0x21000)(out)`. A constant function
  pointer compiles to `movxm p1, #0x21000; jl p1`, so no linker symbol or
  second link step is needed for the call.
- The switch is a **hardware lock**, not a DM flag. A `.bss` flag loses a race
  with the core's startup zeroing (the host's write can be erased before the
  kernel polls it), which made an earlier version intermittently time out. The
  lock is initialized by the PDI and the core blocks on it; the control packet
  releases it after the code stream, so ordering is explicit.
- Both stages are drained to separate host buffers, so stage A is evidence the
  core was running before the switch, not just that B ran.

Negative control: corrupting one word of round 3's streamed code leaves rounds
0-2 and 4-5 passing and fails only round 3 (`B 0xa107` instead of `0xb107`), so
each round executes the image streamed for that round and the loop keeps
running across replacements.

## Limits

- A shim channel has 16 BDs; a full-slot read-back spends one per four words.
- A read carries four words and cannot be batched, so `--verify-all` uses one
  pass per `--read-batch` group.
- A broadcast write plus per-tile read-back does not route: both flows target
  `TileControl`, and a master port belongs to one masterset. Per-tile 1:1 flows
  do route, but the write stream is then duplicated 32x and the NPU DMA window
  on one host buffer is ~128 KiB of used data (2 KiB per tile). `maskpoll`
  verification avoids both the conflict and the cap.
- Polling every word of every tile is 98,304 polls (32 x 3072 words, ~2.7 MB
  of instructions); three points per tile is cheap, full coverage is practical
  for one tile or in slices.

## Running it

```bash
# link layout and tails (no hardware)
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_blocks.py

# transport: one tile, write + read-back
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design slot --batch --packets 768
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design slot --packets 768 --verify-all --read-batch 4

# broadcast: all 32 tiles + polls (default)
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py

# switch: alternate A,B rounds with the slot rewritten each round
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design switch --count 6
# the same rounds unrolled into one dispatch
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design switch --count 32 --one-run

# cost baseline / A/B images; still used by slot --image variant
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py
```

`overlay_packets.py` options: `--design broadcast|slot|switch`, `--packets N`
(4-word packets; 16 bytes each), `--batch` (slot), `--verify-all --read-batch N`
(slot), `--image variant|resident` (slot), `--count N` (broadcast/slot
dispatches, switch rounds; default 1, or 4 for switch), `--one-run` (switch),
`--no-run`. Artifacts land in
`tests/poc/build/overlay_packets/{broadcast,slot,switch}`.

## Rejected experiments

- **Per-tile streams with per-tile read-back**: routed and verified all 32 tiles
  at 2048 bytes per tile, but the 32x blob hit the 128 KiB window and could not
  reach a 12 KiB slot.
- **`aiex.npu.maskpoll` in the MLIR**: the installed `aiecc` predates the op;
  the txn words are appended instead.
- **Broadcast write plus per-column read-back**: the masterset conflict above.

## Next

The switch test runs a hand-written slot function. The natural follow-ups:

- Two **real block images** (Q4_0/Q6_K) at two fixed slots, using
  `overlay_blocks.py`'s link layout, with a dispatcher keyed by format. This is
  the production shape: swap the tail, then call the new format's entry.
- **Broadcast + switch across 32 tiles**: the broadcast write is proven, and a
  per-tile go-lock release is the same mechanism; combine them for a whole-NPU
  format switch.
- Retire `overlay.py`: move its variant builder into `overlay_packets.py` and
  drop the transitional import.
