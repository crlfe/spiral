# Block overlays

This note explores making the format-specific dequant+mmul blocks in
`src/program/kernels` swappable at runtime without reloading the xclbin or the
rest of the worker.  It is a `tests/poc` experiment; nothing in `src/program`
depends on it today.

The working link-only prototype is `tests/poc/overlay_blocks.py`. It uses the
real `program_q4_block.aie.cc` and `program_q6_block.aie.cc` sources and prints
the stable symbols and overlay sizes. The runtime block-write side is
`tests/poc/overlay.py`.

## Shape

Keep one resident worker image and one tail overlay region:

```text
program memory (16 KiB)
0x0000  crt0 / worker loop / config / init / zero / block_begin
...     resident main (same bytes for every format)
X       overlay_entry
...     format block(s), dispatcher, any compiler-rt helpers they pull in
0x4000  end
```

The resident worker calls one external symbol, `overlay_entry`, with the
existing block signature. Only one overlay object is linked at a time, so the
worker no longer references `program_q4_block` and `program_q6_block`
directly. The host decides which overlay is present.

The overlay object is linked **last**. Because the resident object is
identical across variants, the linker gives every resident symbol and
`overlay_entry` the same address in each variant. The host can therefore
replace the bytes starting at `overlay_entry`; no resident code or relocation
changes.

The PoC confirms this with the real block sources:

```text
q4: core=0x20 overlay=0xe0 (4848 B) text=5072 B
q6: core=0x20 overlay=0xe0 (5648 B) text=5872 B

resident __start       0x0
resident core_0_2      0x20
overlay_entry          0xe0
```

The 4848/5648 B tails include the block wrapper, its `unpack_mmul`, and the
compiler-rt helpers that were not already resident in the tiny PoC main. In
the production worker those helpers are mostly resident.  With the shared
block loop, the slot payload is exactly the `unpack_mmul` instantiation:
Q4_0 3200 B, Q6_K 4000 B.

## Instruction reload cost

`tests/poc/overlay.py` measures whether a loaded xclbin can swap a compute
tile's program-memory image without paying the full PDI reconfiguration cost.
It builds two kernels that differ only in a compile-time addend, extracts each
core's `.text` from the per-core ELF that `aiecc` emits, and prepends a small
instruction sequence to the normal `aie-npu-to-cpp` transaction:

```text
write32 core_control = 0        # disable
write32 core_control = 2        # reset, PC = 0
blockwrite 0x20000 <- overlay   # per-tile program memory
write32 core_control = 0        # release reset
write32 core_control = 1        # enable
```

The runtime sequence generated for the xclbin then runs unchanged.  This avoids
a new xclbin, hw_context, or PDI load; only the 16 KiB program-memory window at
tile offset `0x20000` is rewritten.

```bash
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py --tiles 1 --pad 380
~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py --tiles 4 --pad 0
```

The first run uses a production-sized ~10.5 KiB `.text`; the second uses a
~0.5 KiB image.  On the Ryzen AI 9 HX 475 laptop:

```text
tiles  pad  .text/tile  total overlay  reload
    1  380     10672 B       10672 B   62-65 us
    1    0       496 B         496 B   49-53 us
    4  380     10672 B       42688 B   107 us
    4    0       496 B        1984 B   50 us
    6  380     10672 B       64032 B   131 us
```

The marginal cost is ~1.3-1.5 us per KiB of program memory, so reloading every
one of the 32 production computes (~11 KiB each, ~340 KiB total) should cost
roughly 0.45-0.5 ms, still well below the ~2.6 ms PDI reconfiguration.  The
cost is control-processor block-write throughput, not host bandwidth.

The prototype's single split/join block supports at most six compute columns
because one mem tile has six DMA channels per direction.  The production design
uses one L2 block per group of four computes, so it does not have this limit
and can be scaled to all eight columns / 32 computes.

## Control-packet reload (`--load-pdi-to-ctrl-pkt`)

Recent `mlir-aie` can lower a `load_pdi` into an `@ctrl_pkt_overlay` load
followed by a stream of `aiex.control_packet` ops instead of `write32` /
`blockwrite` ops.  The stream does cover program memory: for a tile at
`(0, 2)`, the generated packets include address `0x220000` (`0x20000` in the
tile window), followed by the compiled instruction words.

I benchmarked it against `--expand-load-pdis` (the write32/blockwrite path) and
plain full-PDI reloads on two configurations with a 10.7 KiB `.text` per core.
The full-ELF runtime sequence configures and runs each configuration once per
dispatch, so the figures below are per configuration including its run:

```text
cores  path       per reconfig
    1  PDI              82.1 us
    1  write32          50.7 us
    1  ctrl-pkt         68.8 us
    4  write32          96.3 us
    4  ctrl-pkt        138.6 us
```

The control-packet path is not faster for bulk program memory.  Legalization
splits each block write into 16-byte control packets, and the per-packet
TileControl-port processing dominates: the 4-core, two-configuration stream is
~136 KiB of control packets for ~85 KiB of PM.  For a large quant overlay the
command-processor `blockwrite` remains the better path.  `ctrl-pkt` is aimed at
register-heavy reconfiguration with small `.text`.

Two qualifications: this used the installed September toolchain, which lacks
the newer `--reconfig-teardown` / `--reconfig-init-entry` flags.  The generated
sequence also reloads the control overlay PDI once per configuration (the dump
has both `@ctrl_pkt_overlay` and `@ctrl_pkt_overlay_copy`); `--reconfig-init-entry`
should move that load out of the measured region.  Even so, the per-packet PM
cost remains, so the write32/blockwrite result is unlikely to be overturned for
32 x ~11 KiB overlays.

## Runtime load

The production worker already waits on the input config FIFO between calls.
When it is blocked there, it is executing resident code only, so the host can
block-write the overlay tail into the idle core's program memory **without
resetting the core**. When the next config object arrives, the worker enters
`overlay_entry` and runs the newly loaded code.

This avoids the depth-2 FIFO lock problem seen when resetting a core mid-stream
in `tests/poc/overlay.py`. The load must happen after the previous call has
drained and before the next config/input object is released to the worker.

Relying only on the output drain is slightly racy: the core releases the output
object before it reaches the loop tail and blocks on the next config. For a
robust host sequence, add a worker-to-host "ready for next config" lock that
the worker releases just before its config acquire, and make the host wait on
it before the overlay blockwrite. The existing config FIFO does not by itself
prove that the core is out of the previous overlay.

The overlay bytes travel through the command processor as a normal
`blockwrite` to the tile's `0x20000 + overlay_entry` address. For 32 cores,
loading a ~5 KiB overlay costs roughly `32 * 5 KiB * 1.3 us/KiB ~= 0.2 ms`.
That cost is paid only when the overlay set changes, not per matmul call.

A host manifest should contain, per variant/bundle:

- the `overlay_entry` address (same for all variants),
- the overlay tail bytes and size,
- the set of formats included.

The host can emit the blockwrite commands ahead of the next call's DMA
commands, exactly as `tests/poc/overlay.py` prepends its overlay commands.

## Bundles are the simple multi-format answer

The current link shape gives one tail slot. The simplest way to keep several
formats resident is to build an **overlay bundle**:

```text
overlay_entry(fmt, ...) {
    switch (fmt) {
      case 1: program_q4_block(...); break;
      case 2: program_q6_block(...); break;
      case 5: program_new_block(...); break;
    }
}
```

A bundle is linked once and loaded once, then serves many operations. The host
only reloads when the required set of formats is not already in the current
bundle.

This avoids needing a custom linker script. Q4_0 + Q6_K fit comfortably today;
a third ~5 KiB format still fits in the ~13 KiB left by a ~3 KiB resident
worker. When a set no longer fits, build another bundle and swap it.

Multiple simultaneous fixed slots would need a linker-script change: a
`program` region bounded below the slot area plus one region per slot, with the
corresponding sections placed at fixed addresses. That is a natural extension
to `AIETargetLdScript.cpp` (for example, a core `overlay_slots` attribute), but
it is not required for the first overlay step.

## Fixed-address entries

AIE2 calls are absolute: a C call to a symbol defined as an absolute address
links to `j #0x3000` / `jl #0x3000`, with no GOT or register indirection.
That means a slot entry does not need a function pointer if the caller knows
the slot.  Verified with a C caller plus `.set slot0_entry, 0x3000`: the linked
disassembly is `j #0x3000`.

Placement has two routes:

- **Toolchain change.** The generated `*(.text*)` script has no slot regions,
  so `AIETargetLdScript.cpp` would need one region per slot and a matching
  `.text.slotN` placement.  A core attribute (for example `overlay_slots`)
  could drive it.
- **Custom module link.** Keep aiecc for the resident xclbin, then link each
  module with `ld.lld -T slot.ld` at its base.  The object is compiled with
  `-ffunction-sections`; `--gc-sections` plus a script that keeps the
  `unpack_mmul` section places it at the slot address.  A strong entry name
  can be added with `--defsym=slot0_entry=<mangled unpack>`.  Unresolved
  resident symbols come from `--just-symbols=<resident.elf>`; filter out
  symbols the module itself defines so its own entry address wins.  This was
  tested with the real Q4_0 object: `.text` at `0x3000`, size 3200 B.

For the resident side, an object with `.globl slot0_entry` and
`.set slot0_entry, 0x3000` supplies the absolute definitions to the normal
aiecc link.  A tiny called function can pull that object into `link_files`.
The worker then calls `slot0_entry` directly; if several slots are live, the
loop needs a format branch per tile (or a small per-slot copy) rather than an
indirect pointer.  With the shared loop, the entry payload is exactly the
`unpack_mmul` instantiation: 3200 B for Q4_0 and 4000 B for Q6_K.

## Build changes

If this were taken into the production design:

1. Replace the `q4_0_block` / `q6_k_block` branch in `run.py` with one
   `program_block_overlay` external call. Keep the diagnostic build's existing
   blocks for testing.
2. Compile each overlay variant with the real block sources, renaming the
   selected block to `program_block_overlay` (or compile a dispatcher bundle).
3. Put the overlay object last in `link_files` and emit its tail plus the
   `program_block_overlay` address from the final ELF. The PoC's
   `readelf`/`PT_LOAD` extraction is the reference.
4. Keep the resident main identical across variants. Check `core_0_2`,
   `__start`, and every other resident symbol before trusting an overlay.
5. In the host, choose the overlay/bundle for the next format and prepend its
   blockwrite to the next call while the worker is blocked on its config FIFO.

The only format-specific code is in `.text`; the current Q4_0/Q6_K block
objects have no `.data`/`.rodata` payload. If a future format has
format-specific LUTs or static data, those must either stay resident for every
format or get their own data-overlay region. A `.text`-only overlay will not
move them.

## Caveats found in the prototype

- Program memory has no hardware A/B banks.  "Slots" are just fixed address
  ranges inside the single 16 KiB region, so a resident dispatcher plus a
  partial overlay has to be arranged by the linker.
- The core must be disabled and reset while its PM is rewritten.  Resetting a
  core that is blocked mid-stream on a depth-2 FIFO can leave stale lock state;
  the prototype therefore uses depth-1 control FIFOs and reloads between
  calls.  A production reload should happen at a quiescent point with no
  objects in flight, or should reinitialize the affected locks/DMA (the
  `--expand-load-pdis` reconfiguration path already does this).
- Static data, stack, and the data-memory initialization are not part of the
  overlay.  Only `.text` is swapped, so the variants must agree on their data
  layout (the prototype deliberately uses identical internal symbol names).

## Checks

- `readelf -sW` the two variant ELFs and compare every resident symbol, not
  just `overlay_entry`.
- `readelf -SW` the `.text` size; main + overlay must stay under `0x4000`.
- Run the existing `tests/poc/overlay_blocks.py` after changing block sources
  or compile flags; it is the quickest way to catch a shifted overlay entry.
- Verify the full production golden set after switching the worker to the
  overlay call, especially the Q4_0/Q6_K boundary cases.
