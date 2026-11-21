#!/usr/bin/env python3
"""Broadcast control packets from one host DMA into many compute tiles.

A control packet's stream header carries the routing packet id, and its 20-bit
address field is *tile-relative*: an offset inside the destination tile's
1 MiB window.  One packet the switchboxes copy to N ``TileControl`` ports
therefore performs the same write in every tile's own window.  This is the
control-plane equivalent of a data broadcast, and the shape a PM reload could
take: one host fill instead of 32 command-processor block writes.

Three modes:

  broadcast (default)  one 12 KiB write stream to all 32 compute tiles (8x4),
                       verified by CP maskpolls at start, middle and end of
                       every tile.  No read-back packets.

  slot                 one tile: write a real core .text into PM slot B
                       (0x21000) while the core runs in slot A (0x20000), and
                       read it back with control packets.  --verify-all checks
                       every word; this is the mode that shows the data.

  switch               one tile: run A,B rounds; each round streams a slot_entry
                       function into slot B, releases a lock, and executes it.
                       The slot image alternates between rounds; both stages
                       are checked.  --one-run unrolls all rounds into one
                       dispatch instead of one per round.

Usage::

    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design slot --batch --packets 768
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design slot --packets 768 --verify-all --read-batch 4
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --design switch
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay_packets.py --no-run

Notes, measurements and the rejected experiments are in ``overlay.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE / "build" / "overlay_packets"

# AIE2P / NPU2 address map.
COL_SHIFT = 25
ROW_SHIFT = 20
PROGRAM_MEMORY = 0x20000
LOCK_BASE = 0x1F000
LOCK_STRIDE = 0x10

# Routing id for the control packets (5 bits, 0..31).  It selects the switch
# rule only: the destination tile does not filter on it.  Avoid 15, which is
# the shim's task-completion-token id.
CTRL_PKT_ID = 31
# The raw path (AIETargetNPU.cpp) puts the tile's controller_id pkt_type in the
# stream header; the overlay generator assigns 0.
CTRL_PKT_TYPE = 0
# Id used by the read responses in the slot design.
READ_ID = 2

# Txn instruction used for the broadcast verification (see append_polls).
TXN_OPC_MASKPOLL = 4
MASKPOLL_WORDS = 7


def parity_bit(word: int) -> int:
    """Bit 31 set so the header word has odd parity.

    The TileControl drops a control header whose popcount is even: a packet to
    0x1F000 (five bits) works with bit 31 clear, while 0x1F020 (six bits) is
    silently ignored unless bit 31 is set.  This is also what the upstream
    tests build (``create_ctrl_pkt`` in ``xrt_test_wrapper.h``): a header with
    even parity gets ``1 << 31``.
    """
    return 1 if bin(word & 0x7FFFFFFF).count("1") % 2 == 0 else 0


def stream_header(pkt_id: int, pkt_type: int) -> int:
    word = ((pkt_type & 0x7) << 12) | (pkt_id & 0xFF)
    return word | (parity_bit(word) << 31)


def control_header(address: int, opcode: int, stream_id: int, beats: int) -> int:
    word = (
        ((stream_id & 0xFF) << 24)
        | ((opcode & 0x3) << 22)
        | ((beats & 0x3) << 20)
        | (address & 0xFFFFF)
    )
    return word | (parity_bit(word) << 31)


# ---------------------------------------------------------------------------
# Slot write (one tile)


PM_SLOT_B = PROGRAM_MEMORY + 0x1000  # 0x21000, the slot we stream into
# The control header's beats field is 2 bits, so a packet carries at most four
# data words.
PM_WORDS_PER_PACKET = 4
PACKET_WORDS = 2 + PM_WORDS_PER_PACKET  # stream header + control header + data
SLOT_DEFAULT_PACKETS = 14  # 14 write BDs + 2 read BDs = the 16 per channel
SAMPLE_WORDS = 4


def pattern(words: int) -> list[int]:
    """The deterministic image the poll and fallback paths expect."""
    return [0x1000 + i for i in range(words)]


def slot_reads(nwrite: int, verify_all: bool, read_batch: int) -> int:
    """Read BDs this dispatch issues: every chunk, or the two ends."""
    return min(nwrite, read_batch) if verify_all else 2


def mlir_slot(
    pkt_id: int,
    nwrite: int,
    batch: bool,
    verify_all: bool = False,
    read_batch: int = 15,
) -> str:
    # A read request must be its own BD (upstream AIECtrlPacketToDma only
    # batches opcode 0), so a full-slot verification uses one read BD per
    # chunk and covers at most 15 chunks per dispatch.
    nread = slot_reads(nwrite, verify_all, read_batch)
    pkt_words = nwrite * PACKET_WORDS + nread * 2
    rb_words = nread * PM_WORDS_PER_PACKET if verify_all else 8
    lines: list[str] = []
    add = lines.append
    add("module {")
    add("  aie.device(npu2_4col) {")
    add("    %tile_0_0 = aie.tile(0, 0)")
    add("    %tile_0_2 = aie.tile(0, 2)")
    add(
        '    %out_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "out_lock4"}'
    )
    add(
        '    %out_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "out_lock5"}'
    )
    add('    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<8xi32>')
    add("")
    add(f"    aie.packet_flow({hex(pkt_id)}) {{")
    add("      aie.packet_source<%tile_0_0, DMA : 0>")
    add("      aie.packet_dest<%tile_0_2, TileControl : 0>")
    add("    }")
    add(f"    aie.packet_flow({hex(READ_ID)}) {{")
    add("      aie.packet_source<%tile_0_2, TileControl : 0>")
    add("      aie.packet_dest<%tile_0_0, DMA : 1>")
    add("    }")
    add("")
    add("    %core_0_2 = aie.core(%tile_0_2) {")
    add("      %c0 = arith.constant 0 : index")
    add("      %c1 = arith.constant 1 : index")
    add("      %c8 = arith.constant 8 : index")
    add("      %c0i = arith.constant 0 : i32")
    add("      %c1i = arith.constant 1 : i32")
    add("      %cmax = arith.constant 9223372036854775807 : index")
    add(
        "      %r = scf.for %k = %c0 to %cmax step %c1 iter_args(%ctr = %c0i) -> (i32) {"
    )
    add("        %next = arith.addi %ctr, %c1i : i32")
    add("        aie.use_lock(%out_lock5, AcquireGreaterEqual, %c1i)")
    add("        scf.for %j = %c0 to %c8 step %c1 {")
    add("          memref.store %next, %buf_out[%j] : memref<8xi32>")
    add("        }")
    add("        aie.use_lock(%out_lock4, Release, %c1i)")
    add("        scf.yield %next : i32")
    add("      }")
    add("      aie.end")
    add("    }")
    add("")
    add("    %mem_0_2 = aie.mem(%tile_0_2) {")
    add("      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)")
    add("    ^bb1:")
    add("      %c1_ul = arith.constant 1 : i32")
    add("      aie.use_lock(%out_lock4, AcquireGreaterEqual, %c1_ul)")
    add("      aie.dma_bd(%buf_out : memref<8xi32> offset = 0 len = 8)")
    add("      aie.use_lock(%out_lock5, Release, %c1_ul)")
    add("      aie.next_bd ^bb1")
    add("    ^bb2:")
    add("      aie.end")
    add("    }")
    add("")
    add("    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)")
    add("    aie.shim_dma_allocation @ctrlin0 (%tile_0_0, MM2S, 0)")
    add("    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)")
    add("    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 1)")
    add("")
    add(
        f"    aie.runtime_sequence @seq(%pkts: memref<{pkt_words}xi32>,"
        f" %out: memref<16xi32>, %rb: memref<{rb_words}xi32>) {{"
    )
    add("      %c0 = arith.constant 0 : i64")
    add("      %c1 = arith.constant 1 : i64")
    add("      %c8 = arith.constant 8 : i64")
    add("      %c4 = arith.constant 4 : i64")
    add("")
    add(
        "      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
        " {id = 1 : i64, issue_token = true, metadata = @out0} : memref<16xi32>"
    )
    add(
        "      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %c8] [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
        " {id = 2 : i64, issue_token = true, metadata = @out0} : memref<16xi32>"
    )
    if verify_all:
        add(f"      %crb = arith.constant {rb_words} : i64")
        add(
            "      aiex.npu.dma_memcpy_nd(%rb[%c0, %c0, %c0, %c0]"
            " [%c1, %c1, %c1, %crb] [%c0, %c0, %c0, %c1])"
            f" {{id = 1 : i64, issue_token = true, metadata = @ctrl0}}"
            f" : memref<{rb_words}xi32>"
        )
    else:
        add(
            "      aiex.npu.dma_memcpy_nd(%rb[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c4] [%c0, %c0, %c0, %c1])"
            " {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>"
        )
        add(
            "      aiex.npu.dma_memcpy_nd(%rb[%c0, %c0, %c0, %c4] [%c1, %c1, %c1, %c4] [%c0, %c0, %c0, %c1])"
            " {id = 2 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>"
        )
    add("")
    if batch:
        add(f"      %cb = arith.constant {nwrite * PACKET_WORDS} : i64")
        add(f"      %cbl = arith.constant {nwrite * PACKET_WORDS} : i64")
        add(
            "      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %c0]"
            " [%c1, %c1, %c1, %cbl] [%c0, %c0, %c0, %c1])"
            " {id = 0 : i64, issue_token = true, metadata = @ctrlin0}"
            f" : memref<{pkt_words}xi32>"
        )
    else:
        for k in range(nwrite):
            add(f"      %cp{k} = arith.constant {PACKET_WORDS * k} : i64")
            add(f"      %cs{k} = arith.constant {PACKET_WORDS} : i64")
            add(
                f"      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %cp{k}]"
                f" [%c1, %c1, %c1, %cs{k}] [%c0, %c0, %c0, %c1])"
                f" {{id = {k} : i64, issue_token = true, metadata = @ctrlin0}}"
                f" : memref<{pkt_words}xi32>"
            )

    def read_dma(index: int, offset: int, length: int) -> None:
        rid = 1 + index if batch else nwrite + index
        add(f"      %cro{index} = arith.constant {offset} : i64")
        add(f"      %crl{index} = arith.constant {length} : i64")
        add(
            f"      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %cro{index}]"
            f" [%c1, %c1, %c1, %crl{index}] [%c0, %c0, %c0, %c1])"
            f" {{id = {rid} : i64, issue_token = true,"
            " metadata = @ctrlin0}"
            f" : memref<{pkt_words}xi32>"
        )

    for r in range(nread):
        read_dma(r, nwrite * PACKET_WORDS + 2 * r, 2)
    add("")
    add("      aiex.npu.dma_wait {symbol = @out0}")
    add("      aiex.npu.dma_wait {symbol = @out0}")
    for _ in range(1 if verify_all else 2):
        add("      aiex.npu.dma_wait {symbol = @ctrl0}")
    add("    }")
    add("  }")
    add("}")
    add("")
    return "\n".join(lines)


def fill_words(values: list[int], words: int) -> list[int] | None:
    """Repeat ``values`` to exactly ``words`` words, or None when empty."""
    if not values:
        return None
    return (values * (words // len(values) + 1))[:words]


def variant_text(words: int) -> list[int] | None:
    """A second real kernel's .text, from overlay.py's variant builder.

    Temporary share while overlay.py is still the runnable A/B baseline: the
    slot test streams a kernel that is genuinely different from the one the
    tile runs.  When the switch mode lands, this builder moves here and
    overlay.py is retired.
    """
    try:
        from overlay import build_variant
    except Exception as exc:  # noqa: BLE001 - fall back to the resident image
        print(
            f"  variant image unavailable ({exc}); using resident .text",
            file=sys.stderr,
        )
        return None
    variant = build_variant("slot_image", 2, 40, 1, quiet=True)
    overlay = next(iter(variant["overlays"].values()))
    data = Path(overlay["file"]).read_bytes()
    vals = list(struct.unpack("<%dI" % (len(data) // 4), data))
    return fill_words(vals, words)


def core_text(work: Path, words: int) -> list[int] | None:
    """Repeat a built core's program-memory segment to fill ``words``."""
    elfs = sorted((work / "aie.mlir.prj").glob("elfs_main_core_*/*.elf"))
    if not elfs:
        return None
    data = elfs[0].read_bytes()
    if data[:4] != b"\x7fELF":
        return None
    e_phoff = struct.unpack_from("<I", data, 28)[0]
    e_phentsize = struct.unpack_from("<H", data, 42)[0]
    e_phnum = struct.unpack_from("<H", data, 44)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, _vaddr, p_paddr, p_filesz = struct.unpack_from(
            "<IIIII", data, off
        )
        if p_type != 1 or p_paddr != 0:
            continue
        payload = data[p_offset : p_offset + p_filesz]
        if len(payload) % 4:
            payload += b"\0" * (4 - len(payload) % 4)
        vals = list(struct.unpack("<%dI" % (len(payload) // 4), payload))
        return fill_words(vals, words)
    return None


def slot_blob(
    pkt_id: int,
    nwrite: int,
    image: list[int] | None = None,
    verify_all: bool = False,
    read_batch: int = 15,
) -> tuple[list[int], list[int]]:
    """The control-packet blob and the PM image it writes into slot B.

    The blob holds the write stream followed by the read requests.  For
    ``verify_all`` it holds the first pass's requests; the host rewrites that
    section per pass and the runtime only pushes ``slot_reads()`` of them.
    """
    if image is None:
        image = pattern(nwrite * PM_WORDS_PER_PACKET)
    words: list[int] = []
    for k in range(nwrite):
        addr = PM_SLOT_B + k * PM_WORDS_PER_PACKET * 4
        words += [
            stream_header(pkt_id, CTRL_PKT_TYPE),
            control_header(addr, 0, 0, PM_WORDS_PER_PACKET - 1),
        ]
        words += image[k * PM_WORDS_PER_PACKET : (k + 1) * PM_WORDS_PER_PACKET]
    # Responses are tagged READ_ID and routed by the TileControl -> shim flow.
    nread = slot_reads(nwrite, verify_all, read_batch)
    if verify_all:
        addrs = [PM_SLOT_B + k * PM_WORDS_PER_PACKET * 4 for k in range(nread)]
    else:
        addrs = [
            PM_SLOT_B,
            PM_SLOT_B + (nwrite * PM_WORDS_PER_PACKET - PM_WORDS_PER_PACKET) * 4,
        ]
    for addr in addrs:
        words += [
            stream_header(pkt_id, CTRL_PKT_TYPE),
            control_header(addr, 1, READ_ID, PM_WORDS_PER_PACKET - 1),
        ]
    return words, image


# ---------------------------------------------------------------------------
# Switch: execute a streamed slot

SWITCH_SLOT = PM_SLOT_B

SWITCH_RESIDENT = r"""
#include <stdint.h>

extern "C" void stage_a(int32_t *out, int32_t round) {
  int32_t base = 0xA000 + (round & 1) * 0x100;
  for (int i = 0; i < 8; i++) out[i] = base + i;
}

extern "C" void stage_b(int32_t *out) {
  ((void (*)(int32_t *))0x21000)(out);
}
"""

SWITCH_SLOT_SRC = r"""
#include <stdint.h>

#ifndef SLOT_BASE
#define SLOT_BASE 0xB000
#endif

extern "C" void slot_entry(int32_t *out) {
  for (int i = 0; i < 8; i++) out[i] = SLOT_BASE + i;
}
"""

SWITCH_SLOT_LD = """
SECTIONS {
  . = 0x21000;
  .text : { *(.text.slot_entry) }
}
"""


def _peano() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "llvm-aie" / "bin"


def _aie_tool(name: str, *args, cwd: Path) -> None:
    try:
        subprocess.run(
            [str(_peano() / name), *map(str, args)],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        sys.stderr.write(err.stdout or "")
        sys.stderr.write(err.stderr or "")
        raise


def build_slot_entry(work: Path, tag: str, base: int) -> list[int]:
    """Compile slot_entry with ``base`` at SWITCH_SLOT and return its words."""
    (work / f"slot_entry_{tag}.cc").write_text(SWITCH_SLOT_SRC)
    (work / f"slot_entry_{tag}.ld").write_text(SWITCH_SLOT_LD)
    _aie_tool(
        "clang++",
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-ffunction-sections",
        f"-DSLOT_BASE={base}",
        "-c",
        f"slot_entry_{tag}.cc",
        "-o",
        f"slot_entry_{tag}.o",
        cwd=work,
    )
    _aie_tool(
        "ld.lld",
        "-T",
        f"slot_entry_{tag}.ld",
        f"slot_entry_{tag}.o",
        "-o",
        f"slot_entry_{tag}.elf",
        cwd=work,
    )
    data = (work / f"slot_entry_{tag}.elf").read_bytes()
    e_phoff = struct.unpack_from("<I", data, 28)[0]
    e_phentsize = struct.unpack_from("<H", data, 42)[0]
    e_phnum = struct.unpack_from("<H", data, 44)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, _vaddr, p_paddr, p_filesz = struct.unpack_from(
            "<IIIII", data, off
        )
        if p_type == 1 and p_paddr == SWITCH_SLOT:
            payload = data[p_offset : p_offset + p_filesz]
            if len(payload) % 4:
                payload += b"\0" * (4 - len(payload) % 4)
            return list(struct.unpack("<%dI" % (len(payload) // 4), payload))
    raise RuntimeError(f"slot_entry_{tag}.elf has no segment at 0x21000")


def build_resident_object(work: Path) -> None:
    (work / "switch_resident.cc").write_text(SWITCH_RESIDENT)
    _aie_tool(
        "clang++",
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-c",
        "switch_resident.cc",
        "-o",
        "switch_resident.o",
        cwd=work,
    )


def mlir_switch(npackets: int, rounds: int, one_run: bool) -> str:
    codes = npackets * PACKET_WORDS
    pkt_words = 2 * codes + 3 if one_run else codes + 3
    lines: list[str] = []
    add = lines.append
    add("module {")
    add("  aie.device(npu2_4col) {")
    add("    %tile_0_0 = aie.tile(0, 0)")
    add("    %tile_0_2 = aie.tile(0, 2)")
    add('    %go_lock = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "go_lock"}')
    add(
        '    %out_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "out_lock4"}'
    )
    add(
        '    %out_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "out_lock5"}'
    )
    add('    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<8xi32>')
    add("    func.func private @stage_a(memref<8xi32>, i32)")
    add('        attributes {link_with = "switch_resident.o"}')
    add("    func.func private @stage_b(memref<8xi32>)")
    add('        attributes {link_with = "switch_resident.o"}')
    add("")
    add(f"    aie.packet_flow({hex(CTRL_PKT_ID)}) {{")
    add("      aie.packet_source<%tile_0_0, DMA : 0>")
    add("      aie.packet_dest<%tile_0_2, TileControl : 0>")
    add("    }")
    add("")
    add("    %core_0_2 = aie.core(%tile_0_2) {")
    add("      %c0 = arith.constant 0 : index")
    add("      %c1i = arith.constant 1 : index")
    add("      %cmax = arith.constant 9223372036854775807 : index")
    add("      %c1 = arith.constant 1 : i32")
    add("      scf.for %r = %c0 to %cmax step %c1i {")
    add("        %ri = arith.index_cast %r : index to i32")
    add("        aie.use_lock(%out_lock5, AcquireGreaterEqual, %c1)")
    add("        func.call @stage_a(%buf_out, %ri) : (memref<8xi32>, i32) -> ()")
    add("        aie.use_lock(%out_lock4, Release, %c1)")
    add("        aie.use_lock(%go_lock, AcquireGreaterEqual, %c1)")
    add("        aie.use_lock(%out_lock5, AcquireGreaterEqual, %c1)")
    add("        func.call @stage_b(%buf_out) : (memref<8xi32>) -> ()")
    add("        aie.use_lock(%out_lock4, Release, %c1)")
    add("      }")
    add("      aie.end")
    add("    }")
    add("")
    add("    %mem_0_2 = aie.mem(%tile_0_2) {")
    add("      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)")
    add("    ^bb1:")
    add("      %c1_ul = arith.constant 1 : i32")
    add("      aie.use_lock(%out_lock4, AcquireGreaterEqual, %c1_ul)")
    add("      aie.dma_bd(%buf_out : memref<8xi32> offset = 0 len = 8)")
    add("      aie.use_lock(%out_lock5, Release, %c1_ul)")
    add("      aie.next_bd ^bb1")
    add("    ^bb2:")
    add("      aie.end")
    add("    }")
    add("")
    add("    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)")
    add("    aie.shim_dma_allocation @ctrlin0 (%tile_0_0, MM2S, 0)")
    add("    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)")
    add("")
    if one_run:
        add(
            f"    aie.runtime_sequence @seq(%pkts: memref<{pkt_words}xi32>,"
            f" %out: memref<{16 * rounds}xi32>) {{"
        )
    else:
        add(
            f"    aie.runtime_sequence @seq(%pkts: memref<{pkt_words}xi32>,"
            " %out_a: memref<8xi32>, %out_b: memref<8xi32>) {"
        )
    add("      %c0 = arith.constant 0 : i64")
    add("      %c1 = arith.constant 1 : i64")
    add("      %c8 = arith.constant 8 : i64")
    add(f"      %cc = arith.constant {codes} : i64")
    add("      %c3 = arith.constant 3 : i64")
    if one_run:
        add(f"      %cl = arith.constant {2 * codes} : i64")
        for r in range(rounds):
            aoff = 16 * r
            add(f"      %ca{r} = arith.constant {aoff} : i64")
            add(f"      %cb{r} = arith.constant {aoff + 8} : i64")
            add(f"      %cd{r} = arith.constant {(r % 2) * codes} : i64")
            add(
                f"      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %ca{r}]"
                " [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
                " {id = 1 : i64, issue_token = true, metadata = @out0}"
                f" : memref<{16 * rounds}xi32>"
            )
            add(
                f"      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %cd{r}]"
                " [%c1, %c1, %c1, %cc] [%c0, %c0, %c0, %c1])"
                " {id = 0 : i64, issue_token = false, metadata = @ctrlin0}"
                f" : memref<{pkt_words}xi32>"
            )
            add(
                "      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %cl]"
                " [%c1, %c1, %c1, %c3] [%c0, %c0, %c0, %c1])"
                " {id = 1 : i64, issue_token = false, metadata = @ctrlin0}"
                f" : memref<{pkt_words}xi32>"
            )
            add(
                f"      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %cb{r}]"
                " [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
                " {id = 2 : i64, issue_token = true, metadata = @out0}"
                f" : memref<{16 * rounds}xi32>"
            )
            add("      aiex.npu.dma_wait {symbol = @out0}")
            add("      aiex.npu.dma_wait {symbol = @out0}")
    else:
        add(
            "      aiex.npu.dma_memcpy_nd(%out_a[%c0, %c0, %c0, %c0]"
            " [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
            " {id = 1 : i64, issue_token = true, metadata = @out0}"
            " : memref<8xi32>"
        )
        add(
            "      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %c0]"
            " [%c1, %c1, %c1, %cc] [%c0, %c0, %c0, %c1])"
            " {id = 0 : i64, issue_token = false, metadata = @ctrlin0}"
            f" : memref<{pkt_words}xi32>"
        )
        add(
            "      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %cc]"
            " [%c1, %c1, %c1, %c3] [%c0, %c0, %c0, %c1])"
            " {id = 1 : i64, issue_token = false, metadata = @ctrlin0}"
            f" : memref<{pkt_words}xi32>"
        )
        add(
            "      aiex.npu.dma_memcpy_nd(%out_b[%c0, %c0, %c0, %c0]"
            " [%c1, %c1, %c1, %c8] [%c0, %c0, %c0, %c1])"
            " {id = 2 : i64, issue_token = true, metadata = @out0}"
            " : memref<8xi32>"
        )
        add("      aiex.npu.dma_wait {symbol = @out0}")
        add("      aiex.npu.dma_wait {symbol = @out0}")
    add("    }")
    add("  }")
    add("}")
    add("")
    return "\n".join(lines)


def code_section(npackets: int, image: list[int]) -> list[int]:
    """The control packets that write one slot image."""
    words: list[int] = []
    for k in range(npackets):
        addr = SWITCH_SLOT + k * PM_WORDS_PER_PACKET * 4
        words += [
            stream_header(CTRL_PKT_ID, CTRL_PKT_TYPE),
            control_header(addr, 0, 0, PM_WORDS_PER_PACKET - 1),
        ]
        words += image[k * PM_WORDS_PER_PACKET : (k + 1) * PM_WORDS_PER_PACKET]
    return words


def lock_packet() -> list[int]:
    """Release the go-lock (lock 2 at 0x1F020)."""
    return [
        stream_header(CTRL_PKT_ID, CTRL_PKT_TYPE),
        control_header(LOCK_BASE + 2 * LOCK_STRIDE, 0, 0, 0),
        1,
    ]


GRID_COLS = 8
GRID_ROWS = 4
GRID_TILES = GRID_COLS * GRID_ROWS


def mlir_broadcast(pkt_id: int, nwrite: int) -> str:
    """One broadcast write flow to all 32 compute tiles, no read-back."""
    pkt_words = nwrite * PACKET_WORDS
    lines: list[str] = []
    add = lines.append
    add("module {")
    add("  aie.device(npu2) {")
    seen: set[tuple[int, int]] = set()

    def declare(col: int, row: int) -> None:
        if (col, row) not in seen:
            seen.add((col, row))
            add(f"    %tile_{col}_{row} = aie.tile({col}, {row})")

    for c in range(GRID_COLS):
        for r in range(2, 2 + GRID_ROWS):
            declare(c, r)
        declare(c, 0)
    add("")
    add(f"    aie.packet_flow({hex(pkt_id)}) {{")
    add("      aie.packet_source<%tile_0_0, DMA : 0>")
    for c in range(GRID_COLS):
        for r in range(2, 2 + GRID_ROWS):
            add(f"      aie.packet_dest<%tile_{c}_{r}, TileControl : 0>")
    add("    }")
    add("")
    add("    aie.shim_dma_allocation @ctrlin (%tile_0_0, MM2S, 0)")
    add("")
    add(f"    aie.runtime_sequence @seq(%pkts: memref<{pkt_words}xi32>) {{")
    add("      %c0 = arith.constant 0 : i64")
    add("      %c1 = arith.constant 1 : i64")
    add(f"      %cw = arith.constant {nwrite * PACKET_WORDS} : i64")
    add(
        "      aiex.npu.dma_memcpy_nd(%pkts[%c0, %c0, %c0, %c0]"
        " [%c1, %c1, %c1, %cw] [%c0, %c0, %c0, %c1])"
        " {id = 0 : i64, issue_token = true, metadata = @ctrlin}"
        f" : memref<{pkt_words}xi32>"
    )
    add("      aiex.npu.dma_wait {symbol = @ctrlin}")
    add("    }")
    add("  }")
    add("}")
    add("")
    return "\n".join(lines)


def broadcast_blob(pkt_id: int, nwrite: int) -> list[int]:
    image = pattern(nwrite * PM_WORDS_PER_PACKET)
    words: list[int] = []
    for k in range(nwrite):
        addr = PM_SLOT_B + k * PM_WORDS_PER_PACKET * 4
        words += [
            stream_header(pkt_id, CTRL_PKT_TYPE),
            control_header(addr, 0, 0, PM_WORDS_PER_PACKET - 1),
        ]
        words += image[k * PM_WORDS_PER_PACKET : (k + 1) * PM_WORDS_PER_PACKET]
    return words


def append_polls(work: Path, nwrite: int) -> int:
    """Append one maskpoll per sampled word to a compiled insts.bin.

    The installed toolchain has no ``aiex.npu.maskpoll`` op, so the instruction
    is appended to the txn: opcode 4, 0, address, 0, value, mask, byte size.
    """
    path = work / "insts.bin"
    data = path.read_bytes()
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    image = pattern(nwrite * PM_WORDS_PER_PACKET)
    slot_words = nwrite * PM_WORDS_PER_PACKET
    first = PM_SLOT_B
    middle = PM_SLOT_B + (slot_words // 2) * 4
    last = PM_SLOT_B + (slot_words - SAMPLE_WORDS) * 4
    samples = []
    for at, idx in (
        (first, 0),
        (middle, slot_words // 2),
        (last, slot_words - SAMPLE_WORDS),
    ):
        for k in range(SAMPLE_WORDS):
            samples.append((at + 4 * k, image[idx + k]))
    polls = 0
    for c in range(GRID_COLS):
        for r in range(2, 2 + GRID_ROWS):
            base = (c << COL_SHIFT) | (r << ROW_SHIFT)
            for off, val in samples:
                words += [
                    TXN_OPC_MASKPOLL,
                    0,
                    base | off,
                    0,
                    val,
                    0xFFFFFFFF,
                    MASKPOLL_WORDS * 4,
                ]
                polls += 1
    words[2] += polls
    words[3] = len(words) * 4
    path.write_bytes(b"".join(w.to_bytes(4, "little") for w in words))
    return polls


CC_SWITCH = r"""
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> out(n / 4);
  f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size() * 4);
  return out;
}

int main(int argc, char** argv) {
  int rounds = argc > 1 ? std::stoi(argv[1]) : @ROUNDS@;
  std::string dir = argc > 2 ? argv[2] : ".";

  constexpr int kCodes = @CODES@;
  auto instr = read_words(dir + "/insts.bin");
  auto pkts = read_words(dir + "/pkts.bin");
  auto codes = read_words(dir + "/codes.bin");

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(dir + "/aie.xclbin");
  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid(),
                                 xrt::hw_context::access_mode::exclusive);
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_pkts = xrt::bo(device, pkts.size() * sizeof(uint32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out_a = xrt::bo(device, 8 * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
  auto bo_out_b = xrt::bo(device, 8 * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(5));

  memcpy(bo_pkts.map<void*>(), pkts.data(), pkts.size() * sizeof(uint32_t));
  bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  uint32_t* pktWords = bo_pkts.map<uint32_t*>();

  int errors = 0;
  for (int r = 0; r < rounds; r++) {
    // The write stream is the only thing that changes between rounds.
    memcpy(pktWords, codes.data() + (size_t)r * kCodes,
           kCodes * sizeof(uint32_t));
    bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_out_a.map<void*>(), 0, 8 * sizeof(uint32_t));
    bo_out_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_out_b.map<void*>(), 0, 8 * sizeof(uint32_t));
    bo_out_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    run.set_arg(0, (uint32_t)3);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (uint32_t)instr.size());
    run.set_arg(3, bo_pkts);
    run.set_arg(4, bo_out_a);
    run.set_arg(5, bo_out_b);
    auto t0 = std::chrono::steady_clock::now();
    run.start();
    ert_cmd_state st = run.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (st != ERT_CMD_STATE_COMPLETED) {
      printf("round %d: command did not complete (%d)\n", r, (int)st);
      return 2;
    }
    bo_out_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto a = bo_out_a.map<uint32_t*>();
    auto b = bo_out_b.map<uint32_t*>();

    int round_errors = 0;
    uint32_t abase = 0xA000u + (uint32_t)(r & 1) * 0x100u;
    uint32_t bbase = 0xB000u + (uint32_t)(r & 1) * 0x100u;
    for (int i = 0; i < 8; i++) {
      if (a[i] != abase + (uint32_t)i) round_errors++;
      if (b[i] != bbase + (uint32_t)i) round_errors++;
    }
    printf("  round %d: A %#x B %#x  %s  (%.1f us)\n", r, a[0], b[0],
           round_errors ? "FAIL" : "ok",
           std::chrono::duration<double, std::micro>(t1 - t0).count());
    errors += round_errors;
  }
  if (errors) {
    printf("FAILED: %d word errors\n", errors);
    return 1;
  }
  printf("ok: %d rounds of A,B executed with the streamed slot\n", rounds);
  return 0;
}
"""


CC_SWITCH_ONE = r"""
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> out(n / 4);
  f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size() * 4);
  return out;
}

int main(int argc, char** argv) {
  std::string dir = argc > 2 ? argv[2] : ".";

  constexpr int kRounds = @ROUNDS@;
  auto instr = read_words(dir + "/insts.bin");
  auto pkts = read_words(dir + "/pkts.bin");

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(dir + "/aie.xclbin");
  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid(),
                                 xrt::hw_context::access_mode::exclusive);
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_pkts = xrt::bo(device, pkts.size() * sizeof(uint32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, 16 * kRounds * sizeof(uint32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  memcpy(bo_pkts.map<void*>(), pkts.data(), pkts.size() * sizeof(uint32_t));
  bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bo_out.map<void*>(), 0, 16 * kRounds * sizeof(uint32_t));
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, (uint32_t)3);
  run.set_arg(1, bo_instr);
  run.set_arg(2, (uint32_t)instr.size());
  run.set_arg(3, bo_pkts);
  run.set_arg(4, bo_out);
  auto t0 = std::chrono::steady_clock::now();
  run.start();
  ert_cmd_state st = run.wait();
  auto t1 = std::chrono::steady_clock::now();
  if (st != ERT_CMD_STATE_COMPLETED) {
    printf("one run: command did not complete (%d)\n", (int)st);
    return 2;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto o = bo_out.map<uint32_t*>();

  int errors = 0;
  for (int r = 0; r < kRounds; r++) {
    const uint32_t* a = o + r * 16;
    const uint32_t* b = o + r * 16 + 8;
    uint32_t abase = 0xA000u + (uint32_t)(r & 1) * 0x100u;
    uint32_t bbase = 0xB000u + (uint32_t)(r & 1) * 0x100u;
    int round_errors = 0;
    for (int i = 0; i < 8; i++) {
      if (a[i] != abase + (uint32_t)i) round_errors++;
      if (b[i] != bbase + (uint32_t)i) round_errors++;
    }
    printf("  round %d: A %#x B %#x  %s\n", r, a[0], b[0],
           round_errors ? "FAIL" : "ok");
    errors += round_errors;
  }
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  printf("  %d rounds in one dispatch: %.1f us (%.1f us/round)\n", kRounds, us,
         us / kRounds);
  if (errors) {
    printf("FAILED: %d word errors\n", errors);
    return 1;
  }
  printf("ok: one xrt::run executed %d A,B rounds\n", kRounds);
  return 0;
}
"""


CC_BROADCAST = r"""
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> out(n / 4);
  f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size() * 4);
  return out;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::stoi(argv[1]) : 1;
  std::string dir = argc > 2 ? argv[2] : ".";

  auto instr = read_words(dir + "/insts.bin");
  auto pkts = read_words(dir + "/pkts.bin");

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(dir + "/aie.xclbin");
  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid(),
                                 xrt::hw_context::access_mode::exclusive);
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_pkts = xrt::bo(device, pkts.size() * sizeof(uint32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  memcpy(bo_pkts.map<void*>(), pkts.data(), pkts.size() * sizeof(uint32_t));
  bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (int it = 0; it < iters; it++) {
    auto r = xrt::run(kernel);
    r.set_arg(0, (uint32_t)3);
    r.set_arg(1, bo_instr);
    r.set_arg(2, (uint32_t)instr.size());
    r.set_arg(3, bo_pkts);
    auto t0 = std::chrono::steady_clock::now();
    r.start();
    ert_cmd_state st = r.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (st != ERT_CMD_STATE_COMPLETED) {
      printf("iteration %d: command did not complete (%d) - a poll failed\n",
             it, (int)st);
      return 2;
    }
    printf("  all maskpolls matched in %.1f us\n",
           std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  printf("ok: CP polls verified start/middle/end of %d tiles\n", @NTILES@);
  return 0;
}
"""


CC_SLOT = r"""
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> out(n / 4);
  f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size() * 4);
  return out;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::stoi(argv[1]) : 1;
  std::string dir = argc > 2 ? argv[2] : ".";

  constexpr int kNread = @NREAD@;
  constexpr int kPasses = @PASSES@;
  constexpr int kReoff = @REOFF@;
  constexpr bool kVerifyAll = @VERIFY_ALL@ != 0;

  auto instr = read_words(dir + "/insts.bin");
  auto pkts = read_words(dir + "/pkts.bin");
  auto image = read_words(dir + "/image.bin");
  auto reads = kVerifyAll ? read_words(dir + "/reads.bin")
                          : std::vector<uint32_t>{};

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(dir + "/aie.xclbin");
  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid(),
                                 xrt::hw_context::access_mode::exclusive);
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_pkts = xrt::bo(device, pkts.size() * sizeof(uint32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, 16 * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_rb = xrt::bo(device, @NRB@ * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(5));

  uint32_t* pktWords = bo_pkts.map<uint32_t*>();
  memcpy(pktWords, pkts.data(), pkts.size() * sizeof(uint32_t));
  bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const int kRuns = kVerifyAll ? kPasses : iters;
  const size_t kChunks = image.size() / 4;
  size_t checked = 0;
  int errors = 0;

  for (int it = 0; it < kRuns; it++) {
    if (kVerifyAll) {
      // Send this pass's read requests; the rest of the blob is unchanged.
      memcpy(pktWords + kReoff, reads.data() + (size_t)it * kNread * 2,
             kNread * 2 * sizeof(uint32_t));
      bo_pkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    memset(bo_rb.map<void*>(), 0, @NRB@ * sizeof(uint32_t));
    bo_rb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_out.map<void*>(), 0, 16 * sizeof(uint32_t));
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto r = xrt::run(kernel);
    r.set_arg(0, (uint32_t)3);
    r.set_arg(1, bo_instr);
    r.set_arg(2, (uint32_t)instr.size());
    r.set_arg(3, bo_pkts);
    r.set_arg(4, bo_out);
    r.set_arg(5, bo_rb);
    auto t0 = std::chrono::steady_clock::now();
    r.start();
    ert_cmd_state st = r.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (st != ERT_CMD_STATE_COMPLETED) {
      printf("pass %d: command did not complete (%d)\n", it, (int)st);
      return 2;
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_rb.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto o = bo_out.map<uint32_t*>();
    auto rb = bo_rb.map<uint32_t*>();

    if (it == 0) {
      printf("  %zu bytes into slot B in %.1f us\n", image.size() * 4,
             std::chrono::duration<double, std::micro>(t1 - t0).count());
      printf("  core counter: first snapshot %#x, second %#x\n", o[0], o[8]);
      if (o[0] == 0 || o[8] <= o[0]) {
        printf("  FAIL: core did not keep running across the PM write\n");
        errors++;
      }
    }

    if (kVerifyAll) {
      for (int r = 0; r < kNread; r++) {
        size_t k = (size_t)it * kNread + r;
        if (k >= kChunks) break;
        for (int j = 0; j < 4; j++)
          if (rb[r * 4 + j] != image[k * 4 + j]) {
            if (errors < 8)
              printf("    chunk %zu word %d: got %#x wrote %#x\n", k, j,
                     rb[r * 4 + j], image[k * 4 + j]);
            errors++;
          }
        checked++;
      }
    } else {
      for (size_t i = 0; i < 8 && i < image.size(); i++) {
        uint32_t want = i < 4 ? image[i] : image[image.size() - 8 + i];
        printf("    slot B[%s%zu] = %#x (wrote %#x) %s\n",
               i < 4 ? "" : "end+", i < 4 ? i : i - 4, rb[i], want,
               rb[i] == want ? "ok" : "MISMATCH");
        if (rb[i] != want) errors++;
      }
    }
  }
  if (kVerifyAll)
    printf("  verified %zu of %zu chunks (%d passes of %d reads)\n", checked,
           kChunks, kPasses, kNread);
  if (errors) {
    printf("FAILED: %d errors\n", errors);
    return 1;
  }
  printf("ok: control packets wrote slot B while the core ran\n");
  return 0;
}
"""


def _env() -> dict:
    env = os.environ.copy()
    site = Path(sysconfig.get_paths()["purelib"])
    peano = site / "llvm-aie" / "bin"
    aiecc = site / "mlir_aie" / "bin"
    env["PATH"] = f"{peano}:{aiecc}:" + env.get("PATH", "")
    return env


def build(
    design: str,
    packets: int | None = None,
    batch: bool = False,
    verify_all: bool = False,
    read_batch: int = 15,
    image: str = "variant",
    count: int = 1,
    one_run: bool = False,
) -> Path:
    pkt_id = CTRL_PKT_ID
    npackets = 0
    image: list[int] = []
    if design == "broadcast":
        nwrite = 768 if packets is None else packets
        work = WORK_ROOT / "broadcast"
        mlir = mlir_broadcast(pkt_id, nwrite)
    elif design == "slot":
        nwrite = SLOT_DEFAULT_PACKETS if packets is None else packets
        work = WORK_ROOT / "slot"
        mlir = mlir_slot(pkt_id, nwrite, batch or verify_all, verify_all, read_batch)
    else:
        nwrite = 0
        work = WORK_ROOT / "switch"
        mlir = ""

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    if design == "switch":
        build_resident_object(work)
        images = [
            build_slot_entry(work, "b1", 0xB000),
            build_slot_entry(work, "b2", 0xB100),
        ]
        npackets = max(
            1,
            max(
                (len(img) + PM_WORDS_PER_PACKET - 1) // PM_WORDS_PER_PACKET
                for img in images
            ),
        )
        images = [
            fill_words(img, npackets * PM_WORDS_PER_PACKET) or [] for img in images
        ]
        mlir = mlir_switch(npackets, count, one_run)

    (work / "aie.mlir").write_text(mlir)

    try:
        subprocess.run(
            [
                "aiecc",
                "--get-xclbin",
                "--get-npu-insts",
                "--xclbin-name=aie.xclbin",
                "--npu-insts-name=insts.bin",
                "aie.mlir",
            ],
            check=True,
            cwd=work,
            env=_env(),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        sys.stderr.write(err.stdout or "")
        sys.stderr.write(err.stderr or "")
        raise

    nread = slot_reads(nwrite, verify_all, read_batch)
    passes = 1
    if design == "switch":
        sections = [code_section(npackets, img) for img in images]
        if one_run:
            words = sections[0] + sections[1] + lock_packet()
        else:
            words = sections[0] + lock_packet()
            codes: list[int] = []
            for r in range(count):
                codes += sections[r % 2]
            (work / "codes.bin").write_bytes(
                b"".join(w.to_bytes(4, "little") for w in codes)
            )
        (work / "pkts.bin").write_bytes(
            b"".join(w.to_bytes(4, "little") for w in words)
        )
        template = CC_SWITCH_ONE if one_run else CC_SWITCH
    elif design == "broadcast":
        words = broadcast_blob(pkt_id, nwrite)
        (work / "pkts.bin").write_bytes(
            b"".join(w.to_bytes(4, "little") for w in words)
        )
        polls = append_polls(work, nwrite)
        print(f"  appended {polls} maskpolls", flush=True)
        template = CC_BROADCAST
    else:
        slot_words = nwrite * PM_WORDS_PER_PACKET
        pixels = None
        if image == "variant":
            pixels = variant_text(slot_words)
        if pixels is None:
            pixels = core_text(work, slot_words)
        words, pixels = slot_blob(pkt_id, nwrite, pixels, verify_all, read_batch)
        (work / "pkts.bin").write_bytes(
            b"".join(w.to_bytes(4, "little") for w in words)
        )
        (work / "image.bin").write_bytes(
            b"".join(w.to_bytes(4, "little") for w in pixels)
        )
        if verify_all:
            passes = (nwrite + nread - 1) // nread
            reads: list[int] = []
            for p in range(passes):
                for r in range(nread):
                    k = min(p * nread + r, nwrite - 1)
                    addr = PM_SLOT_B + k * PM_WORDS_PER_PACKET * 4
                    reads += [
                        stream_header(pkt_id, CTRL_PKT_TYPE),
                        control_header(addr, 1, READ_ID, PM_WORDS_PER_PACKET - 1),
                    ]
            (work / "reads.bin").write_bytes(
                b"".join(w.to_bytes(4, "little") for w in reads)
            )
        template = CC_SLOT

    (work / "overlay_packets.cc").write_text(
        template.replace(
            "@NTILES@",
            str(GRID_TILES if design == "broadcast" else 1),
        )
        .replace(
            "@NRB@",
            str(nread * PM_WORDS_PER_PACKET if verify_all else 8),
        )
        .replace("@VERIFY_ALL@", "1" if verify_all else "0")
        .replace("@NREAD@", str(nread))
        .replace("@PASSES@", str(passes))
        .replace("@REOFF@", str(nwrite * PACKET_WORDS))
        .replace("@ROUNDS@", str(count))
        .replace("@CODES@", str(npackets * PACKET_WORDS))
    )
    site = Path(sysconfig.get_paths()["purelib"])
    host_cxx = shutil.which("clang++") or shutil.which("g++")
    subprocess.run(
        [
            host_cxx,
            "-O2",
            "-g",
            f"-I{site / 'mlir_aie' / 'include'}",
            "-o",
            str(work / "overlay_packets"),
            str(work / "overlay_packets.cc"),
            "-lxrt++",
            "-lxrt_coreutil",
        ],
        check=True,
        cwd=work,
    )
    return work


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Broadcast control packets into compute tiles."
    )
    parser.add_argument(
        "--design",
        choices=["broadcast", "slot", "switch"],
        default="broadcast",
        help="broadcast = 32-tile write + maskpoll verify; "
        "slot = one tile, write and read back; "
        "switch = stream a slot and execute it",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="units of work: dispatches for broadcast/slot, A,B rounds for "
        "switch (unrolled with --one-run); slot --verify-all ignores it",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=None,
        help="number of 4-word control packets in the write stream "
        "(default 768 broadcast, 14 slot, from the slot image for switch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="slot: send the whole write stream in one BD; "
        "without it, one BD per packet, capped at 14 by the 16-BD channel",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="slot: read back every word of the slot",
    )
    parser.add_argument(
        "--read-batch",
        type=int,
        default=15,
        help="slot: read requests per pass for --verify-all (at most 15)",
    )
    parser.add_argument(
        "--image",
        choices=["variant", "resident"],
        default="variant",
        help="slot: stream a second kernel from the variant builder "
        "(default) or the resident core's own .text",
    )
    parser.add_argument(
        "--one-run",
        action="store_true",
        help="switch: run all rounds in one dispatch instead of one per round",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    if args.count is None:
        args.count = 4 if args.design == "switch" else 1
    if args.packets is not None and args.packets < 1:
        raise SystemExit("--packets must be positive")
    if not 1 <= args.read_batch <= 15:
        raise SystemExit("--read-batch must be in [1, 15] (16 BDs per channel)")
    if (
        args.design == "slot"
        and not args.batch
        and not args.verify_all
        and args.packets is not None
        and args.packets > SLOT_DEFAULT_PACKETS
    ):
        raise SystemExit(
            f"slot without --batch uses one BD per packet and tops out at "
            f"{SLOT_DEFAULT_PACKETS}; pass --batch for more"
        )
    if args.count < 1:
        raise SystemExit("--count must be positive")

    print(
        f"building design={args.design} count={args.count} " f"packets={args.packets}",
        flush=True,
    )
    work = build(
        args.design,
        args.packets,
        args.batch,
        args.verify_all,
        args.read_batch,
        args.image,
        args.count,
        args.one_run,
    )
    print(f"artifacts in {work}", flush=True)
    if args.no_run:
        return 0
    return subprocess.run(
        [str(work / "overlay_packets"), str(args.count), str(work)]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
