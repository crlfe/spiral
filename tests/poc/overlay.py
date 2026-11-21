#!/usr/bin/env python3
"""Prototype: reload compute-tile program memory without reconfiguring the NPU.

The normal aie-npu-to-cpp path generates a runtime instruction buffer (``insts``)
that is dispatched with an already-loaded xclbin.  This prototype keeps that
xclbin/hw_context alive and prepends a small instruction sequence that:

  1. disables and resets each compute core,
  2. block-writes a new ``.text`` (program memory) image at the tile's 0x20000
     window,
  3. releases reset and enables the cores,

then runs the ordinary generated runtime sequence.  Switching between two
different kernels therefore costs only the program-memory bytes, not a full
PDI reload.

Usage::

    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py --pad 0
    ~/Projects/mlir-aie/ironenv/bin/python tests/poc/overlay.py --tiles 4

The ``--pad`` value controls the generated ``.text`` size (0 is a tiny kernel;
380 is roughly a production-sized 10.5 KiB image).  The script builds two
variants that differ only in a compile-time addend, extracts each core's
program-memory image from the per-core ELF that aiecc emits, compiles a small
C++ harness, and alternates the two overlays on device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    RuntimeData,
    Worker,
)
from aie.iron.device import NPU2, Tile
from aie.utils import config as aie_config
from aie.utils.compile import compile_external_kernels, compile_mlir_module

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE / "build" / "overlay"

LEN = 64
ELEM = np.int32
TENSOR = np.ndarray[(LEN,), np.dtype[ELEM]]

# AIE2P / NPU2 per-tile address map.
COL_SHIFT = 25
ROW_SHIFT = 20
PROGRAM_MEMORY = 0x20000
CORE_CONTROL = 0x32000
PM_SIZE = 0x4000


# ---------------------------------------------------------------------------
# Design


def make_kernel(name: str, delta: int, pad: int) -> ExternalFunction:
    """A tiny vector add whose .text can be padded to a chosen size.

    The padding function is never called (``pad_enable`` is a zero-initialised
    global), but it is reachable so the linker retains it.  This gives us a
    controlled program-memory image without changing runtime behaviour.
    """
    return ExternalFunction(
        name,
        arg_types=[TENSOR, TENSOR],
        inline=True,
        source_string=f"""
#include <aie_api/aie.hpp>

static volatile int32_t pad_enable = 0;

__attribute__((noinline)) void pad(int32_t *p, int32_t x) {{
#pragma clang loop unroll(full)
    for (int32_t i = 0; i < {pad}; i++) {{
        x = x * 1664525 + 1013904223 + i;
        p[i & 63] ^= x;
    }}
}}

extern "C" void {name}(const int32_t * restrict in, int32_t * restrict out) {{
    constexpr int32_t N = 16;
    if (pad_enable) pad(out, in[0]);
    for (int32_t i = 0; i < {LEN}; i += N) {{
        aie::vector<int32_t, N> v = aie::load_v<N>(in + i);
        v = aie::add(v, aie::broadcast<int32_t, N>({delta}));
        aie::store_v(out + i, v);
    }}
}}
""",
    )


def build_mlir(delta: int, pad: int, tiles: int):
    kernels = [
        make_kernel(f"compute_{i}_{delta}_{pad}", delta, pad) for i in range(tiles)
    ]
    block_type = np.ndarray[(LEN * tiles,), np.dtype[ELEM]]

    # One host input block split across the workers, and one joined output
    # block.  This mirrors the production topology and keeps the xclbin path
    # at three host buffers regardless of the number of tiles.
    inputs = ObjectFifo(block_type, depth=1, name="inp")
    l1_inputs = inputs.cons().split(
        offsets=[LEN * i for i in range(tiles)],
        obj_types=[TENSOR] * tiles,
        names=[f"inp_{i}" for i in range(tiles)],
    )
    outputs = ObjectFifo(block_type, depth=1, name="outp")
    l1_outputs = outputs.prod().join(
        offsets=[LEN * i for i in range(tiles)],
        obj_types=[TENSOR] * tiles,
        names=[f"outp_{i}" for i in range(tiles)],
    )

    def compute(inp, outp, kernel):
        ib = inp.acquire(1)
        ob = outp.acquire(1)
        kernel(ib, ob)
        inp.release(1)
        outp.release(1)

    # Pin one worker per column, like the production 8x4 design.
    workers = [
        Worker(
            compute,
            [l1_inputs[i].cons(), l1_outputs[i].prod(), kernels[i]],
            stack_size=2048,
            tile=Tile(i, 2),
        )
        for i in range(tiles)
    ]

    def sequence(inp: RuntimeData, out: RuntimeData, inputs_prod, outputs_cons):
        inputs_prod.fill(inp)
        outputs_cons.drain(out, wait=True)

    rt = Runtime(
        sequence,
        [block_type, block_type, inputs.prod(), outputs.cons()],
    )
    return Program(NPU2(), rt, workers).resolve_program()


# ---------------------------------------------------------------------------
# Build helpers


def _env() -> dict:
    env = os.environ.copy()
    env["PATH"] = (
        str(Path(aie_config.peano_install_dir()) / "bin")
        + ":"
        + str(Path(aie_config.aiecc_path()).parent)
        + ":"
        + env.get("PATH", "")
    )
    return env


def _tool(*args, cwd: Path, quiet: bool = False):
    try:
        subprocess.run(
            list(map(str, args)),
            check=True,
            cwd=cwd,
            env=_env(),
            capture_output=quiet,
            text=quiet,
        )
    except subprocess.CalledProcessError as err:
        if quiet:
            sys.stderr.write(err.stdout or "")
            sys.stderr.write(err.stderr or "")
        raise


def _parse_core_elf(path: Path) -> bytes:
    """Return the loadable program-memory bytes (p_paddr < PM_SIZE)."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1:
        raise ValueError(f"{path}: expected ELF32")
    e_phoff = struct.unpack_from("<I", data, 28)[0]
    e_phentsize = struct.unpack_from("<H", data, 42)[0]
    e_phnum = struct.unpack_from("<H", data, 44)[0]
    segments = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_offset, _vaddr, p_paddr, p_filesz, _memsz, _flags, _align = (
            struct.unpack_from("<IIIIIIII", data, off)
        )
        if p_type == 1 and p_paddr < PM_SIZE:
            payload = data[p_offset : p_offset + p_filesz]
            if len(payload) % 4:
                payload += b"\0" * (4 - len(payload) % 4)
            segments.append((p_paddr, payload))
    if len(segments) != 1 or segments[0][0] != 0:
        raise ValueError(
            f"{path}: expected exactly one program-memory segment at address 0, "
            f"got {[(hex(a), len(b)) for a, b in segments]}"
        )
    if len(segments[0][1]) > PM_SIZE:
        raise ValueError(f"{path}: .text overflows the {PM_SIZE}-byte PM")
    return segments[0][1]


def build_variant(
    name: str, delta: int, pad: int, tiles: int, quiet: bool = False
) -> dict:
    work = WORK_ROOT / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    ExternalFunction._instances.clear()
    mlir = build_mlir(delta, pad, tiles)
    compile_external_kernels(ExternalFunction._instances, work, "aie2p")
    compile_mlir_module(mlir, work_dir=work, device=NPU2())

    _tool(
        "aie-opt",
        "--aie-substitute-shim-dma-allocations",
        "--aie-assign-runtime-sequence-bd-ids",
        "--aie-dma-tasks-to-npu",
        "--aie-dma-to-npu",
        "-o",
        "lowered.mlir",
        "input_with_addresses.mlir",
        cwd=work,
        quiet=quiet,
    )
    _tool(
        "aie-translate",
        "--aie-npu-to-cpp",
        "-o",
        "generate_txn.h",
        "lowered.mlir",
        cwd=work,
        quiet=quiet,
    )
    _tool(
        "aiecc",
        "--get-xclbin",
        "--tmpdir=xclbin_build",
        "lowered.mlir",
        cwd=work,
        quiet=quiet,
    )

    # Patch generated overflow returns into throws, as invoke.py does.
    txn_path = work / "generate_txn.h"
    src = txn_path.read_text()
    patched, count = re.subn(
        r"return\s+std::nullopt\s*;",
        'throw std::runtime_error(std::string("Overflow at ") + __FILE__ + '
        '":" + std::to_string(__LINE__));',
        src,
    )
    if count:
        txn_path.write_text("#include <stdexcept>\n" + patched)

    overlays = {}
    for elf in sorted((work / "xclbin_build").glob("elfs_*_core_*_*/*.elf")):
        m = re.search(r"_core_(\d+)_(\d+)/", str(elf))
        if not m:
            continue
        col, row = int(m.group(1)), int(m.group(2))
        payload = _parse_core_elf(elf)
        out = work / f"overlay_{col}_{row}.bin"
        out.write_bytes(payload)
        overlays[f"{col},{row}"] = {
            "col": col,
            "row": row,
            "file": str(out),
            "bytes": len(payload),
            "text_elf": str(elf),
        }
    if len(overlays) != tiles:
        raise RuntimeError(
            f"{name}: found {len(overlays)} core ELFs for {tiles} workers"
        )
    (work / "overlays.json").write_text(json.dumps(overlays, indent=2))
    return {"work": work, "overlays": overlays}


# ---------------------------------------------------------------------------
# C++ harness


CC_TEMPLATE = r"""
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "generate_txn.h"

static constexpr int32_t kLen = @LEN@;
static constexpr int32_t kTiles = @NTILES@;
static constexpr uint32_t kColShift = @COL_SHIFT@;
static constexpr uint32_t kRowShift = @ROW_SHIFT@;
static constexpr uint32_t kPmBase = @PM@u;
static constexpr uint32_t kCoreCtrl = @CORE_CTRL@u;

struct Tile {
  int32_t col, row;
  std::vector<uint32_t> overlay;
};

static const int32_t kTileCols[] = @TILE_COLS@;
static const int32_t kTileRows[] = @TILE_ROWS@;

static std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint32_t> out(n / 4);
  f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size() * 4);
  return out;
}

static uint32_t tile_addr(const Tile& t, uint32_t local) {
  return (uint32_t(t.col) << kColShift) | (uint32_t(t.row) << kRowShift) | local;
}

// Prepend overlay loads to the generated runtime sequence.  The generated
// sequence ends in a DMA await, so the core must already be running before it.
static std::vector<uint32_t> build_txn(const std::vector<Tile>& tiles,
                                       const std::vector<uint32_t>& base) {
  std::vector<uint32_t> txn;
  aie_runtime::txn_init(txn);
  for (const Tile& t : tiles) {
    aie_runtime::txn_append_write32(txn, tile_addr(t, kCoreCtrl), 0x0);
    aie_runtime::txn_append_write32(txn, tile_addr(t, kCoreCtrl), 0x2);
  }
  for (const Tile& t : tiles)
    aie_runtime::txn_append_blockwrite(txn, tile_addr(t, kPmBase),
                                       t.overlay.data(), t.overlay.size(),
                                       t.col, t.row);
  for (const Tile& t : tiles) {
    aie_runtime::txn_append_write32(txn, tile_addr(t, kCoreCtrl), 0x0);
    aie_runtime::txn_append_write32(txn, tile_addr(t, kCoreCtrl), 0x1);
  }
  uint32_t overlay_ops = uint32_t(tiles.size() * 5);
  txn.insert(txn.end(), base.begin() + 4, base.end());
  txn[0] = base[0];
  txn[1] = base[1];
  txn[2] = base[2] + overlay_ops;
  txn[3] = uint32_t(txn.size() * sizeof(uint32_t));
  return txn;
}

struct Bos {
  xrt::bo instr, in, out;
};

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::stoi(argv[1]) : 16;
  std::string dir = argc > 2 ? argv[2] : ".";

  std::vector<Tile> a, b;
  for (int i = 0; i < kTiles; i++) {
    char ca[512], cb[512];
    snprintf(ca, sizeof(ca), "%s/overlay_a_%d.bin", dir.c_str(), i);
    snprintf(cb, sizeof(cb), "%s/overlay_b_%d.bin", dir.c_str(), i);
    a.push_back({kTileCols[i], kTileRows[i], read_words(ca)});
    b.push_back({kTileCols[i], kTileRows[i], read_words(cb)});
  }

  auto base_opt = generate_txn_main_sequence();
  if (!base_opt) throw std::runtime_error("base txn generation failed");
  auto base = *base_opt;
  auto txn_a = build_txn(a, base);
  auto txn_b = build_txn(b, base);

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(dir + "/aie.xclbin");
  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid(),
                                 xrt::hw_context::access_mode::exclusive);
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  Bos bos;
  size_t max_txn = std::max(txn_a.size(), txn_b.size());
  bos.instr = xrt::bo(device, max_txn * sizeof(uint32_t),
                      xrt::bo::flags::cacheable, kernel.group_id(1));
  bos.in = xrt::bo(device, (size_t)kLen * kTiles * sizeof(int32_t),
                   xrt::bo::flags::host_only, kernel.group_id(3));
  bos.out = xrt::bo(device, (size_t)kLen * kTiles * sizeof(int32_t),
                    xrt::bo::flags::host_only, kernel.group_id(4));
  auto pin = bos.in.map<int32_t*>();
  for (int j = 0; j < kLen * kTiles; j++) pin[j] = j;
  bos.in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run_once = [&](const std::vector<uint32_t>& txn, int32_t delta,
                      bool verify) -> double {
    bos.instr.write(txn.data(), txn.size() * sizeof(uint32_t), 0);
    bos.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto r = xrt::run(kernel);
    r.set_arg(0, (uint32_t)3);
    r.set_arg(1, bos.instr);
    r.set_arg(2, (uint32_t)txn.size());
    r.set_arg(3, bos.in);
    r.set_arg(4, bos.out);
    auto t0 = std::chrono::steady_clock::now();
    r.start();
    r.wait2(std::chrono::milliseconds(5000));
    auto t1 = std::chrono::steady_clock::now();
    if (verify) {
      int errors = 0;
      bos.out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto p = bos.out.map<int32_t*>();
      for (int j = 0; j < kLen * kTiles; j++) {
        int32_t want = j + delta;
        if (p[j] != want) {
          if (errors < 5)
            printf("  [%d] (tile %d): got %d want %d\n", j, j / kLen, p[j], want);
          errors++;
        }
      }
      printf("  verify delta %+d: %s\n", delta, errors ? "FAIL" : "ok");
      if (errors) throw std::runtime_error("verification failed");
    }
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };

  run_once(txn_a, 1, true);
  run_once(txn_b, 2, true);
  run_once(txn_a, 1, true);

  double ta = 0, tb = 0;
  for (int i = 0; i < iters; i++) {
    ta += run_once(txn_a, 1, false);
    tb += run_once(txn_b, 2, false);
  }
  size_t bytes = 0;
  for (const Tile& t : a) bytes += t.overlay.size() * 4;
  printf("  reload A: %.1f us   reload B: %.1f us   overlays: %zu B/tile, %zu B total\n",
         ta / iters, tb / iters, bytes / a.size(), bytes);
  return 0;
}
"""


def write_harness(variant: dict, tiles: int, out: Path) -> None:
    overlays = list(variant["overlays"].values())
    cols = ", ".join(str(o["col"]) for o in overlays)
    rows = ", ".join(str(o["row"]) for o in overlays)
    text = (
        CC_TEMPLATE.replace("@LEN@", str(LEN))
        .replace("@NTILES@", str(tiles))
        .replace("@COL_SHIFT@", str(COL_SHIFT))
        .replace("@ROW_SHIFT@", str(ROW_SHIFT))
        .replace("@PM@", hex(PROGRAM_MEMORY))
        .replace("@CORE_CTRL@", hex(CORE_CONTROL))
        .replace("@TILE_COLS@", "{" + cols + "}")
        .replace("@TILE_ROWS@", "{" + rows + "}")
    )
    out.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--pad", type=int, default=380)
    parser.add_argument("--iters", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.tiles <= 6:
        raise SystemExit(
            "--tiles must be in [1, 6]: this prototype uses one split/join "
            "L2 block, and a mem tile has six DMA channels per direction"
        )

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"building variant A (+1, pad={args.pad}, tiles={args.tiles})", flush=True)
    a = build_variant("a", 1, args.pad, args.tiles)
    print(f"building variant B (+2, pad={args.pad}, tiles={args.tiles})", flush=True)
    b = build_variant("b", 2, args.pad, args.tiles)
    if a["overlays"].keys() != b["overlays"].keys():
        raise RuntimeError("variants placed workers on different tiles")

    run_dir = WORK_ROOT / "run"
    run_dir.mkdir(exist_ok=True)
    # Base xclbin/transaction come from A; B supplies only program memory.
    for name in ("aie.xclbin", "generate_txn.h"):
        (run_dir / name).write_bytes((a["work"] / name).read_bytes())
    for variant, tag in ((a, "a"), (b, "b")):
        for i, o in enumerate(variant["overlays"].values()):
            (run_dir / f"overlay_{tag}_{i}.bin").write_bytes(
                Path(o["file"]).read_bytes()
            )

    write_harness(a, args.tiles, run_dir / "overlay.cc")
    site = Path(sysconfig.get_paths()["purelib"])
    # The harness is host code, so use the system C++ compiler rather than the
    # Peano/aiecc tools on the modified PATH used for the device build.
    host_cxx = (
        "/usr/bin/clang++"
        if Path("/usr/bin/clang++").exists()
        else (shutil.which("clang++") or "g++")
    )
    subprocess.run(
        [
            host_cxx,
            "-O2",
            "-g",
            f"-I{site / 'mlir_aie' / 'include'}",
            f"-I{run_dir}",
            "-o",
            str(run_dir / "overlay"),
            str(run_dir / "overlay.cc"),
            "-lxrt++",
            "-lxrt_coreutil",
        ],
        check=True,
        cwd=run_dir,
    )

    for o, tag in ((a["overlays"], "A"), (b["overlays"], "B")):
        sizes = sorted({v["bytes"] for v in o.values()})
        print(f"  {tag}: {len(o)} cores, .text {sizes[0]}-{sizes[-1]} bytes")

    print("running on NPU")
    _tool(run_dir / "overlay", str(args.iters), str(run_dir), cwd=run_dir)


if __name__ == "__main__":
    main()
