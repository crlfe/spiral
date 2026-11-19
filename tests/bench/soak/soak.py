#!/usr/bin/env python3
"""Steady-state thermal soak: run until temps/fan plateau, then cool to idle.

The machine configuration is verified first (see hostconfig.py) so results stay
comparable with the baseline table.

usage: soak.py cpu|npu|igpu [tokens] [--allow-drift]
"""

import os
import re
import subprocess
import sys
import time

import hostconfig
import sensors
from xdna import Xdna

HOME = os.path.expanduser("~")
LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR", f"{HOME}/Projects/llama.cpp")
MODEL = os.environ.get("SPIRAL_MODEL",
                       f"{HOME}/Projects/models/gemma-4-E2B_q4_0-it.gguf")
PROMPT = "Write a detailed guide to the history of the Roman Empire."
BUILD_SPIRAL = f"{LLAMA_CPP}/build-spiral/bin"
BUILD_ROCM = f"{LLAMA_CPP}/build-rocm/bin"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
allow_drift = "--allow-drift" in sys.argv[1:]
if not args:
    raise SystemExit("usage: soak.py cpu|npu|igpu [tokens] [--allow-drift]")
mode = args[0]
tokens = int(args[1]) if len(args) > 1 else 3000

paths = hostconfig.sensor_paths()
IGPU_C = paths["igpu_c"]
IGPU_MHZ = paths["igpu_mhz"]
CPU_C = paths["cpu_c"]
FANS = paths["fans"]

rd = sensors.rd


def fan():
    return sensors.fan(FANS)


PEAK_FAN = [0] * len(FANS)


def read_fans():
    """Fan values for this sample, updating the per-fan peaks."""
    values = sensors.fans(FANS)
    for i, value in enumerate(values):
        PEAK_FAN[i] = max(PEAK_FAN[i], value)
    return max(values, default=0)


if mode == "cpu":
    cmd = [f"{BUILD_SPIRAL}/llama-completion", "-m", MODEL, "-ngl", "0",
           "-t", "12"]
    env, cwd = dict(os.environ), BUILD_SPIRAL
elif mode == "npu":
    cmd = [f"{BUILD_SPIRAL}/llama-completion", "-m", MODEL, "-ngl", "999"]
    env, cwd = dict(os.environ, SPIRAL_CPU_THREADS="1"), BUILD_SPIRAL
elif mode == "igpu":
    cmd = [f"{BUILD_ROCM}/llama-completion", "-m", MODEL, "-ngl", "999"]
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/core-10.0/lib:/opt/rocm/lib64"
    cwd = BUILD_ROCM
else:
    raise SystemExit("mode must be cpu|npu|igpu")

cmd += ["-no-cnv", "--ignore-eos", "-n", str(tokens), "--temp", "0",
        "-p", PROMPT]

hostconfig.require(allow_drift)


cool0 = sensors.wait_cool(CPU_C, FANS)
base_t = (rd(CPU_C) or 0) / 1000
print(f"mode={mode} idle_Tctl={base_t:.1f}C fan={fan()} cooldown={cool0}", flush=True)

xdna = Xdna()
rows = []
t0 = time.monotonic()
proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE, text=True)
while proc.poll() is None:
    rows.append((time.monotonic(), rd(CPU_C), rd(IGPU_C), read_fans(),
                 *xdna.clocks(), (rd(IGPU_MHZ) or 0) // 1000000))
    time.sleep(0.5)
err = proc.stderr.read()
t_end = time.monotonic()

# post-run cooldown, sampled until fans are off and Tctl is back near idle
while True:
    now = time.monotonic()
    rows.append((now, rd(CPU_C), rd(IGPU_C), read_fans(),
                 *xdna.clocks(), (rd(IGPU_MHZ) or 0) // 1000000))
    if (now - t_end > 420 or
            ((rd(CPU_C) or 0) / 1000 <= base_t + 2 and fan() == 0)):
        break
    time.sleep(1.0)
cool_back = time.monotonic() - t_end

m = re.search(r"eval time =.*?/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second",
              err)
n = int(m.group(1)) if m else tokens
tps = float(m.group(3)) if m else float("nan")

run_end = max(i for i, r in enumerate(rows) if r[0] <= t_end)
dur = rows[run_end][0] - rows[0][0]
run = rows[:run_end + 1]

times = [r[0] - t0 for r in run]
temps = [(r[1] or 0) / 1000 for r in run]
igpu_t = [(r[2] or 0) / 1000 for r in run]
fans = [r[3] for r in run]

pi = sensors.plateau(times, temps, fans)
steady = (f"plateau {times[pi]:5.1f}s  steady {times[-1] - times[pi]:5.1f}s"
          if pi is not None else f"plateau not reached in {dur:5.1f}s")

fans_str = "/".join(map(str, PEAK_FAN))
print(f"{mode:<5} {dur:6.1f}s  {tps:5.2f} t/s  Tctl {max(temps):5.1f}C  "
      f"iGPU {max(igpu_t):5.1f}C  fan {fans_str:>9}  |  {steady}  |  "
      f"cool-back {cool_back:5.1f}s", flush=True)
mp = [r[4] for r in run if r[4] is not None]
hh = [r[5] for r in run if r[5] is not None]
ig = [r[6] for r in run if r[6]]
if mp:
    print(f"npu: clock mp {min(mp)}-{max(mp)} MHz (mean {sum(mp)//len(mp)}), "
          f"H {min(hh)}-{max(hh)} MHz  |  iGPU sclk {min(ig)}-{max(ig)} MHz "
          f"(mean {sum(ig)//len(ig)})", flush=True)
xdna.close()
