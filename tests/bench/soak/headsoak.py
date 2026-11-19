#!/usr/bin/env python3
"""Sample temps/fans/clocks around an NPU-only busy loop (test --head-soak).

usage: headsoak.py [iters] [--allow-drift]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import hostconfig
import sensors
from xdna import Xdna

REPO = Path(__file__).resolve().parents[3]
BIN = Path(os.environ.get("SPIRAL_PROGRAM_BUILD_DIR",
                          REPO / "src/program/build-production"))

args = [a for a in sys.argv[1:] if not a.startswith("--")]
allow_drift = "--allow-drift" in sys.argv[1:]
iters = int(args[0]) if args else 20000

paths = hostconfig.sensor_paths()
IGPU_C = paths["igpu_c"]
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


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


hostconfig.require(allow_drift)

xdna = Xdna()
rows = []
t0 = time.monotonic()
proc = subprocess.Popen(["./test", "--head-soak", str(iters)], cwd=BIN,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True)
while proc.poll() is None:
    rows.append((time.monotonic(), rd(CPU_C), rd(IGPU_C), read_fans(),
                 *cpu_times(), *xdna.clocks()))
    time.sleep(0.5)
out, err = proc.communicate()
t_end = time.monotonic()
while time.monotonic() - t_end < 60:
    rows.append((time.monotonic(), rd(CPU_C), rd(IGPU_C), fan(),
                 *cpu_times(), *xdna.clocks()))
    time.sleep(1.0)

run_end = max(i for i, r in enumerate(rows) if r[0] <= t_end)
dur = rows[run_end][0] - rows[0][0]
fans_str = "/".join(map(str, PEAK_FAN))
busy = 1.0 - (rows[run_end][5] - rows[0][5]) / max(
    1, rows[run_end][4] - rows[0][4])
run = rows[:run_end + 1]
print(f"head-soak {dur:6.1f}s  busy {busy*100:4.0f}%  "
      f"Tctl {max(r[1] for r in run)/1000:5.1f}C  "
      f"iGPU {max(r[2] for r in run)/1000:5.1f}C  "
      f"fan {fans_str:>9}", flush=True)
mp = [r[6] for r in run if r[6] is not None]
hh = [r[7] for r in run if r[7] is not None]
if mp:
    print(f"npu: clock mp {min(mp)}-{max(mp)} MHz (mean {sum(mp)//len(mp)}), "
          f"H {min(hh)}-{max(hh)} MHz")
xdna.close()
print(err.strip()[-400:])
