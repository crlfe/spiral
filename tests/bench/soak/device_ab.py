#!/usr/bin/env python3
"""One short generation with cooldown, energy and fan sampling.

usage: device_ab.py cpu|npu|igpu [tokens] [--allow-drift]
"""

import os
import re
import subprocess
import sys
import time

import hostconfig
import sensors

HOME = os.path.expanduser("~")
LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR", f"{HOME}/Projects/llama.cpp")
MODEL = os.environ.get("SPIRAL_MODEL",
                       f"{HOME}/Projects/models/gemma-4-E2B_q4_0-it.gguf")
PROMPT = ("Write a detailed guide to the history of the Roman Empire, from its "
          "founding to its fall. Include the major emperors, wars, and reforms.")
BUILD_SPIRAL = f"{LLAMA_CPP}/build-spiral/bin"
BUILD_ROCM = f"{LLAMA_CPP}/build-rocm/bin"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
allow_drift = "--allow-drift" in sys.argv[1:]
if not args:
    raise SystemExit("usage: device_ab.py cpu|npu|igpu [tokens] [--allow-drift]")
mode = args[0]
tokens = int(args[1]) if len(args) > 1 else 300

paths = hostconfig.sensor_paths()
IGPU_C = paths["igpu_c"]
CPU_C = paths["cpu_c"]
FANS = paths["fans"]

rd = sensors.rd


PEAK_FAN = [0] * len(FANS)


def read_fans():
    values = sensors.fans(FANS)
    for i, value in enumerate(values):
        PEAK_FAN[i] = max(PEAK_FAN[i], value)
    return max(values, default=0)


def fan():
    return sensors.fan(FANS)


if mode == "cpu":
    cmd = [f"{BUILD_SPIRAL}/llama-completion", "-m", MODEL, "-ngl", "0",
           "-t", "12"]
    env = dict(os.environ)
    cwd = BUILD_SPIRAL
elif mode == "npu":
    cmd = [f"{BUILD_SPIRAL}/llama-completion", "-m", MODEL, "-ngl", "999"]
    env = dict(os.environ, SPIRAL_CPU_THREADS="1")
    cwd = BUILD_SPIRAL
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




def wait_cool(limit=45.0, timeout=300):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if (rd(CPU_C) or 0) / 1000 <= limit and fan() == 0:
            return time.monotonic() - t0
        time.sleep(1.0)
    return None


cool = wait_cool()
print(f"mode={mode} cooldown={cool if cool is None else round(cool)}s", flush=True)
rows = []
t0 = time.monotonic()
proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
while proc.poll() is None:
    rows.append((time.monotonic(), rd(CPU_C), rd(IGPU_C), read_fans()))
    time.sleep(0.25)
out, err = proc.communicate()
t_end = time.monotonic()
while time.monotonic() - t_end < 60:
    rows.append((time.monotonic(), rd(CPU_C), rd(IGPU_C), read_fans()))
    time.sleep(1.0)

m = re.search(r"eval time =.*?/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second",
              err)
n = int(m.group(1)) if m else tokens
tps = float(m.group(3)) if m else float("nan")
dur = t_end - t0
hot = [r for r in rows if r[0] <= t_end]
fan_rows = [r for r in rows if r[3] > 0]
first = (min(r[0] for r in fan_rows) - t0) if fan_rows else -1.0
span = (max(r[0] for r in fan_rows) - min(r[0] for r in fan_rows)
        if fan_rows else 0.0)
print(f"{mode:<5} {dur:6.1f}s  {tps:5.2f} t/s  "
      f"Tctl {max(r[1] for r in hot)/1000:4.1f}C  "
      f"iGPU {max(r[2] for r in hot)/1000:4.1f}C  "
      f"fan {'/'.join(map(str, PEAK_FAN)):>9} "
      f"(first {first:5.1f}s, on {span:5.1f}s)  n={n}")
