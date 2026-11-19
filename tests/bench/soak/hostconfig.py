#!/usr/bin/env python3
"""Verify the machine configuration used for power/fan measurements.

Numbers are only comparable when the devices are in the same state, so the
soak harnesses call `require()` at startup and abort on drift.  The baseline is
the machine's normal boot state with two deliberate changes, a 2400 MHz iGPU
SCLK cap and NPU pmode `turbo`:

| setting | value |
| :-- | :-- |
| NPU power mode | turbo (4) |
| iGPU DPM force level | manual |
| iGPU SCLK OD bounds | 600 / 2400 MHz |
| platform profile | balanced |
| CPU governor | powersave on every policy |
| PM QoS | off (`SPIRAL_DMA_LATENCY` unset) |

The NPU power mode is set to `turbo` by the `spiral-npu-tune` unit (it is
worth ~11% on NPU decode and does not survive a reboot).  The fclk/socclk
levels are **not** controlled: pinning them was measured to make no difference,
so they are printed as observations for context and never asserted.

`--allow-drift` reports drift and continues, which is useful for exploratory
runs but must not be used for numbers that go in the notes.  On a mismatch the
restore commands are printed; the harness never runs `sudo` itself.

usage: hostconfig.py [--allow-drift]
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SYS = Path("/sys")

PMODE_NAMES = {0: "default", 1: "low", 2: "medium", 3: "high", 4: "turbo"}

IGPU_DPM = "manual"
IGPU_SCLK = (600, 2400)
PLATFORM_PROFILE = "balanced"
CPU_GOVERNOR = "powersave"
CPU_EPP = "balance_performance"


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hwmon(name):
    """Resolve a hwmon directory by its `name` attribute (indices move)."""
    for entry in sorted(SYS.glob("class/hwmon/hwmon*")):
        if _read(entry / "name") == name:
            return entry
    return None


def igpu_device():
    """The amdgpu PCI device that owns the DPM and OD knobs."""
    for entry in sorted(SYS.glob("class/drm/card[0-9]*/device")):
        if (entry / "power_dpm_force_performance_level").exists():
            return entry
    return None


def fan_paths():
    """The laptop's real fans, in a stable order.

    `hp` exposes the two chassis fans (`fan1`, `fan2`); `acpi_fan` is a
    placeholder that reads 0 here, so it is only used when `hp` is absent.
    """
    for name in ("hp", "acpi_fan"):
        entry = hwmon(name)
        if entry is None:
            continue
        paths = sorted(entry.glob("fan*_input"))
        if paths:
            return paths
    return []


def sensor_paths():
    """Paths the harnesses sample, resolved once at startup."""
    gpu = igpu_device()
    amdgpu = hwmon("amdgpu")
    cpu = hwmon("k10temp")
    if gpu is None or amdgpu is None or cpu is None:
        missing = [n for n, v in (("amdgpu DPM", gpu), ("amdgpu hwmon", amdgpu),
                                  ("k10temp", cpu)) if v is None]
        raise SystemExit(f"hostconfig: missing sensor source: {', '.join(missing)}")
    return {
        "igpu_w": amdgpu / "power1_average",
        "igpu_c": amdgpu / "temp1_input",
        "igpu_mhz": amdgpu / "freq1_input",
        "cpu_c": cpu / "temp1_input",
        "fans": fan_paths(),
        "dpm": gpu / "power_dpm_force_performance_level",
        "od": gpu / "pp_od_clk_voltage",
    }


def _amd_smi_clocks():
    """Current clock level per domain: {domain: (level, mhz)}."""
    try:
        res = subprocess.run(["amd-smi", "static", "--clock"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    current, levels, domain = {}, {}, None
    for raw in res.stdout.splitlines():
        line = raw.strip()
        match = re.match(r"^(SYS|MEM|DF|SOC|DCEF):$", line)
        if match:
            domain = match.group(1)
            levels[domain] = {}
            continue
        if domain is None:
            continue
        match = re.match(r"^CURRENT_LEVEL:\s*(\d+)", line)
        if match:
            current[domain] = int(match.group(1))
            continue
        match = re.match(r"^LEVEL\s+(\d+):\s*(\d+)\s*MHz", line)
        if match:
            levels[domain][int(match.group(1))] = int(match.group(2))
    return {d: (lvl, levels.get(d, {}).get(lvl)) for d, lvl in current.items()}


def _od_sclk_bounds(od_path):
    """The two OD_SCLK entries as (min, max) MHz."""
    text = _read(od_path)
    if text is None:
        return None
    values, inside = [], False
    for line in text.splitlines():
        if line.strip() == "OD_SCLK:":
            inside = True
            continue
        if inside:
            match = re.match(r"^\s*(\d+):\s*(\d+)\s*Mhz", line)
            if not match:
                break
            values.append(int(match.group(2)))
    return (min(values), max(values)) if values else None


def _npu_power_mode():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from xdna import Xdna
    except ImportError:
        return None
    try:
        device = Xdna()
    except OSError:
        return None
    try:
        return device.power_mode()
    finally:
        device.close()


def _cpu_policies():
    return sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))


@dataclass
class Check:
    name: str
    expected: str
    actual: str
    ok: bool
    fix: str | None = None
    severity: str = "error"


def observations():
    """Uncontrolled state, printed for context and never asserted."""
    parts = []
    clocks = _amd_smi_clocks()
    for domain, label in (("DF", "fclk"), ("SOC", "socclk")):
        entry = clocks.get(domain) if clocks else None
        if entry is None:
            parts.append(f"{label} unavailable")
        else:
            parts.append(f"{label} level {entry[0]} ({entry[1]} MHz)")
    print("observed (not controlled): " + ", ".join(parts), flush=True)


def checks():
    sensors = sensor_paths()
    rows = []

    mode = _npu_power_mode()
    rows.append(Check("NPU power mode", "turbo (4)",
                      "unavailable" if mode is None
                      else f"{PMODE_NAMES.get(mode, '?')} ({mode})",
                      mode == 4, "sudo xrt-smi configure --pmode turbo"))

    dpm = _read(sensors["dpm"])
    rows.append(Check("iGPU DPM level", IGPU_DPM, dpm or "unavailable",
                      dpm == IGPU_DPM,
                      f"echo {IGPU_DPM} | sudo tee {sensors['dpm']}"))

    bounds = _od_sclk_bounds(sensors["od"])
    rows.append(Check(
        "iGPU SCLK bounds", f"{IGPU_SCLK[0]}/{IGPU_SCLK[1]} MHz",
        "unavailable" if bounds is None else f"{bounds[0]}/{bounds[1]} MHz",
        bounds == IGPU_SCLK,
        f"echo 's 0 {IGPU_SCLK[0]}' | sudo tee {sensors['od']}; "
        f"echo 's 1 {IGPU_SCLK[1]}' | sudo tee {sensors['od']}; "
        f"echo c | sudo tee {sensors['od']}"))

    profile = _read(SYS / "firmware/acpi/platform_profile")
    rows.append(Check("platform profile", PLATFORM_PROFILE, profile or "unavailable",
                      profile == PLATFORM_PROFILE,
                      f"echo {PLATFORM_PROFILE} | sudo tee "
                      f"{SYS}/firmware/acpi/platform_profile"))

    policies = _cpu_policies()
    governors = {_read(p / "scaling_governor") for p in policies}
    gov_ok = len(governors) == 1 and governors == {CPU_GOVERNOR}
    rows.append(Check(
        "CPU governor", f"{CPU_GOVERNOR} on all {len(policies)} policies",
        ",".join(sorted(g or "?" for g in governors)) or "unavailable", gov_ok,
        f"sudo cpupower frequency-set -g {CPU_GOVERNOR}"))

    epp = {_read(p / "energy_performance_preference") for p in policies}
    epp_ok = len(epp) == 1 and epp <= {CPU_EPP, None}
    rows.append(Check(
        "CPU EPP", f"{CPU_EPP} on all policies",
        ",".join(sorted(e or "?" for e in epp)) or "unavailable", epp_ok,
        f"echo {CPU_EPP} | sudo tee "
        f"/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference",
        severity="warn"))

    qos = os.environ.get("SPIRAL_DMA_LATENCY")
    rows.append(Check("PM QoS", "off (SPIRAL_DMA_LATENCY unset)",
                      f"SPIRAL_DMA_LATENCY={qos}" if qos else "off",
                      not qos, "unset SPIRAL_DMA_LATENCY"))

    return rows


def report(rows):
    print("measurement configuration:", flush=True)
    for row in rows:
        mark = "ok  " if row.ok else ("FAIL" if row.severity == "error"
                                      else "warn")
        print(f"  [{mark}] {row.name:<20} {row.actual:<28} "
              f"expected {row.expected}", flush=True)
    bad = [r for r in rows if not r.ok]
    if not bad:
        return 0
    print("\nconfiguration drift:", flush=True)
    for row in bad:
        print(f"  {row.name}: {row.actual} (expected {row.expected})",
              flush=True)
    fixes = [r.fix for r in bad if r.fix]
    if fixes:
        print("\nrestore with:", flush=True)
        for fix in fixes:
            print(f"  {fix}", flush=True)
    return 2


def require(allow_drift=False):
    """Abort unless the machine matches the measurement baseline."""
    rows = checks()
    status = report(rows)
    observations()
    if status != 0 and not allow_drift:
        print("\nrefusing to measure; fix the configuration or pass "
              "--allow-drift", flush=True)
        raise SystemExit(status)
    return status


def main(argv):
    allow_drift = "--allow-drift" in argv
    status = require(allow_drift)
    if status != 0:
        print("\ncontinuing with drift (results must not be compared to the "
              "notes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
