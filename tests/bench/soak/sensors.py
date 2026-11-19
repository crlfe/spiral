#!/usr/bin/env python3
"""Sampled signals for the soak harnesses: temperatures, fan RPM, clocks.

Power is deliberately not measured: package RAPL is root-only in this
container, and socket power has to be integrated from a rough sensor.  Fan RPM
and Tctl are the user-visible, low-approximation proxy for the same behaviour.
"""


def rd(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return None


def fans(paths):
    return [rd(p) or 0 for p in paths]


def fan(paths):
    return max(fans(paths), default=0)


def wait_cool(cpu_c, fans, limit=45.0, timeout=420):
    """Block until the machine is at the idle baseline; return seconds, or None."""
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if (rd(cpu_c) or 0) / 1000 <= limit and fan(fans) == 0:
            return time.monotonic() - t0
        time.sleep(1.0)
    return None


def plateau(times, temp, fans, win=20.0, dtemp=0.5, dfan=50):
    """Earliest index where the following `win` seconds stop rising."""
    n = len(times)
    for i in range(n):
        j = i
        while j < n and times[j] - times[i] < win:
            j += 1
        if j >= n:
            return None
        seg_t = temp[i:j]
        seg_f = fans[i:j]
        if (max(seg_t) - min(seg_t) <= dtemp and
                max(seg_f) - min(seg_f) <= dfan):
            return i
    return None
