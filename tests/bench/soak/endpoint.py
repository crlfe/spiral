#!/usr/bin/env python3
"""Soak any local OpenAI-compatible endpoint and report t/s, Tctl and fan.

This is the same measurement as `soak.py`, but the load is an HTTP request
instead of `llama-completion`, so different servers and frontends can be
compared on the same machine state.

usage: endpoint.py URL [tokens] [--model NAME] [--api completions|chat]
                       [--label NAME] [--ignore-eos] [--allow-drift]
                       [--max-cool-back SECONDS]

examples:
  endpoint.py http://127.0.0.1:8080 3000
  endpoint.py http://127.0.0.1:8080 300 --api chat --label llama-server
  endpoint.py http://127.0.0.1:11434 300 --label ollama
  endpoint.py http://127.0.0.1:52625 300 --max-cool-back 60   # quick look

The decode rate is measured from the first streamed token to the last, which
matches the `eval time` that `soak.py` reads out of llama-completion.  Prefill
time is reported separately.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request

import hostconfig
import sensors
from xdna import Xdna

PROMPT = "Write a detailed guide to the history of the Roman Empire."
PLATEAU_WINDOW = 20.0


def usage():
    raise SystemExit(__doc__.strip())


def parse_args(argv):
    opts = {"api": "completions", "label": None, "model": None,
            "ignore_eos": False, "allow_drift": False, "cool_back_max": 420}
    positional = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--api":
            i += 1
            opts["api"] = argv[i]
        elif arg == "--label":
            i += 1
            opts["label"] = argv[i]
        elif arg == "--model":
            i += 1
            opts["model"] = argv[i]
        elif arg == "--ignore-eos":
            opts["ignore_eos"] = True
        elif arg == "--allow-drift":
            opts["allow_drift"] = True
        elif arg == "--max-cool-back":
            i += 1
            opts["cool_back_max"] = int(argv[i])
        elif arg in ("-h", "--help"):
            usage()
        elif arg.startswith("--"):
            raise SystemExit(f"unknown option {arg}")
        else:
            positional.append(arg)
        i += 1
    if not positional:
        usage()
    base = positional[0].rstrip("/")
    tokens = int(positional[1]) if len(positional) > 1 else 3000
    if opts["api"] not in ("completions", "chat"):
        raise SystemExit("--api must be completions or chat")
    return base, tokens, opts


def _post(url, payload, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def pick_model(base):
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as resp:
        body = json.loads(resp.read())
    data = body.get("data") or []
    return data[0]["id"] if data else "model"


def stream(base, api, model, prompt, tokens, ignore_eos):
    """Run one request; return (token_count, start, first, last)."""
    start = time.monotonic()
    if api == "chat":
        path = f"{base}/v1/chat/completions"
        payload = {"model": model, "messages": [{"role": "user",
                                                 "content": prompt}],
                   "max_tokens": tokens, "temperature": 0, "stream": True}
    else:
        path = f"{base}/v1/completions"
        payload = {"model": model, "prompt": prompt, "max_tokens": tokens,
                   "temperature": 0, "stream": True}
    if ignore_eos:
        payload["ignore_eos"] = True

    counted = 0
    usage_tokens = None
    first = last = None
    resp = _post(path, payload)
    ctype = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in ctype:
        body = json.loads(resp.read())
        usage_tokens = (body.get("usage") or {}).get("completion_tokens")
        first = last = time.monotonic()
        return usage_tokens or 1, start, first, last

    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        now = time.monotonic()
        if first is None:
            first = now
        last = now
        usage = obj.get("usage") or {}
        if usage.get("completion_tokens"):
            usage_tokens = usage["completion_tokens"]
        choice = (obj.get("choices") or [{}])[0]
        text = choice.get("text")
        if text is None:
            text = (choice.get("delta") or {}).get("content")
        if text:
            counted += 1
    resp.close()
    if first is None:
        first = last = time.monotonic()
    return (usage_tokens or counted or 1), start, first, last


def main(argv):
    base, tokens, opts = parse_args(argv)
    label = opts["label"] or base.split("://", 1)[-1].rstrip("/")

    paths = hostconfig.sensor_paths()
    CPU_C = paths["cpu_c"]
    IGPU_C = paths["igpu_c"]
    IGPU_MHZ = paths["igpu_mhz"]
    FANS = paths["fans"]

    peak_fan = [0] * len(FANS)

    def read_fans():
        values = sensors.fans(FANS)
        for i, value in enumerate(values):
            peak_fan[i] = max(peak_fan[i], value)
        return max(values, default=0)

    hostconfig.require(opts["allow_drift"])

    model = opts["model"] or pick_model(base)
    print(f"endpoint {base}  model {model}  api {opts['api']}  "
          f"max_tokens {tokens}", flush=True)

    cool0 = sensors.wait_cool(CPU_C, FANS)
    base_t = (sensors.rd(CPU_C) or 0) / 1000
    print(f"mode={label} idle_Tctl={base_t:.1f}C "
          f"fan={sensors.fan(FANS)} cooldown={cool0}", flush=True)

    result = {}

    def worker():
        try:
            result["r"] = stream(base, opts["api"], model, PROMPT, tokens,
                                 opts["ignore_eos"])
        except (urllib.error.URLError, OSError, ValueError) as exc:
            result["error"] = exc

    xdna = Xdna()
    rows = []
    thread = threading.Thread(target=worker, daemon=True)
    t0 = time.monotonic()
    thread.start()
    while thread.is_alive():
        rows.append((time.monotonic(), sensors.rd(CPU_C), sensors.rd(IGPU_C),
                     read_fans(), *xdna.clocks(),
                     (sensors.rd(IGPU_MHZ) or 0) // 1000000))
        time.sleep(0.5)
    t_end = time.monotonic()

    if "error" in result:
        print(f"request failed: {result['error']}", file=sys.stderr)
        return 1
    n, start, first, last = result["r"]

    while True:
        now = time.monotonic()
        rows.append((now, sensors.rd(CPU_C), sensors.rd(IGPU_C),
                     sensors.fan(FANS), *xdna.clocks(),
                     (sensors.rd(IGPU_MHZ) or 0) // 1000000))
        if (now - t_end > opts["cool_back_max"] or
                ((sensors.rd(CPU_C) or 0) / 1000 <= base_t + 2 and
                 sensors.fan(FANS) == 0)):
            break
        time.sleep(1.0)
    cool_back = time.monotonic() - t_end

    decode = last - first
    tps = n / decode if decode > 0 else float("nan")
    prefill = first - start

    run_end = max(i for i, r in enumerate(rows) if r[0] <= t_end)
    run = rows[:run_end + 1]
    times = [r[0] - t0 for r in run]
    temps = [(r[1] or 0) / 1000 for r in run]
    igpu_t = [(r[2] or 0) / 1000 for r in run]
    fans = [r[3] for r in run]
    pi = sensors.plateau(times, temps, fans)
    steady = (f"plateau {times[pi]:5.1f}s  steady {times[-1] - times[pi]:5.1f}s"
              if pi is not None else f"plateau not reached in {times[-1]:5.1f}s")

    fans_str = "/".join(map(str, peak_fan))
    print(f"{label:<12} {t_end - t0:6.1f}s  {tps:5.2f} t/s  {n} tok  "
          f"prefill {prefill*1000:6.0f} ms  Tctl {max(temps):5.1f}C  "
          f"iGPU {max(igpu_t):5.1f}C  fan {fans_str:>9}  |  {steady}  |  "
          f"cool-back {cool_back:5.1f}s", flush=True)
    mp = [r[4] for r in run if r[4] is not None]
    hh = [r[5] for r in run if r[5] is not None]
    ig = [r[6] for r in run if r[6]]
    if mp:
        print(f"npu: clock mp {min(mp)}-{max(mp)} MHz "
              f"(mean {sum(mp)//len(mp)}), H {min(hh)}-{max(hh)} MHz  |  "
              f"iGPU sclk {min(ig)}-{max(ig)} MHz (mean {sum(ig)//len(ig)})",
              flush=True)
    xdna.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
