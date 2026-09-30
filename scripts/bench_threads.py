#!/usr/bin/env python3
"""Compare CPU thread counts for the default (CPU/XNNPACK) pipeline on a connected phone.

    scripts/bench_threads.py --threads 4 6 8 --reps 3 [--serial R5CY...]

Every run starts from a cooled device: before each generation the script waits until the AP
temperature reported by the thermal HAL is back within --cool-margin of the idle baseline and the
thermal status is 0. Thread counts are interleaved (a Latin square over the repetitions), so
residual heat cannot favour one of them. Needs the app installed and its models + weight cache
already in place (one generation done). Rows go to stdout and out/bench_threads.csv.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import time

PKG = "io.github.alexeyw.zimage"
PROMPT = "a cozy cabin in a snowy forest at night, warm light in the windows"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ADB = shutil.which("adb") or os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")


def adb(serial, *args, timeout=60):
    cmd = [ADB] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def thermal(serial):
    """-> (thermal status, current AP temperature from the HAL)."""
    out = adb(serial, "shell", "dumpsys", "thermalservice")
    status = int(re.search(r"Thermal Status: (\d+)", out).group(1))
    hal = out.split("Current temperatures from HAL:", 1)[1]
    ap = float(re.search(r"mValue=([\d.]+), mType=0, mName=AP,", hal).group(1))
    return status, ap


def cool_down(serial, target, max_wait):
    t0 = time.time()
    while True:
        status, ap = thermal(serial)
        if (status == 0 and ap <= target) or time.time() - t0 > max_wait:
            return ap, time.time() - t0
        time.sleep(10)


def run_once(serial, threads, seed, steps):
    adb(serial, "shell", "am", "force-stop", PKG)
    adb(serial, "logcat", "-c")
    adb(serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP")
    adb(serial, "shell", "am", "start", "-n", f"{PKG}/.MainActivity", "--ez", "autorun", "true",
        "--es", "prompt", f"'{PROMPT}'", "--el", "seed", str(seed), "--ei", "steps", str(steps),
        "--es", "backend", "CPU", "--ei", "threads", str(threads))
    deadline = time.time() + 600
    while time.time() < deadline:
        log = adb(serial, "logcat", "-d", "-s", "ZImage:V", "AndroidRuntime:E")
        if "generated " in log or "FATAL" in log or "generation failed" in log:
            break
        time.sleep(3)
    total = re.search(r"generated \S+ in ([\d,.]+)s", log)
    if not total:
        raise RuntimeError(f"no result for threads={threads}:\n{log[-2000:]}")
    runs = {}
    for name, run in re.findall(r"ZImage\s*:\s*(\w+)\s+×\d+\s+load [\d,.]+s\s+run ([\d,.]+)s", log):
        runs[name] = float(run.replace(",", "."))
    dit = sum(v for k, v in runs.items() if k.startswith(("zc_", "z_refx", "z_embx")))
    return float(total.group(1).replace(",", ".")), runs.get("qwen_enc", 0.0), dit, runs.get("zvae", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--serial")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cool-margin", type=float, default=3.0, help="°C above the idle AP baseline")
    ap.add_argument("--max-wait", type=float, default=420, help="seconds to wait for cooling")
    a = ap.parse_args()

    _, baseline = thermal(a.serial)
    target = baseline + a.cool_margin
    n = len(a.threads)
    order = [a.threads[(r + i) % n] for r in range(a.reps) for i in range(n)]  # Latin square
    print(f"idle AP {baseline:.1f} °C, cool-down target {target:.1f} °C, order {order}", flush=True)

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    path = os.path.join(ROOT, "out", "bench_threads.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "ap_start_c", "waited_s", "total_s", "text_s", "dit_s", "vae_s"])
        for threads in order:
            ap_start, waited = cool_down(a.serial, target, a.max_wait)
            total, text, dit, vae = run_once(a.serial, threads, a.seed, a.steps)
            row = [threads, f"{ap_start:.1f}", f"{waited:.0f}", f"{total:.1f}", f"{text:.1f}", f"{dit:.1f}", f"{vae:.1f}"]
            w.writerow(row)
            f.flush()
            print("threads={} ap_start={}°C waited={}s total={}s text={}s dit={}s vae={}s".format(*row), flush=True)
    print("csv:", path)


if __name__ == "__main__":
    main()
