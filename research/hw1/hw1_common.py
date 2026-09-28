"""HW1 shared plumbing: device-synchronised timing, device/env fingerprint, the results schema.

Every HW1 script emits ONE results file of schema `shard-hw1/1` (validate() below is the contract;
collect.py and calib_fit.py refuse anything that fails it). The schema carries the evidence a
reviewer needs to reject a meaningless number: how it was timed (cuda events + synchronize, or
perf_counter on CPU), how many warm-up calls were discarded, whether the ops were stubs, and which
device family / kernel pins produced it.

Pure stdlib + torch (torch only if a timed op needs it); runs on CPU.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import statistics
import time

SCHEMA = "shard-hw1/1"
ENV_KEYS = ("M25_DIR", "M25_BATCH", "M25_BATCH_MOE", "M25_KV_MAXLEN", "M25_KV_FP8", "M25_CUDA_GRAPH",
            "M25_GRAPH_MAX", "M25_STATIC_KV", "K3_MOE_MAX_TOKENS", "CUDA_VISIBLE_DEVICES",
            "HW1_BOX_LABEL")
MIN_WARMUP = 2          # a timed record with fewer discarded warm-up calls is invalid (JIT/autotune/alloc)
MIN_ITERS = 3

# sm -> family. The family is what the cross-family diff (research/hw1/xfamily.py) groups by.
FAMILY = {"sm_80": "ampere", "sm_86": "ampere", "sm_89": "ada", "sm_90": "hopper",
          "sm_100": "blackwell", "sm_120": "blackwell", "sm_121": "blackwell-gb10"}


def _torch():
    try:
        import torch
        return torch
    except Exception:           # the sys bench runs without torch
        return None


def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def device_info(device="cuda", label=None):
    """Fingerprint of the box: GPU name, sm, family, VRAM, kernel pins. CPU if no CUDA."""
    torch = _torch()
    info = {"host_arch": platform.machine(), "python": platform.python_version(),
            "torch": _ver("torch"), "vllm": _ver("vllm"), "flashinfer": _ver("flashinfer"),
            "label": label or os.environ.get("HW1_BOX_LABEL")}
    if torch is not None and device != "cpu" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        sm = f"sm_{p.major}{p.minor}"
        info.update(kind="cuda", name=p.name, sm=sm, family=FAMILY.get(sm, "unknown"),
                    vram_gb=round(p.total_memory / 2**30, 2), cuda=torch.version.cuda)
    else:
        info.update(kind="cpu", name=platform.processor() or "cpu", sm=None, family="cpu",
                    vram_gb=None, cuda=None)
    # numerics class per ARCHITECTURE-v2 §6 (backend, pins, arch class) -- identical here = comparable
    pins = f"{info['torch']}|{info['vllm']}|{info['flashinfer']}|{info['sm']}"
    info["numerics_class"] = hashlib.sha256(pins.encode()).hexdigest()[:16]
    return info


def env_snapshot():
    return {k: os.environ[k] for k in ENV_KEYS if k in os.environ}


def _sync(device):
    torch = _torch()
    if torch is not None and device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, *, warmup=3, iters=8, device="cuda"):
    """Time `fn()` with the warm-up discarded and the device drained.

    CUDA: per-iteration cuda events bracket each call and the stream is synchronized before the
    events are read; the whole loop is ALSO wall-clocked (perf_counter after a synchronize), so a
    host-launch-bound op shows up as wall >> event sum. CPU: perf_counter per call.
    Returns a stats dict (ms): median, p10, p90, mean, n, warmup, method, wall_ms_per_iter."""
    if warmup < MIN_WARMUP or iters < MIN_ITERS:
        raise ValueError(f"warmup>={MIN_WARMUP} and iters>={MIN_ITERS} required")
    torch = _torch()
    cuda = torch is not None and device != "cpu" and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    _sync(device)
    per = []
    t0 = time.perf_counter()
    if cuda:
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for _ in range(iters)]
        for a, b in evs:
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        per = [a.elapsed_time(b) for a, b in evs]
        method = "cuda_event"
    else:
        for _ in range(iters):
            s = time.perf_counter()
            fn()
            per.append((time.perf_counter() - s) * 1e3)
        method = "perf_counter"
    _sync(device)
    wall = (time.perf_counter() - t0) * 1e3 / iters
    per_sorted = sorted(per)
    q = lambda f: per_sorted[min(len(per_sorted) - 1, int(f * len(per_sorted)))]
    return {"ms": statistics.median(per), "ms_p10": q(0.1), "ms_p90": q(0.9),
            "ms_mean": statistics.fmean(per), "n": iters, "warmup": warmup, "method": method,
            "wall_ms_per_iter": wall}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Results:
    """Accumulates records/checks for one script run and writes the schema-valid JSON."""

    def __init__(self, script, mode, *, device="cuda", stub=False, label=None, argv=None):
        self.doc = {"schema": SCHEMA, "script": script, "mode": mode, "stub": bool(stub),
                    "device": device_info(device, label), "env": env_snapshot(),
                    "argv": list(argv or []), "started": now(), "finished": None,
                    "records": [], "checks": [], "derived": {}}

    def record(self, rid, op, params, stats=None, *, skipped=None, **extra):
        r = {"id": rid, "op": op, "params": dict(params), "skipped": skipped}
        if stats is not None:
            r.update(stats)
        r.update(extra)
        self.doc["records"].append(r)
        return r

    def check(self, cid, passed, detail="", **extra):
        c = {"id": cid, "pass": bool(passed), "detail": detail}
        c.update(extra)
        self.doc["checks"].append(c)
        return c

    def write(self, path):
        self.doc["finished"] = now()
        errs = validate(self.doc)
        if errs:
            raise ValueError("results fail schema: " + "; ".join(errs[:5]))
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.doc, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path


def validate(doc):
    """Return a list of schema errors (empty = valid). Hand-rolled to stay stdlib-only."""
    e = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if doc.get("schema") != SCHEMA:
        e.append(f"schema != {SCHEMA}")
    for k, t in (("script", str), ("mode", str), ("stub", bool), ("device", dict), ("env", dict),
                 ("records", list), ("checks", list), ("derived", dict), ("started", str)):
        if not isinstance(doc.get(k), t):
            e.append(f"{k}: want {t.__name__}")
    dev = doc.get("device") or {}
    for k in ("kind", "family", "numerics_class"):
        if not dev.get(k):
            e.append(f"device.{k} missing")
    for i, r in enumerate(doc.get("records") or []):
        w = f"records[{i}]"
        if not isinstance(r, dict) or not r.get("id") or not r.get("op") or not isinstance(r.get("params"), dict):
            e.append(f"{w}: id/op/params required")
            continue
        if r.get("skipped"):
            if not isinstance(r["skipped"], str):
                e.append(f"{w}: skipped must be a reason string")
            continue
        if "ms" not in r:
            continue                                   # an untimed record (bytes, hashes, footprint)
        ms = r["ms"]
        if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            e.append(f"{w}: ms must be finite > 0")
        if r.get("method") not in ("cuda_event", "perf_counter"):
            e.append(f"{w}: timing method missing")
        if not isinstance(r.get("warmup"), int) or r["warmup"] < MIN_WARMUP:
            e.append(f"{w}: warmup < {MIN_WARMUP} (warm-up must be discarded)")
        if not isinstance(r.get("n"), int) or r["n"] < MIN_ITERS:
            e.append(f"{w}: n < {MIN_ITERS}")
        if dev.get("kind") == "cuda" and r.get("method") != "cuda_event":
            e.append(f"{w}: cuda box but not cuda_event timing")
    for i, c in enumerate(doc.get("checks") or []):
        if not isinstance(c, dict) or not c.get("id") or not isinstance(c.get("pass"), bool):
            e.append(f"checks[{i}]: id/pass required")
    return e


def load(path):
    with open(path) as f:
        doc = json.load(f)
    errs = validate(doc)
    if errs:
        raise ValueError(f"{path}: " + "; ".join(errs[:5]))
    return doc
