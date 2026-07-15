"""python -m shard.stage — the stage entrypoint the node daemon execs (c0mpute NODE_DAEMON.md §4).

Promotes the operator SSH launch string (phase0/m25_scatter_pipe.py:stage_cmd) into a first-class
CLI: the ring assignment arrives as flags, the engine env is derived HERE (not in a hand-built
shell prefix), and the process speaks a machine-readable stdout contract a supervisor can wait on:

    SHARD_STAGE_OK    {...}   --check preflight passed (engine imports, model dir sane), exit 0
    SHARD_STAGE_READY {...}   weights loaded, forward link up, listening (emitted by serve())
    SHARD_STAGE_FATAL {...}   unrecoverable error; the process exits nonzero

Layout-portable: works from a repo checkout (phase0/ beside shard/) AND the flat single-dir box
layout — no PYTHONPATH hand-patching (the landmine that bit the first residential join). Secrets
(SHARD_SWARM_TOKEN, SHARD_PSK) stay env-only: they must never appear in argv, which is world-
readable via ps.

  serve:     python -m shard.stage --stage 1 --nstages 3 --lo 12 --hi 24 --next 127.0.0.1:29611 --dir ~/m25
  preflight: python -m shard.stage --check --dir ~/m25
"""
import argparse
import json
import os
import sys
import traceback


def _emit(tag, **fields):
    print(tag + " " + json.dumps(fields), flush=True)


def _fatal(msg, **fields):
    _emit("SHARD_STAGE_FATAL", error=msg, **fields)
    return 1


def _bootstrap_path():
    """Make the phase0 engine modules importable from a repo checkout; on the flat box layout
    (every file in one dir, sys.path[0] = that dir) they already are."""
    p0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0")
    if os.path.isdir(p0) and p0 not in sys.path:
        sys.path.insert(0, p0)


def _apply_env(a):
    """The assignment → engine env, BEFORE the engine import (m25_stage/m25_pipe read env at module
    level). Explicit flags win; mode defaults only fill gaps so an operator env stays in charge."""
    if a.dir:
        os.environ["M25_DIR"] = os.path.abspath(os.path.expanduser(a.dir))
    # identity/encryption are the sidecar's job; the local sidecar is also the only legitimate
    # dialer, so the engine hop binds loopback (raw TCP must not bypass the sidecar allowlist)
    os.environ.setdefault("SHARD_TRANSPORT", "libp2p")
    os.environ.setdefault("M25_ENGINE_BIND", "127.0.0.1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.batch is not None:
        os.environ["M25_BATCH"] = str(a.batch)
    if a.kv_maxlen is not None:
        os.environ["M25_KV_MAXLEN"] = str(a.kv_maxlen)
    if a.receipts:
        os.environ["SHARD_RECEIPTS"] = "1"
    if a.graph_off:                       # a probe-measured graph-corrupt card serves at its eager
        os.environ["M25_CUDA_GRAPH"] = "0"  # speed — eager IS its reference numerics


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m shard.stage",
        description="run one pipeline stage of the sharded model (the node daemon's exec target)")
    ap.add_argument("--stage", type=int, help="this stage's index in the ring [0, nstages)")
    ap.add_argument("--nstages", type=int, help="total stages in the ring")
    ap.add_argument("--lo", type=int, help="first layer held (inclusive)")
    ap.add_argument("--hi", type=int, help="last layer held (exclusive)")
    ap.add_argument("--port", type=int, default=29610, help="engine listen port (default 29610)")
    ap.add_argument("--next", dest="nxt", default=None,
                    help="host:port of the forward ring leg (omit on the tail)")
    ap.add_argument("--timeout", type=int, default=600, help="per-frame recv deadline, seconds")
    ap.add_argument("--dir", default=None, help="model dir (wins over M25_DIR)")
    ap.add_argument("--batch", type=int, default=None, help="batched serving width (M25_BATCH)")
    ap.add_argument("--kv-maxlen", type=int, default=None, help="KV buffer cap (M25_KV_MAXLEN)")
    ap.add_argument("--receipts", action="store_true", help="sign per-stage receipts (SHARD_RECEIPTS=1)")
    ap.add_argument("--graph-off", action="store_true",
                    help="force eager compute (graph-corrupt card relegated by its probe verdict)")
    ap.add_argument("--check", action="store_true",
                    help="preflight only: import the engine against the model dir, print SHARD_STAGE_OK, exit")
    a = ap.parse_args(argv)

    _apply_env(a)
    _bootstrap_path()

    mdir = os.environ.get("M25_DIR")
    if not mdir:
        return _fatal("no model dir: pass --dir or set M25_DIR")
    missing = [f for f in ("config.json", "model.safetensors.index.json")
               if not os.path.isfile(os.path.join(mdir, f))]
    if missing:
        return _fatal(f"model dir is missing {missing}", dir=mdir)

    try:
        import m25_pipe as MP             # heavy: torch + m25_stage's module-level M25_DIR init
    except ImportError as e:
        return _fatal(f"engine import failed: {e}",
                      hint="run from a shard checkout (phase0/ beside shard/) or the flat box layout")

    if a.check:
        import torch
        _emit("SHARD_STAGE_OK", dir=mdir, transport=os.environ["SHARD_TRANSPORT"],
              cuda=torch.cuda.is_available(),
              device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        return 0

    if None in (a.stage, a.nstages, a.lo, a.hi):
        return _fatal("--stage/--nstages/--lo/--hi are required to serve")
    try:
        MP.serve(a.stage, a.nstages, a.lo, a.hi, a.port, a.nxt, a.timeout)
    except KeyboardInterrupt:
        return 0
    except Exception as e:                # the supervisor contract: fatal is LOUD + machine-readable,
        traceback.print_exc()             # full traceback kept for the human reading the log
        return _fatal(f"{type(e).__name__}: {e}", stage=a.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
