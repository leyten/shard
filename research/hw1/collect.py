"""Collect HW1 results: validate every *.json against shard-hw1/1 and write one bundle.

  python research/hw1/collect.py results/hw1 [more dirs/files] --out results/hw1/bundle.json

The bundle lists each file (valid or the schema errors), the device family it ran on, all checks, and
flags stub data. Exit 1 if any file is invalid or any gate-check failed -- a run is not "done" until
this is green or the failures are explained.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw1_common as C  # noqa: E402


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    return [f for f in files if not f.endswith("bundle.json")]


def bundle(paths):
    out = {"schema": "shard-hw1-bundle/1", "created": C.now(), "files": [], "checks": [], "stub": False}
    ok = True
    for f in gather(paths):
        try:
            doc = json.load(open(f))
        except (OSError, ValueError) as e:
            out["files"].append({"path": f, "valid": False, "errors": [str(e)]})
            ok = False
            continue
        errs = C.validate(doc)
        ent = {"path": f, "valid": not errs, "errors": errs[:10]}
        if not errs:
            ent.update(script=doc["script"], mode=doc["mode"], family=doc["device"]["family"],
                       device=doc["device"].get("name"), records=len(doc["records"]), stub=doc["stub"])
            out["stub"] |= doc["stub"]
            for c in doc["checks"]:
                out["checks"].append(dict(c, file=os.path.basename(f)))
                ok &= c["pass"]
        ok &= not errs
        out["files"].append(ent)
    out["ok"] = bool(ok)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    b = bundle(a.paths)
    with open(a.out, "w") as f:
        json.dump(b, f, indent=1, sort_keys=True)
    for e in b["files"]:
        print(f"{'OK ' if e['valid'] else 'BAD'} {e['path']} {e.get('family', '')} {'; '.join(e['errors'][:2])}")
    for c in b["checks"]:
        print(f"{'PASS' if c['pass'] else 'FAIL'} {c['id']} {c['file']}: {c['detail']}")
    return 0 if b["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
