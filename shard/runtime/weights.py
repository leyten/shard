"""WeightSource over local safetensors files (a ranged-fetch dir, shard/fetch.py).

numpy has no bf16 / fp8 dtypes, so `read` hands over the STORAGE (BF16 as uint16 bit patterns, FP8
as uint8, the same convention as shard/runtime/api.Rows) and `dtype_of` names the checkpoint's
declared dtype, which an adapter checks against its weight plan: without it an FP8 tensor and a U8
one are indistinguishable. Headers are validated by shard.fetch.parse_safetensors_header (spans ==
dtype x shape, in bounds, non-overlapping); a name held by two files is refused, never picked.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

from ..fetch import FetchError, parse_safetensors_header

ST_STORAGE = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
              "F8_E4M3": np.uint8, "F8_E5M2": np.uint8, "F8_E8M0": np.uint8, "U8": np.uint8,
              "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U16": np.uint16,
              "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_}


class SafetensorsSource:
    """`names()` / `read(name)` / `dtype_of(name)` over a set of safetensors files.

    artifact_cid: pass the manifest's content id when there is one. The default is derived from the
    headers only ("hdr-..."): it names the tensor LAYOUT, not the bytes, and says so."""

    def __init__(self, paths, artifact_cid: str | None = None):
        self._where: dict[str, tuple[str, dict]] = {}
        h = hashlib.sha256()
        for path in sorted(paths):
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    raise FetchError(f"{path}: not a safetensors file")
                raw = head + f.read(struct.unpack("<Q", head)[0])
            for name, t in parse_safetensors_header(raw, size, f"{path}: ").items():
                if name in self._where:
                    raise FetchError(f"{name} is in both {self._where[name][0]} and {path}")
                self._where[name] = (path, t)
                h.update(json.dumps([name, t["dtype"], t["shape"]]).encode())
        self.artifact_cid = artifact_cid or "hdr-" + h.hexdigest()[:32]

    @classmethod
    def from_dir(cls, root: str, artifact_cid: str | None = None) -> "SafetensorsSource":
        """Every *.safetensors under root (recursively, skipping the fetch's .source/ cache)."""
        paths = []
        for d, subdirs, files in os.walk(root):
            subdirs[:] = sorted(s for s in subdirs if not s.startswith("."))
            paths += [os.path.join(d, f) for f in files if f.endswith(".safetensors")]
        if not paths:
            raise FetchError(f"no .safetensors under {root}")
        return cls(paths, artifact_cid)

    def names(self):
        return sorted(self._where)

    def dtype_of(self, name: str) -> str:
        return self._where[name][1]["dtype"]

    def read(self, name: str) -> np.ndarray:
        path, t = self._where[name]
        st = ST_STORAGE.get(t["dtype"])
        if st is None:
            raise FetchError(f"{name}: dtype {t['dtype']} has no numpy storage")
        count = int(np.prod(t["shape"])) if t["shape"] else 1
        with open(path, "rb") as f:
            f.seek(t["start"])
            buf = f.read(t["nbytes"])
        if len(buf) != t["nbytes"]:
            raise FetchError(f"{name}: short read from {path}")
        return np.frombuffer(buf, dtype=st, count=count).reshape(t["shape"])
