"""Tensor-granular ranged fetch (shard/fetch.py, shard v2 R1), CPU only, no network.

MiMo-V2.6 is sharded by EXPERT GROUP, so every layer spans every file and a file-granular pull of
any layer range is the whole model. The ranged path reads each file's safetensors header and pulls
exactly the byte ranges of the wanted tensors. These tests serve synthetic checkpoints with MiMo's
real tensor NAMES and real qkv/scale SHAPES (Flash global 13568 rows / 108 scale rows, SWA and MTP
14848 / 116, Pro 27136 / 216; small hidden size), experts split across files by expert group, an
MTP stack and an audio tower that also number `layers.N`, from a local HTTP server that honours
Range and can lie, drop connections or corrupt bytes. They pin:

  - selection: exactly the decoder layers asked for (never MTP/audio `layers.N`), the right boundary
    tensors, generic across MiMo / M2.5 / K3 / DeepSeek-V4 names;
  - bytes: only the selected tensors' spans are requested (plus headers), and they land exactly;
  - resume: a mid-range disconnect resumes at the byte it stopped; a killed run keeps its journaled
    ranges, re-hashes them, and refetches only what is missing or corrupt on disk;
  - integrity: a corrupted range is caught by the expected per-tensor hash (the R1b hook) and only
    its ranges are refetched; a server that ignores/shifts Range, swaps the file or lies about
    lengths is refused; lying headers are refused; a non-TLS source needs expected hashes.

Run: python3 -m pytest tests/test_fetch_ranged.py -q
Real network (skipped by default): SHARD_NET_TESTS=1 python3 -m pytest tests/test_fetch_ranged.py -k network
"""
import hashlib
import json
import math
import os
import re
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save as st_save

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard import fetch as F                               # noqa: E402
from shard.fetch import FetchError, RangeSource            # noqa: E402
from shard.formats.mimo_v2 import qkv_rechunk              # noqa: E402
from shard.mlx_runtime import select_weight_keys           # noqa: E402

REV = "5711b268169967567844e1e560e8a3966da959b1"


# ── a local HTTP server that honours Range (and can misbehave) ───────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        srv = self.server.owner
        path = self.path.lstrip("/")
        data = srv.files.get(path)
        if data is None:
            self.send_error(404)
            return
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        a, b = (int(m.group(1)), min(int(m.group(2)), len(data) - 1)) if m else (0, len(data) - 1)
        if a >= len(data):
            self.send_error(416)
            return
        with srv.lock:
            srv.log.append((path, a, b))
        resp = {"status": 206 if m else 200, "content_range": f"bytes {a}-{b}/{len(data)}" if m else None,
                "body": data[a:b + 1], "length": None, "cut": None, "headers": {}}
        if srv.fault is not None:
            srv.fault(path, a, b, resp)
        if resp["status"] >= 400:
            self.send_error(resp["status"])
            return
        body = resp["body"]
        self.send_response(resp["status"])
        if resp["content_range"]:
            self.send_header("Content-Range", resp["content_range"])
        for k, v in resp["headers"].items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body) if resp["length"] is None else resp["length"]))
        self.end_headers()
        self.wfile.write(body if resp["cut"] is None else body[:resp["cut"]])
        self.wfile.flush()


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass                    # a client refusing a lying response mid-body is the expected case


class RangeServer:
    def __init__(self, files):
        self.files, self.log, self.fault, self.lock = dict(files), [], None, threading.Lock()
        self.httpd = _QuietServer(("127.0.0.1", 0), _Handler)
        self.httpd.owner = self
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def source(self, **kw):
        """The local server stands in for the TLS origin (trusted), unless a test says otherwise."""
        kw.setdefault("trusted", True)
        kw.setdefault("retries", 3)
        return RangeSource(self.url, **kw)

    def data_requests(self):
        """Requests past each file's header (i.e. tensor bytes), as (path, start, end_exclusive)."""
        out = []
        for path, a, b in self.log:
            if path.endswith(".safetensors") and a >= _data_start(self.files[path]):
                out.append((path, a, b + 1))
        return out

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(F.time, "sleep", lambda s: None)


# ── MiMo-shaped synthetic checkpoints ────────────────────────────────────────────────────────────

def _data_start(raw: bytes) -> int:
    return 8 + struct.unpack("<Q", raw[:8])[0]


def _spans(raw: bytes) -> dict:
    """{tensor: (abs_start, abs_end)} straight off the JSON header (independent of shard.fetch)."""
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    return {k: (8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1])
            for k, v in hdr.items() if k != "__metadata__"}


def _rand(g, shape, dtype):
    nb = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return torch.randint(0, 256, (nb,), dtype=torch.uint8, generator=g).view(dtype).reshape(shape)


def _fp8(g, shape):
    return (torch.randn(shape, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)


VARIANTS = {
    # layers, attention pattern (0 global, 1 SWA), heads, checkpoint TP, experts, EP files, MTP layers
    "flash": dict(n_layers=6, pattern=[0, 1, 1, 1, 1, 0], glob=(64, 4), swa=(64, 8), tp=4,
                  n_exp=8, n_ep=4, n_mtp=3),
    "pro": dict(n_layers=3, pattern=[0, 1, 0], glob=(128, 8), swa=(128, 8), tp=8,
                n_exp=6, n_ep=2, n_mtp=1),
}
H, VOCAB, INTER, MOE_INTER = 128, 512, 256, 64


def _qkv(g, n_q, n_kv, tp):
    rows = n_q * 192 + n_kv * 192 + n_kv * 128
    scale_rows = tp * math.ceil(rows // tp / 128)
    return _fp8(g, (rows, H)), (torch.rand(scale_rows, H // 128, generator=g) + 0.5)


def _attn(g, P, n_q, n_kv, tp, swa, t):
    t[P + "self_attn.qkv_proj.weight"], t[P + "self_attn.qkv_proj.weight_scale_inv"] = _qkv(g, n_q, n_kv, tp)
    t[P + "self_attn.o_proj.weight"] = _rand(g, (H, n_q * 128), torch.bfloat16)
    if swa:
        t[P + "self_attn.attention_sink_bias"] = _rand(g, (n_q,), torch.bfloat16)


def _dense_fp8(g, P, t):
    for n, shape in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER))):
        t[f"{P}mlp.{n}.weight"] = _fp8(g, shape)
        t[f"{P}mlp.{n}.weight_scale_inv"] = torch.rand(shape[0] // 128, shape[1] // 128, generator=g) + 0.5


def build_mimo_like(variant: str, seed: int = 0):
    """{file: bytes} for a MiMo-V2.6-shaped checkpoint: model_pp0_ep{i}_shard0 holds experts
    [i*per, (i+1)*per) of EVERY MoE layer, ep0 also every non-expert tensor (+ embed/norm/lm_head,
    an audio tower numbering its own `layers.N`, a vision block), model_mtp holds the MTP stack."""
    v = VARIANTS[variant]
    g = torch.Generator().manual_seed(seed)
    per = v["n_exp"] // v["n_ep"]
    files = {f"model_pp0_ep{i}_shard0.safetensors": {} for i in range(v["n_ep"])}
    ep0 = files["model_pp0_ep0_shard0.safetensors"]
    for L in range(v["n_layers"]):
        P = f"model.layers.{L}."
        swa = v["pattern"][L] == 1
        n_q, n_kv = v["swa"] if swa else v["glob"]
        ep0[P + "input_layernorm.weight"] = _rand(g, (H,), torch.bfloat16)
        ep0[P + "post_attention_layernorm.weight"] = _rand(g, (H,), torch.bfloat16)
        _attn(g, P, n_q, n_kv, v["tp"], swa, ep0)
        if L == 0:
            _dense_fp8(g, P, ep0)
            continue
        ep0[P + "mlp.gate.weight"] = _rand(g, (v["n_exp"], H), torch.bfloat16)
        ep0[P + "mlp.gate.e_score_correction_bias"] = torch.rand(v["n_exp"], generator=g) + 1.7
        for e in range(v["n_exp"]):
            t = files[f"model_pp0_ep{e // per}_shard0.safetensors"]
            E = f"{P}mlp.experts.{e}."
            for n, (rows, k) in (("gate_proj", (MOE_INTER, H)), ("up_proj", (MOE_INTER, H)),
                                 ("down_proj", (H, MOE_INTER))):
                t[f"{E}{n}.weight"] = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=g)
                t[f"{E}{n}.weight_scale"] = torch.randint(100, 150, (rows, k // 32), dtype=torch.uint8,
                                                          generator=g)
    mtp = {}
    for k in range(v["n_mtp"]):
        P = f"model.mtp.layers.{k}."
        for n in ("enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm"):
            mtp[f"{P}{n}.weight"] = _rand(g, (H,), torch.bfloat16)
        mtp[P + "eh_proj.weight"] = _rand(g, (H, 2 * H), torch.bfloat16)
        _attn(g, P, *v["swa"], v["tp"], True, mtp)
        _dense_fp8(g, P, mtp)
    files["model_mtp.safetensors"] = mtp
    for j in range(2):
        ep0[f"audio_encoder.input_local_transformer.layers.{j}.self_attn.q_proj.weight"] = \
            _rand(g, (16, 16), torch.bfloat16)
    ep0["audio_encoder.input_local_transformer.norm.weight"] = _rand(g, (16,), torch.bfloat16)
    ep0["visual.blocks.0.attn.qkv.weight"] = _rand(g, (48, 16), torch.bfloat16)
    ep0["speech_embeddings.0.weight"] = _rand(g, (8, 16), torch.bfloat16)
    ep0["model.embed_tokens.weight"] = _rand(g, (VOCAB, H), torch.bfloat16)
    ep0["model.norm.weight"] = _rand(g, (H,), torch.bfloat16)
    ep0["lm_head.weight"] = _rand(g, (VOCAB, H), torch.bfloat16)

    out = {fn: st_save(t) for fn, t in files.items()}
    weight_map = {n: fn for fn, t in files.items() for n in t}
    out["model.safetensors.index.json"] = json.dumps(
        {"metadata": {"total_size": sum(len(b) for b in out.values()), "tp_size": v["tp"]},
         "weight_map": weight_map}).encode()
    out["config.json"] = json.dumps({"model_type": "mimo_v2", "num_hidden_layers": v["n_layers"],
                                     "hybrid_layer_pattern": v["pattern"], "num_key_value_heads": v["tp"],
                                     "tie_word_embeddings": False}).encode()
    return out, weight_map


@pytest.fixture(scope="module")
def flash():
    return build_mimo_like("flash")


@pytest.fixture(scope="module")
def pro():
    return build_mimo_like("pro", seed=1)


@pytest.fixture
def server(flash):
    s = RangeServer(flash[0])
    yield s
    s.close()


def _expected(files, weight_map, names):
    """True per-tensor {sha256, dtype, shape} of the source (what a signed manifest v2 would carry)."""
    out = {}
    for n in names:
        raw = files[weight_map[n]]
        a, b = _spans(raw)[n]
        hdr = json.loads(raw[8:_data_start(raw)])[n]
        out[n] = {"sha256": hashlib.sha256(raw[a:b]).hexdigest(), "dtype": hdr["dtype"], "shape": hdr["shape"]}
    return out


def _assert_fetched_equals_source(out_dir, files, weight_map, rec):
    for grp in rec["groups"].values():
        with safe_open(os.path.join(out_dir, grp["path"]), "pt") as f:
            assert set(f.keys()) == set(grp["tensors"])
            for n in f.keys():
                raw = files[weight_map[n]]
                a, b = _spans(raw)[n]
                got = f.get_tensor(n)
                assert got.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes() == raw[a:b], n
                assert grp["tensors"][n]["sha256"] == hashlib.sha256(raw[a:b]).hexdigest()


def _union(spans):
    """Merge [a, b) spans per file into a sorted list of disjoint spans."""
    out = {}
    for path, a, b in sorted(spans):
        cur = out.setdefault(path, [])
        if cur and a <= cur[-1][1]:
            cur[-1][1] = max(cur[-1][1], b)
        else:
            cur.append([a, b])
    return out


# ── selection: exact, and generic across checkpoint namespaces ───────────────────────────────────

def test_select_mimo_layers_is_exact_and_skips_other_numbered_towers(flash):
    _, wm = flash
    g = F.select_tensors(wm, 1, 3)
    assert list(g) == ["layer-00001", "layer-00002"]
    for j in (1, 2):
        assert g[f"layer-{j:05d}"] == sorted(n for n in wm if n.startswith(f"model.layers.{j}."))
    picked = {n for v in g.values() for n in v}
    assert not any(n.startswith(("model.mtp.", "audio_encoder.")) for n in picked)
    # the file-granular/loader union takes MTP layer 1 and audio layer 1 too: bytes a
    # tensor-granular fetch must not pay
    union = set(select_weight_keys(wm, 1, 3, is_head=False, is_tail=False))
    assert union - picked and all(n.startswith(("model.mtp.", "audio_encoder.")) for n in union - picked)


def test_select_mimo_boundaries(flash):
    _, wm = flash
    g = F.select_tensors(wm, 0, 1, head=True)
    assert g["embed"] == ["model.embed_tokens.weight"]                    # not speech_embeddings
    g = F.select_tensors(wm, 5, 6, tail=True)
    assert g["norm"] == ["model.norm.weight"]                             # not the audio tower's norm
    assert g["lm_head"] == ["lm_head.weight"]
    g = F.select_tensors(wm, 5, 6, tail=True, tied=True)
    assert "lm_head" not in g and g["embed"] == ["model.embed_tokens.weight"]


def test_select_mtp_by_extra_or_by_layout(flash):
    _, wm = flash
    g = F.select_tensors(wm, 5, 6, tail=True, extra=("model.mtp",))
    assert g["extra-model.mtp"] == sorted(n for n in wm if n.startswith("model.mtp."))
    g = F.select_tensors(wm, 0, 2, layout={"layers": "model.mtp.layers"})
    assert list(g) == ["layer-00000", "layer-00001"]
    assert all(n.startswith(("model.mtp.layers.0.", "model.mtp.layers.1.")) for v in g.values() for n in v)


def _m25_map(n_layers=6):
    wm = {}
    for j in range(n_layers):
        fn = f"model-{1 if j < n_layers // 2 else 2:05d}-of-00003.safetensors"
        for name in ("input_layernorm.weight", "self_attn.q_proj.weight",
                     "block_sparse_moe.gate.weight", "block_sparse_moe.e_score_correction_bias"):
            wm[f"model.layers.{j}.{name}"] = fn
        for e in range(4):
            wm[f"model.layers.{j}.block_sparse_moe.experts.{e}.w1.weight"] = fn
    wm["model.embed_tokens.weight"] = "model-00001-of-00003.safetensors"
    wm["model.norm.weight"] = "model-00002-of-00003.safetensors"
    wm["lm_head.weight"] = "model-00003-of-00003.safetensors"
    return wm


@pytest.mark.parametrize("lo,hi", [(0, 1), (0, 3), (2, 5), (5, 6), (0, 6)])
def test_select_m25_equals_the_loaders_selection(lo, hi):
    """One numbered tower (M2.5): the ranged selection is exactly the tensor set the loader
    materializes (mlx_runtime.select_weight_keys), for every role."""
    wm = _m25_map()
    for head in (False, True):
        for tail in (False, True):
            for tied in (False, True):
                got = {n for v in F.select_tensors(wm, lo, hi, head=head, tail=tail, tied=tied).values()
                       for n in v}
                assert got == set(select_weight_keys(wm, lo, hi, is_head=head, is_tail=tail, tied=tied))


def test_select_k3_namespace_excludes_the_vision_tower():
    wm = {f"language_model.model.layers.{j}.self_attn.q_proj.weight": "t.safetensors" for j in range(4)}
    wm.update({f"vision_tower.encoder.layers.{j}.attn.qkv.weight": "v.safetensors" for j in range(4)})
    wm["language_model.model.embed_tokens.weight"] = "t.safetensors"
    wm["language_model.model.norm.weight"] = "t.safetensors"
    wm["language_model.lm_head.weight"] = "t.safetensors"
    g = F.select_tensors(wm, 1, 2, head=True, tail=True)
    assert g == {"layer-00001": ["language_model.model.layers.1.self_attn.q_proj.weight"],
                 "embed": ["language_model.model.embed_tokens.weight"],
                 "norm": ["language_model.model.norm.weight"],
                 "lm_head": ["language_model.lm_head.weight"]}


def test_select_deepseek_v4_names_need_a_layout_and_work_with_one():
    """V4's names (`layers.N`, `embed`, `head`, `norm`, `mtp.0.*`, top-level `hc_head_*`): the
    conventional boundary names do not exist, so a head/tail fails LOUD until the per-model layout
    names them; the final norm is `norm`, never the MTP block's `mtp.0.norm`."""
    wm = {f"layers.{j}.attn.wq_a.weight": "m.safetensors" for j in range(4)}
    wm.update({"embed.weight": "m.safetensors", "head.weight": "m.safetensors",
               "norm.weight": "m.safetensors", "mtp.0.norm.weight": "m.safetensors",
               "mtp.0.e_proj.weight": "m.safetensors", "hc_head_fn": "m.safetensors",
               "hc_head_base": "m.safetensors", "hc_head_scale": "m.safetensors"})
    assert F.select_tensors(wm, 0, 2) == {"layer-00000": ["layers.0.attn.wq_a.weight"],
                                          "layer-00001": ["layers.1.attn.wq_a.weight"]}
    with pytest.raises(FetchError, match="layout"):
        F.select_tensors(wm, 0, 1, head=True)
    g = F.select_tensors(wm, 3, 4, tail=True, layout={"embed": "embed", "lm_head": "head"},
                         extra=("hc_head_fn", "hc_head_base", "hc_head_scale"))
    assert g["norm"] == ["norm.weight"] and g["lm_head"] == ["head.weight"]
    assert g["extra-hc_head_fn"] == ["hc_head_fn"]


def test_select_fails_loud(flash):
    _, wm = flash
    with pytest.raises(FetchError, match=r"layer\(s\) \[6, 7\]"):
        F.select_tensors(wm, 5, 8)                                        # past the model
    with pytest.raises(FetchError, match="selected twice"):
        F.select_tensors(wm, 1, 2, extra=("model.layers.1",))
    with pytest.raises(FetchError, match="layer pattern"):
        F.select_tensors([f"transformer.h.{j}.w" for j in range(3)], 0, 1)


# ── the fetch: exactly the needed bytes, landed exactly ──────────────────────────────────────────

def test_fetch_requests_only_the_selected_tensor_bytes(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path / "w")
    rec = F.fetch_layer_tensors(server.source(), out, 0, 2, head=True, workers=4, max_job=1 << 20)
    groups = F.select_tensors(wm, 0, 2, head=True)
    want = [(wm[n], *_spans(files[wm[n]])[n]) for v in groups.values() for n in v]
    assert _union(server.data_requests()) == _union(want)             # nothing more, nothing less
    # header ranges only for the files the selection touches (MTP file never contacted)
    touched = {p for p, _, _ in server.log}
    assert "model_mtp.safetensors" not in touched
    assert touched == {wm[n] for v in groups.values() for n in v} | {
        "model.safetensors.index.json", "config.json"}
    _assert_fetched_equals_source(out, files, wm, rec)
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert set(idx) == {n for v in groups.values() for n in v}
    assert idx["model.layers.1.mlp.experts.7.down_proj.weight"] == "layer-00001.safetensors"
    t = rec["groups"]["layer-00001"]["tensors"]["model.layers.1.mlp.experts.7.down_proj.weight"]
    assert t["src"] == "model_pp0_ep3_shard0.safetensors"               # expert group 3 of 4
    fetched = sum(b - a for _, a, b in want)
    whole = sum(len(b) for p, b in files.items() if p.endswith(".safetensors"))
    assert fetched < whole / 3


@pytest.mark.parametrize("variant,layer,scale_rows", [("flash", 0, 108), ("flash", 1, 116), ("pro", 0, 216),
                                                     ("pro", 1, 216)])
def test_fetched_qkv_decodes_exactly_like_the_source(variant, layer, scale_rows, flash, pro, tmp_path):
    """The fetched bytes feed the format primitives unchanged: the per-chunk qkv grid (108 on Flash
    global, 116 on SWA, 216 on Pro) re-chunks identically from the fetched file and the source."""
    files, wm = flash if variant == "flash" else pro
    v = VARIANTS[variant]
    srv = RangeServer(files)
    try:
        rec = F.fetch_layer_tensors(srv.source(), str(tmp_path), layer, layer + 1, max_job=1 << 18)
    finally:
        srv.close()
    P = f"model.layers.{layer}.self_attn.qkv_proj."
    n_q, n_kv = v["swa"] if v["pattern"][layer] == 1 else v["glob"]
    with safe_open(str(tmp_path / rec["groups"][f"layer-{layer:05d}"]["path"]), "pt") as f:
        w, s = f.get_tensor(P + "weight"), f.get_tensor(P + "weight_scale_inv")
    raw = files[wm[P + "weight"]]                                     # the source bytes, decoded directly
    a, b = _spans(raw)[P + "weight"]
    w0 = torch.frombuffer(bytearray(raw[a:b]), dtype=torch.float8_e4m3fn).reshape(w.shape)
    a, b = _spans(raw)[P + "weight_scale_inv"]
    s0 = torch.frombuffer(bytearray(raw[a:b]), dtype=torch.float32).reshape(s.shape)
    assert s.shape[0] == scale_rows
    assert torch.equal(w.view(torch.uint8), w0.view(torch.uint8)) and torch.equal(s, s0)
    got = qkv_rechunk(w, s, n_q, n_kv, 192, 128, ckpt_tp=v["tp"])
    # independent of qkv_rechunk: chunk 1's Q heads, block by block with chunk 1's OWN scale rows
    # (the rows a plain r//128 grid or a plain [q|k|v] split gets wrong)
    tp = v["tp"]
    rpc, q_pc = w0.shape[0] // tp, n_q // tp * 192
    cs = math.ceil(rpc / 128)
    for blk in range(math.ceil(q_pc / 128)):
        r0, r1 = blk * 128, min((blk + 1) * 128, q_pc)
        want = w0[rpc + r0:rpc + r1].float() * s0[cs + blk]
        assert torch.equal(got[0][q_pc + r0:q_pc + r1], want), blk


# ── resume ───────────────────────────────────────────────────────────────────────────────────────

def test_mid_range_disconnect_resumes_at_the_byte_it_stopped(server, flash, tmp_path):
    files, wm = flash
    name = "model.layers.1.self_attn.o_proj.weight"                   # 2 MiB, one big job
    a, b = _spans(files[wm[name]])[name]
    state = {"cut": False}

    def fault(path, s, e, resp):
        if not state["cut"] and path == wm[name] and s <= a < e:
            state["cut"] = True
            resp["cut"] = 12345                                       # drop the connection mid-body
    server.fault = fault
    rec = F.fetch_tensors(server.source(), wm, {"g": [name]}, str(tmp_path), max_job=64 << 20)
    reqs = [(s, e) for p, s, e in server.data_requests() if p == wm[name]]
    assert reqs == [(a, b), (a + 12345, b)]                           # resumed, not restarted
    _assert_fetched_equals_source(str(tmp_path), files, wm, rec)


def test_killed_run_resumes_keeping_and_rechecking_its_journaled_ranges(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path)
    names = F.select_tensors(wm, 1, 2)
    served = {"n": 0}

    def fail_after_5(path, s, e, resp):
        if s >= _data_start(files[path]):                             # a tensor-data request
            served["n"] += 1
            if served["n"] > 5:
                resp["status"] = 500
    server.fault = fail_after_5
    with pytest.raises(FetchError, match="without progress"):
        F.fetch_tensors(server.source(retries=2), wm, names, out, workers=1, max_job=1 << 18)
    part = os.path.join(out, "layer-00001.safetensors.part")
    journal = open(part[:-5] + ".journal").read().splitlines()
    assert len(journal) == 1 + 5 and not os.path.exists(part[:-5])    # 5 ranges journaled, not promoted
    first_run = server.data_requests()[:5]
    # a journaled range rotted on disk between runs: flip one byte inside the 2nd one
    hdr_len = 8 + struct.unpack("<Q", open(part, "rb").read(8))[0]
    i, h = journal[2].split()
    jobs = F._plan_group(names["layer-00001"], wm, {fn: {"tensors": F.parse_safetensors_header(
        files[fn][:_data_start(files[fn])], len(files[fn])), "size": len(files[fn])}
        for fn in {wm[n] for n in names["layer-00001"]}}, 1 << 18)[3]
    with open(part, "r+b") as f:
        f.seek(hdr_len + jobs[int(i)][3] + 7)
        c = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([c[0] ^ 0xFF]))
    server.fault, server.log = None, []
    rec = F.fetch_tensors(server.source(), wm, names, out, workers=1, max_job=1 << 18)
    second = server.data_requests()
    rotted = (jobs[int(i)][0], jobs[int(i)][1], jobs[int(i)][1] + jobs[int(i)][2])
    kept = [r for r in first_run if r != rotted]
    assert rotted in second and not any(r in second for r in kept)
    assert len(second) == len(jobs) - 4
    _assert_fetched_equals_source(out, files, wm, rec)
    assert not os.path.exists(part) and not os.path.exists(part[:-5] + ".journal")


def test_second_run_is_a_no_op_and_a_tampered_group_is_refetched(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path)
    F.fetch_layer_tensors(server.source(), out, 2, 4)
    server.log = []
    F.fetch_layer_tensors(server.source(), out, 2, 4)
    assert server.log == []                                           # headers, index, groups all cached
    p = os.path.join(out, "layer-00003.safetensors")
    with open(p, "r+b") as f:
        f.seek(os.path.getsize(p) - 1)
        c = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([c[0] ^ 1]))                                    # same size, wrong content
    rec = F.fetch_layer_tensors(server.source(), out, 2, 4)
    assert server.data_requests() and all(
        any(n.startswith("model.layers.3.") for n, (a, b) in _spans(files[p_]).items() if a <= s < b)
        for p_, s, _ in server.data_requests())
    _assert_fetched_equals_source(out, files, wm, rec)


def test_incremental_ranges_share_a_dir_but_never_mix_sources(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path)
    F.fetch_layer_tensors(server.source(), out, 0, 1, head=True)
    rec = F.fetch_layer_tensors(server.source(), out, 1, 2, tail=True)
    assert set(rec["groups"]) == {"embed", "layer-00000", "layer-00001", "norm", "lm_head"}
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert idx["model.embed_tokens.weight"] == "embed.safetensors" and idx["lm_head.weight"] == "lm_head.safetensors"
    _assert_fetched_equals_source(out, files, wm, rec)
    other = RangeServer(files)
    try:
        with pytest.raises(FetchError, match="refusing to mix"):
            F.fetch_layer_tensors(other.source(), out, 2, 3)
    finally:
        other.close()


# ── integrity: corrupted bytes, lying servers, lying headers, untrusted transport ────────────────

def test_corrupted_range_is_caught_by_the_expected_hash_and_only_it_is_refetched(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path)
    groups = F.select_tensors(wm, 1, 2)
    expected = _expected(files, wm, groups["layer-00001"])
    bad = "model.layers.1.mlp.experts.5.up_proj.weight"
    a, _ = _spans(files[wm[bad]])[bad]

    def corrupt(path, s, e, resp):                                    # same length, one flipped byte
        if path == wm[bad] and s <= a + 3 < e + 1:
            body = bytearray(resp["body"])
            body[a + 3 - s] ^= 0x40
            resp["body"] = bytes(body)
    server.fault = corrupt
    with pytest.raises(FetchError, match=re.escape(bad)):
        F.fetch_tensors(server.source(), wm, groups, out, expected=expected, max_job=1 << 16)
    assert not os.path.exists(os.path.join(out, "layer-00001.safetensors"))
    # without an expected hash the same corruption is NOT detectable (same length, TLS is the
    # only root): that is the gap manifest v2 closes, pinned here rather than hidden
    gap = F.fetch_tensors(server.source(), wm, groups, str(tmp_path / "gap"), max_job=1 << 16)
    assert gap["groups"]["layer-00001"]["tensors"][bad]["sha256"] != expected[bad]["sha256"]
    server.fault, server.log = None, []
    rec = F.fetch_tensors(server.source(), wm, groups, out, expected=expected, max_job=1 << 16)
    again = server.data_requests()
    assert again and all(p == wm[bad] for p, _, _ in again)          # only the bad tensor's ranges
    _assert_fetched_equals_source(out, files, wm, rec)


@pytest.mark.parametrize("lie,match", [
    ("ignore_range", "HTTP 200"),
    ("shift", "server sent"),
    ("other_file", "different file"),
    ("long_length", "Content-Length"),
    ("no_content_range", "Content-Range"),
    ("gzip", "Content-Encoding"),
    ("redirect_http", "insecure redirect"),
])
def test_a_lying_server_is_refused_and_nothing_is_promoted(lie, match, server, flash, tmp_path):
    files, wm = flash
    name = "model.layers.2.self_attn.qkv_proj.weight"
    a, b = _spans(files[wm[name]])[name]

    def fault(path, s, e, resp):
        if not (path == wm[name] and s >= a):
            return
        if lie == "ignore_range":
            resp.update(status=200, content_range=None, body=files[path])
        elif lie == "shift":
            resp.update(content_range=f"bytes {s + 1}-{e + 1}/{len(files[path])}",
                        body=files[path][s + 1:e + 2])
        elif lie == "other_file":
            resp.update(content_range=f"bytes {s}-{e}/{len(files[path]) + 4096}")
        elif lie == "long_length":
            resp.update(length=len(resp["body"]) + 10)
        elif lie == "no_content_range":
            resp.update(content_range=None)
        elif lie == "gzip":
            resp["headers"]["Content-Encoding"] = "gzip"
        elif lie == "redirect_http":
            resp.update(status=302, content_range=None, body=b"")
            resp["headers"]["Location"] = server.url + "elsewhere/" + path
    server.fault = fault
    with pytest.raises(FetchError, match=match):
        F.fetch_tensors(server.source(), wm, {"g": [name]}, str(tmp_path))
    assert not os.path.exists(tmp_path / "g.safetensors")


def _hdr(entries, data_len, n_pad=0):
    hb = json.dumps(entries).encode() + b" " * n_pad
    return struct.pack("<Q", len(hb)) + hb, 8 + len(hb) + data_len


@pytest.mark.parametrize("entries,data_len,match", [
    ({"a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 6]}}, 6, "header span"),
    ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
      "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, 12, "overlaps"),
    ({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}, 9, "but the file is"),
    ({"a": {"dtype": "Q4_K", "shape": [4], "data_offsets": [0, 4]}}, 4, "unknown safetensors dtype"),
    ({"a": {"dtype": "U8", "shape": [-4], "data_offsets": [0, 4]}}, 4, "bad tensor shape"),
    ({"a": {"dtype": "U8", "shape": [4], "data_offsets": [4, 0]}}, 4, "bad data_offsets"),
    ({"a": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}, 2, "whole number of bytes"),
])
def test_lying_headers_are_refused(entries, data_len, match):
    raw, size = _hdr(entries, data_len)
    with pytest.raises(FetchError, match=match):
        F.parse_safetensors_header(raw, size)


def test_header_edge_cases():
    raw, size = _hdr({"__metadata__": {"format": "pt"},
                      "a": {"dtype": "F4", "shape": [2, 4], "data_offsets": [0, 4]},
                      "z": {"dtype": "BF16", "shape": [0], "data_offsets": [4, 4]}}, 4)
    got = F.parse_safetensors_header(raw, size)
    assert got["a"]["nbytes"] == 4 and got["a"]["start"] == len(raw) and got["z"]["nbytes"] == 0
    dup = b'{"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}'
    with pytest.raises(FetchError, match="duplicate"):
        F.parse_safetensors_header(struct.pack("<Q", len(dup)) + dup, 8 + len(dup) + 1)
    with pytest.raises(FetchError, match="implausible"):
        F.parse_safetensors_header(struct.pack("<Q", 1 << 40) + b"{}", 1 << 41)


def test_index_pointing_at_the_wrong_file_is_refused(server, flash, tmp_path):
    _, wm = flash
    lying = dict(wm)
    lying["model.layers.1.mlp.gate.weight"] = "model_pp0_ep2_shard0.safetensors"
    with pytest.raises(FetchError, match="whose header has no such tensor"):
        F.fetch_tensors(server.source(), lying, {"g": ["model.layers.1.mlp.gate.weight"]}, str(tmp_path))
    lying["model.layers.1.mlp.gate.weight"] = "../main/model_pp0_ep0_shard0.safetensors"
    with pytest.raises(FetchError, match="unsafe remote file name"):
        F.fetch_tensors(server.source(), lying, {"g": ["model.layers.1.mlp.gate.weight"]}, str(tmp_path))


def test_untrusted_transport_needs_expected_hashes(server, flash, tmp_path):
    files, wm = flash
    src = RangeSource(server.url)                                     # plain http: not an integrity root
    assert src.trusted is False
    names = {"g": ["model.layers.3.input_layernorm.weight", "model.layers.3.mlp.gate.weight"]}
    with pytest.raises(FetchError, match="not a trusted"):
        F.fetch_tensors(src, wm, names, str(tmp_path / "a"))
    with pytest.raises(FetchError, match="not a trusted"):
        F.fetch_tensors(src, wm, names, str(tmp_path / "a"),
                        expected=_expected(files, wm, names["g"][:1]))   # partial cover is not enough
    with pytest.raises(FetchError, match="trusted"):
        F.fetch_layer_tensors(src, str(tmp_path / "b"), 0, 1)           # index/config have no hash yet
    assert server.log == []                                           # refused before any byte moved
    rec = F.fetch_tensors(src, wm, names, str(tmp_path / "c"), expected=_expected(files, wm, names["g"]))
    _assert_fetched_equals_source(str(tmp_path / "c"), files, wm, rec)


def test_huggingface_source_pins_a_commit():
    for rev in ("main", "v1.0", REV[:12]):
        with pytest.raises(FetchError, match="commit sha"):
            RangeSource.huggingface("XiaomiMiMo/MiMo-V2.6-Flash-RL", rev)
    src = RangeSource.huggingface("XiaomiMiMo/MiMo-V2.6-Flash-RL", REV, {"Authorization": "Bearer SECRET"})
    assert src.trusted and src.base.endswith(f"/resolve/{REV}/") and "SECRET" not in repr(src)


def test_expected_binds_dtype_and_shape_not_only_bytes(server, flash, tmp_path):
    """A source relabelling a tensor (BF16 -> F16, same bytes) passes a bytes-only hash; the
    expected entry carries dtype + shape, so the relabel is refused before any data moves."""
    files, wm = flash
    name = "model.layers.1.self_attn.o_proj.weight"
    fn = wm[name]
    with safe_open_bytes(files[fn]) as tensors:
        tensors[name] = tensors[name].view(torch.float16)
        relabelled = st_save(tensors)
    evil = RangeServer({**files, fn: relabelled})
    try:
        src = RangeSource(evil.url)                                   # untrusted: expected is the root
        with pytest.raises(FetchError, match="expected BF16"):
            F.fetch_tensors(src, wm, {"g": [name]}, str(tmp_path), expected=_expected(files, wm, [name]))
        assert not any(a >= _data_start(relabelled) for p, a, _ in evil.log if p == fn)
        with pytest.raises(FetchError, match="must carry sha256, dtype and shape"):
            F.fetch_tensors(src, wm, {"g": [name]}, str(tmp_path), expected={name: "0" * 64})
    finally:
        evil.close()


class safe_open_bytes:
    """All tensors of a safetensors blob, as a mutable dict (for building a lying variant)."""
    def __init__(self, blob):
        from safetensors.torch import load
        self.t = load(blob)

    def __enter__(self):
        return self.t

    def __exit__(self, *a):
        return False


def test_default_trust_needs_https_pinned_to_a_commit():
    assert RangeSource(f"https://huggingface.co/x/y/resolve/{REV}/").trusted
    assert not RangeSource("https://huggingface.co/x/y/resolve/main/").trusted      # a moving branch
    assert not RangeSource("https://mirror.example/weights/").trusted              # nothing pinned
    assert not RangeSource(f"http://huggingface.co/x/y/resolve/{REV}/").trusted     # no TLS


def test_concurrent_fetches_into_one_dir_keep_every_group(server, flash, tmp_path):
    files, wm = flash
    out = str(tmp_path)
    errs = []

    def run(lo):
        try:
            F.fetch_layer_tensors(server.source(), out, lo, lo + 1, max_job=1 << 18)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=run, args=(j,)) for j in (1, 2, 3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    rec = json.load(open(os.path.join(out, "ranged.json")))
    assert set(rec["groups"]) == {"layer-00001", "layer-00002", "layer-00003"}
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert {v for v in idx.values()} == {f"layer-0000{j}.safetensors" for j in (1, 2, 3)}
    _assert_fetched_equals_source(out, files, wm, rec)


def test_nested_index_resolves_its_files_under_its_own_dir(tmp_path):
    """A drafter shipped under dflash/ with its own index, config and weights: its file names and
    config resolve under dflash/, never to the same-named files at the repo root."""
    def ckpt(val):
        return st_save({f"model.layers.{j}.fc.weight": torch.full((4, 4), val) for j in range(2)}
                       | {"model.norm.weight": torch.full((4,), val)})
    wm = {f"model.layers.{j}.fc.weight": "model.safetensors" for j in range(2)}
    wm["model.norm.weight"] = "model.safetensors"
    idx = json.dumps({"weight_map": wm}).encode()
    files = {"model.safetensors": ckpt(1.0), "model.safetensors.index.json": idx,
             "config.json": b'{"tie_word_embeddings": true}',
             "dflash/model.safetensors": ckpt(2.0), "dflash/model.safetensors.index.json": idx,
             "dflash/config.json": b'{"tie_word_embeddings": false}'}
    srv = RangeServer(files)
    try:
        rec = F.fetch_layer_tensors(srv.source(), str(tmp_path), 0, 2, index="dflash/model.safetensors.index.json")
    finally:
        srv.close()
    assert {p for p, _, _ in srv.log} == {"dflash/model.safetensors.index.json", "dflash/config.json",
                                           "dflash/model.safetensors"}
    with safe_open(str(tmp_path / "layer-00001.safetensors"), "pt") as f:
        assert f.get_tensor("model.layers.1.fc.weight")[0, 0].item() == 2.0
    assert json.load(open(tmp_path / "config.json")) == {"tie_word_embeddings": False}
    assert rec["groups"]["layer-00000"]["tensors"]["model.layers.0.fc.weight"]["src"] == "dflash/model.safetensors"


def test_cli_ranged_refuses_an_untrusted_base(server, tmp_path, capsys):
    rc = F._main_ranged(["--base", server.url, "--dir", str(tmp_path), "--lo", "0", "--hi", "1"])
    out = capsys.readouterr().out
    assert rc == 1 and "SHARD_FETCH_FATAL" in out and "trusted" in out
    assert server.log == []


# ── real network (skipped by default) ────────────────────────────────────────────────────────────

@pytest.mark.skipif(os.environ.get("SHARD_NET_TESTS") != "1", reason="network: set SHARD_NET_TESTS=1")
def test_network_range_fetch_one_real_mimo_tensor(tmp_path):
    """One real 13.8 KB tensor off the real 64-file checkpoint: layer 0's qkv scale grid, which must
    be the per-chunk 108 rows (not the plain 106). Reads ~240 KB of header + the tensor."""
    src = RangeSource.huggingface("XiaomiMiMo/MiMo-V2.6-Flash-RL", REV, F._hf_headers())
    name = "model.layers.0.self_attn.qkv_proj.weight_scale_inv"
    rec = F.fetch_tensors(src, {name: "model_pp0_ep0_shard0.safetensors"}, {"qkv_scale": [name]},
                          str(tmp_path))
    t = rec["groups"]["qkv_scale"]["tensors"][name]
    assert t["dtype"] == "F32" and t["shape"] == [108, 32]
    with safe_open(str(tmp_path / "qkv_scale.safetensors"), "pt") as f:
        s = f.get_tensor(name)
    assert torch.isfinite(s).all() and (s > 0).all()
