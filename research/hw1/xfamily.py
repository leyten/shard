"""HW1 part D: the SAME MiMo layers on two device families (3090 sm_86 vs 5090 sm_120, optionally 4090).

  on each box:  python research/hw1/mimo_ops.py --mode dump --weights W --inputs golden.safetensors \
                    --dump out/<label>_dump.safetensors --out out/<label>_dump.json
                python research/hw1/mimo_ops.py --mode ops ... --out out/<label>_ops.json
  anywhere:     python research/hw1/xfamily.py --a out/5090_dump.json --b out/3090_dump.json \
                    --a-ops out/5090_ops.json --b-ops out/3090_ops.json --out out/xfamily.json

The dump is a fixed-input, layer-shaped op chain per layer (rmsnorm -> fused qkv -> sink attention
-> o_proj -> residual -> fp32 router -> experts -> residual; no RoPE -- the op chain, not the vendor
layer, which is R2's job). Every intermediate is saved, so the diff says WHERE families diverge
(the 3090 has no FP8/FP4 hardware: qkv is upcast at load, MXFP4 is dequantized in-kernel by Marlin).
Compare reports per tensor: max|a-b|, relative error, bit-equal?, router top-k set agreement; and
the speed ratio b/a at matched (op, params) from the two ops files.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw1_common as C  # noqa: E402

DUMP_T = 17
K10_REL, K10_AGREE = 2e-2, 0.97       # pre-registered (README K10)
K11_RATIO = 2.5                       # pre-registered (README K11); b / a with a = the 5090


def dump(res, Ls, experts, attn, d, device, inputs, path, router, rms):
    from safetensors.torch import save_file
    if inputs is not None:
        x = inputs.reshape(-1, inputs.shape[-1])[:DUMP_T]
    else:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(DUMP_T, d["hidden"], generator=g) * 0.5
    x = x.bfloat16().to(device)
    out = {"input": x}
    use_attn = attn
    for L in Ls:
        h = x
        a = rms(h, L.ln1, d["eps"])
        qkv = a @ L.wqkv.t()
        nq, nk = d["heads"] * d["qk"], L.nkv * d["qk"]
        q = qkv[:, :nq].reshape(DUMP_T, d["heads"], d["qk"]).contiguous()
        k = qkv[:, nq:nq + nk].reshape(DUMP_T, L.nkv, d["qk"]).contiguous()
        v = qkv[:, nq + nk:].reshape(DUMP_T, L.nkv, d["v"]).contiguous()
        win = d["window"] if L.kind == "swa" else 0
        if use_attn is not None and (L.kind == "swa" or use_attn.backend == "torch_sink"):
            o = use_attn(q, k, v, L.sinks, win, 0)
            ab = use_attn.backend
        else:
            import mimo_ops
            o = mimo_ops.attn_ref(q, k, v, L.sinks, win, 0)
            ab = "torch_sink"
        h = h + o.reshape(DUMP_T, -1) @ L.wo.t()
        p = f"L{L.li}."
        out.update({p + "qkv": qkv, p + "attn": o, p + "post_attn": h})
        if L.moe:
            m = rms(h, L.ln2, d["eps"])
            ids, w = router(L, m)
            y = experts[L.li](m, ids, w)
            h = h + y
            out.update({p + "router_ids": ids.to(torch.int32), p + "router_w": w, p + "experts": y})
        out[p + "out"] = h
        res.record("D", "dump_layer", {"layer": L.li, "T": DUMP_T}, attn_backend=ab,
                   experts_backend=experts[L.li].backend if L.moe else None)
    cpu = {k: v.detach().cpu().clone().contiguous() for k, v in out.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(cpu, path)
    res.doc["derived"]["dump_path"] = os.path.abspath(path)
    res.doc["derived"]["dump_sha256"] = {k: hashlib.sha256(v.view(torch.uint8).numpy().tobytes()
                                                           if v.dtype == torch.bfloat16 else
                                                           v.numpy().tobytes()).hexdigest()
                                         for k, v in cpu.items()}
    return path


def _key(r):
    return (r["id"], r["op"], tuple(sorted((k, str(v)) for k, v in r["params"].items())))


def compare(a_doc, b_doc, a_ops=None, b_ops=None, out=None):
    from safetensors.torch import load_file
    res = C.Results("xfamily", "compare", device="cpu", stub=a_doc["stub"] or b_doc["stub"])
    fa, fb = a_doc["device"]["family"], b_doc["device"]["family"]
    res.doc["derived"].update(a=a_doc["device"], b=b_doc["device"])
    ta = load_file(a_doc["derived"]["dump_path"])
    tb = load_file(b_doc["derived"]["dump_path"])
    if not torch.equal(ta["input"], tb["input"]):
        res.check("same_input", False, "the two dumps were made from different inputs -- diff meaningless")
    else:
        res.check("same_input", True, "identical input bytes")
    for k in sorted(set(ta) & set(tb)):
        if k == "input":
            continue
        a, b = ta[k], tb[k]
        prm = {"tensor": k, "a": fa, "b": fb}
        if "router_ids" in k:
            agree = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)) / a.numel()
            res.record("D", "xfamily_router", prm, topk_set_agreement=agree, exact=bool(torch.equal(a, b)))
            continue
        af, bf = a.float(), b.float()
        mx = (af - bf).abs().max().item()
        rel = mx / (af.abs().max().item() + 1e-12)
        res.record("D", "xfamily_diff", prm, max_abs=mx, rel=rel, bit_equal=bool(torch.equal(a, b)))
    outs = [r for r in res.doc["records"] if r["op"] == "xfamily_diff" and r["params"]["tensor"].endswith(".out")]
    routs = [r for r in res.doc["records"] if r["op"] == "xfamily_router"]
    if outs:
        worst = max(r["rel"] for r in outs)
        agree = min((r["topk_set_agreement"] for r in routs), default=1.0)
        res.check("K10", worst <= K10_REL and agree >= K10_AGREE,
                  f"{fb} vs {fa}: worst layer-out rel {worst:.3e} (<= {K10_REL}), "
                  f"router top-k agreement {agree:.3f} (>= {K10_AGREE})", rel=worst, agreement=agree)
    if a_ops and b_ops:
        la, lb = a_ops["derived"].get("layer_ms_T9"), b_ops["derived"].get("layer_ms_T9")
        if la and lb:
            res.check("K11", lb / la <= K11_RATIO, f"{fb}/{fa} per-layer time at T=9: {lb / la:.2f}x "
                      f"(<= {K11_RATIO})", ratio=lb / la)

        ia = {_key(r): r for r in a_ops["records"] if "ms" in r and not r.get("skipped")}
        for r in b_ops["records"]:
            if "ms" in r and not r.get("skipped") and _key(r) in ia:
                ra = ia[_key(r)]
                res.record("D", "xfamily_speed", dict(r["params"], op=r["op"], a=fa, b=fb),
                           a_ms=ra["ms"], b_ms=r["ms"], ratio_b_over_a=r["ms"] / ra["ms"],
                           a_backend=ra.get("backend"), b_backend=r.get("backend"))
    if out:
        res.write(out)
    return res.doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-ops")
    ap.add_argument("--b-ops")
    ap.add_argument("--out", required=True)
    x = ap.parse_args(argv)
    compare(C.load(x.a), C.load(x.b), C.load(x.a_ops) if x.a_ops else None,
            C.load(x.b_ops) if x.b_ops else None, x.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
