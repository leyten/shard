"""add_model — the afternoon path: a HF/local model -> a signed registry row (M4).

Wraps the end-state checklist so onboarding a model in a runtime-supported arch is mechanical,
not a per-arch engine integration. What it does:

  1. read config.architectures, num_hidden_layers, hidden/kv dims, quant hint
  2. compute gbPerLayer + kvGbPerLayer for the scheduler fit (plan_ring / scheduler_svc)
  3. emit the models.json row (id, hfArch, layerCount, fit, quant, adapter, tokenizerId, ...)
  4. merge it into the source registry and re-sign (publish_registry)
  5. (operator step) quantize the weights + host the shards + publish the weight manifest
  6. (operator step) tokenizer + chat-template round-trip check on the coordinator
  7. (operator step) scheduler plan + a single 3-stage smoke gen + receipt verify

Steps 1-4 are pure + offline-provable (this file + add_model_test.py, $0). Steps 5-7 are the
irreducible per-model cost (quantization) + the metered fleet smoke — this CLI PRINTS the exact
commands for them and, where a registry + key are present, performs 1-4 directly.

  # derive a row from a HF config (no weight download) and print it
  python phase0/add_model.py derive --hf zai-org/GLM-5.2 --id shard-glm-5.2 \
      --quant nvfp4 --adapter glm-nvfp4 --bytes-per-param 0.5

  # derive AND merge+sign into the registry
  python phase0/add_model.py add --hf openai/gpt-oss-120b --id shard-gpt-oss-120b \
      --quant mxfp4 --adapter generic-vllm --bytes-per-param 0.5 \
      --src registry/models.src.json --key ~/.shard/publisher.key --out registry/models.json

Boundary law: control-plane only. Computes fit numbers + a registry row; never touches $.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from shard import registry as reg  # noqa: E402


# ── fit math: model bytes/layer + KV bytes/layer from a config dict ────────────
# These two numbers are what plan_ring / scheduler_svc size rings from. Getting them right at
# add time is what lets a new model schedule correctly on day one (cross-cutting requirement).
def bytes_per_param_for_quant(quant: str) -> float:
    """Approx stored bytes per weight parameter at a served quant. The 4-bit formats carry
    per-block scales, so the effective rate is a bit above the nominal 0.5 B/param — these are
    deliberately conservative so the VRAM fit never over-commits a card."""
    return {
        "nvfp4": 0.55,    # 4-bit + fp8 block scales
        "mxfp4": 0.55,    # 4-bit + e8m0 block scales (gpt-oss)
        "fp8": 1.05,      # 8-bit + scales
        "int8": 1.05,
        "bf16": 2.0,
        "fp16": 2.0,
        "fp32": 4.0,
    }.get(quant.lower(), 2.0)


def params_per_layer(cfg: dict) -> int:
    """Rough parameter count for ONE decoder layer, from the config. Covers both dense and MoE
    layers: attention (q/k/v/o) + the MLP, where an MoE layer's MLP is num_experts * expert MLP
    plus an optional shared expert. This is a sizing estimate for the VRAM fit, not an exact
    count — the scheduler adds headroom + boundary slack on top.
    """
    h = cfg["hidden_size"]
    n_heads = cfg.get("num_attention_heads", h // 128)
    head_dim = cfg.get("head_dim", h // n_heads)
    n_kv = cfg.get("num_key_value_heads", n_heads)

    # attention: q_proj [h, n_heads*head_dim] + k,v [h, n_kv*head_dim] + o_proj [n_heads*head_dim, h]
    q = h * n_heads * head_dim
    kv = 2 * h * n_kv * head_dim
    o = n_heads * head_dim * h
    attn = q + kv + o

    # MLP. MoE if num_experts/num_local_experts present, else dense.
    inter = cfg.get("intermediate_size", 4 * h)
    moe_inter = cfg.get("moe_intermediate_size", inter)
    n_exp = cfg.get("num_local_experts", cfg.get("num_experts", 0)) or 0
    # gate_up + down for one expert MLP (SwiGLU -> ~3 matrices of [h, inter])
    one_mlp = 3 * h * moe_inter if n_exp else 3 * h * inter
    if n_exp:
        mlp = n_exp * one_mlp
        # optional shared expert (DeepSeek/GLM style)
        n_shared = cfg.get("n_shared_experts", cfg.get("num_shared_experts", 0)) or 0
        if n_shared:
            mlp += n_shared * 3 * h * moe_inter
        mlp += h * n_exp  # router gate
    else:
        mlp = one_mlp
    return int(attn + mlp)


def gb_per_layer(cfg: dict, quant: str) -> float:
    """model bytes per layer in GB at the served quant."""
    return round(params_per_layer(cfg) * bytes_per_param_for_quant(quant) / 1e9, 4)


def kv_gb_per_layer(cfg: dict, ctx: int, kv_bytes: float = 2.0) -> float:
    """KV-cache bytes per layer in GB at context length `ctx`. 2 (K and V) * n_kv * head_dim *
    ctx * bytes-per-elem (default bf16=2). For MLA models this overestimates (the latent KV is
    far smaller) — conservative is correct for a fit bound."""
    h = cfg["hidden_size"]
    n_heads = cfg.get("num_attention_heads", h // 128)
    head_dim = cfg.get("head_dim", h // n_heads)
    n_kv = cfg.get("num_key_value_heads", n_heads)
    return round(2 * n_kv * head_dim * ctx * kv_bytes / 1e9, 4)


# ── config source ──────────────────────────────────────────────────────────────
def load_config_hf(repo: str) -> dict:
    """fetch config.json from HF without downloading weights."""
    tok = os.environ.get("HF_TOKEN", "")
    headers = {"User-Agent": "shard-add-model/1"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60).read())


def load_config_dir(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def build_row(cfg: dict, *, model_id: str, quant: str, adapter: str,
              worker_model: str, engine_path: str, tokenizer_id: str,
              ctx: int, chat_template: str | None = None,
              weight_manifest_cid: str = "", defaults: dict | None = None) -> dict:
    """assemble a registry row from a config + the operator's choices."""
    arch = (cfg.get("architectures") or ["unknown"])[0]
    layer_count = cfg["num_hidden_layers"]
    row = {
        "id": model_id,
        "hfArch": arch,
        "workerModel": worker_model,
        "enginePath": engine_path,
        "layerCount": int(layer_count),
        "gbPerLayer": gb_per_layer(cfg, quant),
        "kvGbPerLayer": kv_gb_per_layer(cfg, ctx),
        "quant": quant,
        "adapter": adapter,
        "tokenizerId": tokenizer_id,
        "chatTemplate": chat_template,
        "weightManifestCid": weight_manifest_cid,
        "defaults": defaults or {"K": 4, "depth": 2, "draftCtx": ctx},
    }
    return row


def merge_into_source(src_path: str, row: dict) -> dict:
    """load the unsigned source registry, replace-or-append the row by id, return the dict."""
    with open(src_path) as f:
        src = json.load(f)
    src.pop("signature", None)
    src.pop("publisher_pubkey", None)
    models = src.setdefault("models", [])
    for i, m in enumerate(models):
        if m.get("id") == row["id"]:
            models[i] = row
            break
    else:
        models.append(row)
    return src


def _next_commands(row: dict) -> str:
    """the operator steps add-model can't do for you (the irreducible + metered work)."""
    mid, quant, eng = row["id"], row["quant"], row["enginePath"]
    return (
        "\nNEXT (operator steps — the irreducible per-model cost + the metered smoke):\n"
        f"  1. QUANTIZE weights to {quant} (see docs/runbooks/add-a-model.md recipe per format),\n"
        f"     host the shards, then publish the weight manifest + CID:\n"
        f"       python phase0/publish_manifest.py --dir {eng} --key ~/.shard/publisher.key \\\n"
        f"           --out registry/manifests/{mid}.json\n"
        f"     put the printed CID in the row's weightManifestCid and re-sign the registry.\n"
        f"  2. TOKENIZER round-trip on the coordinator (a wrong chat template silently degrades\n"
        f"     quality with NO error — treat a failing round-trip as a hard blocker):\n"
        f"       python phase0/tokenizer_roundtrip.py --model {mid}\n"
        f"  3. PLAN + 3-STAGE SMOKE + receipt verify (metered fleet, ~$2/hr/box):\n"
        f"       python phase0/plan_ring.py --ids <a,b,c> --model {mid}\n"
        f"       python phase0/launch.py --model {mid} --stages <...> --layers <...> --receipts\n"
    )


def cmd_derive(a):
    cfg = load_config_hf(a.hf) if a.hf else load_config_dir(a.dir)
    eng = a.engine_path or (a.dir or f"/root/models/{a.id}")
    row = build_row(cfg, model_id=a.id, quant=a.quant, adapter=a.adapter,
                    worker_model=a.worker_model or a.id,
                    engine_path=eng, tokenizer_id=a.tokenizer or a.hf or a.id,
                    ctx=a.ctx, chat_template=a.chat_template)
    print(json.dumps(row, indent=2))
    print(_next_commands(row), file=sys.stderr)


def cmd_add(a):
    if not (a.src and a.key and a.out):
        sys.exit("add requires --src, --key, --out")
    cfg = load_config_hf(a.hf) if a.hf else load_config_dir(a.dir)
    eng = a.engine_path or (a.dir or f"/root/models/{a.id}")
    row = build_row(cfg, model_id=a.id, quant=a.quant, adapter=a.adapter,
                    worker_model=a.worker_model or a.id,
                    engine_path=eng, tokenizer_id=a.tokenizer or a.hf or a.id,
                    ctx=a.ctx, chat_template=a.chat_template)
    if row["adapter"] not in reg.KNOWN_ADAPTERS:
        sys.exit(f"adapter {row['adapter']!r} not in KNOWN_ADAPTERS {sorted(reg.KNOWN_ADAPTERS)}")
    merged = merge_into_source(a.src, row)
    merged["version"] = int(merged.get("version", 0)) + 1     # bump so consumers refetch
    # write the updated source, then sign it via the registry path.
    with open(a.src, "w") as f:
        json.dump(merged, f, indent=2)
    from shard import manifest as mf
    priv = mf.load_key(a.key) if os.path.exists(a.key) else None
    if priv is None:
        os.makedirs(os.path.dirname(a.key) or ".", exist_ok=True)
        priv = mf.gen_key(); mf.save_key(priv, a.key)
        print(f"generated new publisher key -> {a.key}", file=sys.stderr)
    reg._validate_rows(merged)
    signed = reg.sign_registry(merged, priv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(signed, f, indent=2)
    reg.verify_registry(signed, expected_pubkey=mf.pub_b64(priv))
    print(json.dumps({"ok": True, "id": row["id"], "version": merged["version"],
                      "layerCount": row["layerCount"], "gbPerLayer": row["gbPerLayer"],
                      "kvGbPerLayer": row["kvGbPerLayer"], "out": a.out,
                      "publisher_pubkey": mf.pub_b64(priv)}, indent=2))
    print(_next_commands(row), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="onboard a model to the signed registry (M4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("derive", "add"):
        p = sub.add_parser(name)
        srcg = p.add_mutually_exclusive_group(required=True)
        srcg.add_argument("--hf", metavar="REPO", help="HF repo id (config fetched, no weights)")
        srcg.add_argument("--dir", metavar="PATH", help="local checkpoint dir")
        p.add_argument("--id", required=True, help="user-facing model id (e.g. shard-glm-5.2)")
        p.add_argument("--quant", required=True, help="served quant: nvfp4|mxfp4|fp8|...")
        p.add_argument("--adapter", required=True, help="ModelRuntime adapter: glm-nvfp4|generic-vllm|generic-hf")
        p.add_argument("--worker-model", help="the `model` string workers register (default: id)")
        p.add_argument("--engine-path", help="on-box weight path (default: --dir or /root/models/<id>)")
        p.add_argument("--tokenizer", help="tokenizer id (default: --hf or id)")
        p.add_argument("--chat-template", help="explicit chat template ref (default: tokenizer built-in)")
        p.add_argument("--ctx", type=int, default=16384, help="target context for the KV fit")
        if name == "add":
            p.add_argument("--src", help="unsigned source registry to merge into")
            p.add_argument("--key", help="publisher ed25519 key (created if absent)")
            p.add_argument("--out", help="signed registry output path")
    a = ap.parse_args()
    (cmd_derive if a.cmd == "derive" else cmd_add)(a)


if __name__ == "__main__":
    main()
