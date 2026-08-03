#!/usr/bin/env python3
"""NO-GPU config probe for the MiniMax-M2.5 NVFP4 port (S0).

Reads the REAL config of the NVFP4 checkpoint(s) and prints every value the
m25_stage FusedMoE build + the scheduler/fetch byte budget depend on, so we
don't guess (a wrong router-grouping silently routes to the wrong experts).

Downloads only small JSON/text side-files via the HF resolve API + HF token;
no weights, no GPU.
"""
import json, sys, urllib.request, urllib.error

TOKEN = open("/root/.hf_token").read().strip()
REPOS = ["lukealonso/MiniMax-M2.5-NVFP4", "nvidia/MiniMax-M2.5-NVFP4"]
SMALL = ["config.json", "hf_quant_config.json", "generation_config.json"]


def fetch(repo, fname):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        return f"__HTTP_{e.code}__".encode()
    except Exception as e:  # noqa
        return f"__ERR_{e}__".encode()


def getj(repo, fname):
    raw = fetch(repo, fname)
    if raw.startswith(b"__"):
        return None, raw.decode()
    try:
        return json.loads(raw), None
    except Exception as e:  # noqa
        return None, f"__BADJSON_{e}__"


def show(label, val):
    print(f"  {label:34s}: {val}")


def probe(repo):
    print(f"\n{'='*78}\n# {repo}\n{'='*78}")
    cfg, err = getj(repo, "config.json")
    if cfg is None:
        print(f"  config.json -> {err}  (repo may not exist / be gated)")
        return
    g = cfg.get

    print("\n[ core arch ]")
    for k in ("model_type", "num_hidden_layers", "hidden_size",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "max_position_embeddings", "rope_theta",
              "tie_word_embeddings", "rms_norm_eps"):
        show(k, g(k, "—"))
    # rotary / partial-rope
    for k in ("rotary_dim", "partial_rotary_factor", "qk_nope_head_dim",
              "qk_rope_head_dim", "use_qk_norm", "qk_norm_type"):
        if k in cfg:
            show(k, cfg[k])
    rs = g("rope_scaling")
    if rs:
        show("rope_scaling", rs)

    print("\n[ MoE router — the build-critical block ]")
    for k in ("num_local_experts", "num_experts", "n_routed_experts",
              "num_experts_per_tok", "moe_intermediate_size",
              "intermediate_size", "shared_intermediate_size",
              "n_shared_experts", "num_shared_experts",
              "first_k_dense_replace", "scoring_func",
              "use_grouped_topk", "n_group", "topk_group",
              "norm_topk_prob", "routed_scaling_factor",
              "router_aux_loss_coef", "use_routing_bias",
              "use_expert_bias", "e_score_correction_bias"):
        if k in cfg:
            show(k, cfg[k])

    print("\n[ MTP / multi-token-prediction ]")
    for k in ("num_mtp_modules", "num_nextn_predict_layers",
              "mtp_transformer_layers", "mtp_num_layers"):
        if k in cfg:
            show(k, cfg[k])

    print("\n[ attn_type_list (confirm all-standard) ]")
    atl = g("attn_type_list") or g("layer_types")
    if atl is not None:
        uniq = sorted(set(atl)) if isinstance(atl, list) else atl
        show("len / unique values", f"{len(atl) if isinstance(atl,list) else '?'} / {uniq}")

    print("\n[ NVFP4 quant layout (hf_quant_config.json) ]")
    q, qerr = getj(repo, "hf_quant_config.json")
    if q is None:
        show("hf_quant_config.json", qerr)
    else:
        # modelopt-style: producer + quantization{quant_algo, kv_cache, exclude_modules...}
        qd = q.get("quantization", q)
        for k in ("quant_algo", "kv_cache_quant_algo", "group_size",
                  "exclude_modules"):
            if k in qd:
                v = qd[k]
                if k == "exclude_modules" and isinstance(v, list):
                    v = f"{len(v)} modules, e.g. {v[:6]}"
                show(k, v)
        if "exclude_modules" not in qd and "ignore" in q:
            show("ignore", f"{len(q['ignore'])} e.g. {q['ignore'][:6]}")

    print("\n[ generation defaults ]")
    gen, _ = getj(repo, "generation_config.json")
    if gen:
        for k in ("temperature", "top_p", "top_k"):
            if k in gen:
                show(k, gen[k])


def probe_weight_map(repo):
    print(f"\n{'='*78}\n# {repo}  —  weight_map / tensor namespace\n{'='*78}")
    idx, err = getj(repo, "model.safetensors.index.json")
    if idx is None:
        print(f"  index.json -> {err}")
        return
    keys = list(idx.get("weight_map", {}).keys())
    print(f"  total tensors: {len(keys)}")
    def has(sub):
        return sum(1 for k in keys if sub in k)
    for pat in ("experts", "shared_expert", "mtp", "nextn",
                "embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj",
                "q_norm", "k_norm", "qk_norm", "e_score_correction_bias",
                "weight_scale_2", "input_scale"):
        n = has(pat)
        if n:
            ex = next((k for k in keys if pat in k), "")
            show(f"contains '{pat}' (x{n})", ex)
    # detect MTP / extra-layer namespace beyond num_hidden_layers
    import re
    layer_ids = sorted({int(m.group(1)) for k in keys
                        for m in [re.search(r"model\.layers\.(\d+)\.", k)] if m})
    if layer_ids:
        show("model.layers.N range", f"{layer_ids[0]}..{layer_ids[-1]} ({len(layer_ids)} layers)")
    mtp_keys = [k for k in keys if "mtp" in k or "nextn" in k][:3]
    if mtp_keys:
        show("MTP tensor sample", mtp_keys)


if __name__ == "__main__":
    for repo in REPOS:
        probe(repo)
    # weight map only for the primary (pinned) ckpt
    probe_weight_map(REPOS[0])
    print("\nDONE.")
