"""add_model_test — $0 proof of the model-onboarding core (M4).

  python3 phase0/add_model_test.py

No network, no GPU, no weights. Drives the pure parts of add_model:
  - fit math (params_per_layer / gb_per_layer / kv_gb_per_layer) is sane for dense AND MoE
  - build_row emits a row that passes registry validation
  - merge_into_source replaces-by-id (no dup) and appends new ids
  - add (merge+sign) yields a registry that verify_registry accepts, with a bumped version
  - M4 ACCEPTANCE (offline half): a brand-new model in a supported arch goes from "nothing"
    to a signed, verifiable registry row with ZERO engine code — just data.

The quantize + host + 3-stage smoke (steps 5-7) are the irreducible/metered remainder; this
proves everything add_model can do at $0.
"""
import copy
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phase0"))
import add_model as am          # noqa: E402
from shard import registry as reg   # noqa: E402
from shard import manifest as mf    # noqa: E402

passed = failed = 0


def ok(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}{(' ' + detail) if detail else ''}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


# a dense config (llama-ish) and an MoE config (deepseek/glm-ish)
DENSE = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32, "hidden_size": 4096,
         "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
         "intermediate_size": 14336}
MOE = {"architectures": ["Glm4MoeForCausalLM"], "num_hidden_layers": 78, "hidden_size": 5120,
       "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
       "intermediate_size": 12288, "moe_intermediate_size": 1536,
       "num_experts": 160, "n_shared_experts": 1}


def main():
    # ── fit math ──
    dpl = am.params_per_layer(DENSE)
    mpl = am.params_per_layer(MOE)
    ok(dpl > 0 and mpl > 0, "params_per_layer positive", f"(dense={dpl/1e6:.0f}M moe={mpl/1e6:.0f}M)")
    ok(mpl > dpl, "MoE layer has more params than dense layer")
    # nvfp4 ~0.55 B/param -> a 160-expert MoE layer is multiple GB
    g = am.gb_per_layer(MOE, "nvfp4")
    ok(g > 0, "gb_per_layer positive", f"({g} GB/layer)")
    ok(am.gb_per_layer(MOE, "bf16") > am.gb_per_layer(MOE, "nvfp4"),
       "bf16 heavier than nvfp4 per layer")
    kv = am.kv_gb_per_layer(MOE, ctx=16384)
    ok(kv > 0, "kv_gb_per_layer positive", f"({kv} GB/layer @16k)")
    ok(am.kv_gb_per_layer(MOE, 32768) > kv, "longer ctx -> more KV/layer")

    # ── build_row passes registry validation ──
    row = am.build_row(MOE, model_id="shard-newmoe-1", quant="nvfp4", adapter="generic-vllm",
                       worker_model="NewMoE-1", engine_path="/root/models/newmoe",
                       tokenizer_id="org/NewMoE-1", ctx=16384)
    ok(row["layerCount"] == 78, "row layerCount from config", f"({row['layerCount']})")
    ok(row["hfArch"] == "Glm4MoeForCausalLM", "row hfArch from config.architectures")
    # validate as a one-row registry
    reg._validate_rows({"schema": reg.SCHEMA, "models": [row]})
    ok(True, "build_row passes registry _validate_rows")

    # ── merge: replace by id, append new ──
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "models.src.json")
        out_path = os.path.join(d, "models.json")
        key_path = os.path.join(d, "pub.key")
        # seed a source with one existing model
        seed = {"schema": reg.SCHEMA, "version": 5, "models": [
            {"id": "shard-glm-5.2", "hfArch": "Glm4MoeForCausalLM", "workerModel": "GLM-5.2",
             "enginePath": "/root/models/GLM-5.2", "layerCount": 78, "gbPerLayer": 1.05,
             "kvGbPerLayer": 0.04, "quant": "nvfp4", "adapter": "glm-nvfp4",
             "tokenizerId": "zai-org/GLM-5.2", "chatTemplate": None, "weightManifestCid": "",
             "defaults": {"K": 4, "depth": 2, "draftCtx": 16384}}]}
        with open(src_path, "w") as f:
            json.dump(seed, f)

        # merge a NEW model -> appended
        merged = am.merge_into_source(src_path, row)
        ids = [m["id"] for m in merged["models"]]
        ok(ids == ["shard-glm-5.2", "shard-newmoe-1"], "new id appended", f"({ids})")

        # merge an UPDATE to an existing id -> replaced in place, no dup
        upd = copy.deepcopy(seed["models"][0]); upd["layerCount"] = 78; upd["gbPerLayer"] = 1.11
        merged2 = am.merge_into_source(src_path, upd)
        glm_rows = [m for m in merged2["models"] if m["id"] == "shard-glm-5.2"]
        ok(len(glm_rows) == 1 and glm_rows[0]["gbPerLayer"] == 1.11,
           "existing id replaced in place (no dup)")

        # ── full add (merge + sign) via cmd_add over an argparse Namespace ──
        import argparse
        a = argparse.Namespace()
        a.hf = None; a.dir = None
        # cmd_add reads config from --hf/--dir; inject a local dir with our MoE config.
        mdir = os.path.join(d, "model"); os.makedirs(mdir)
        with open(os.path.join(mdir, "config.json"), "w") as f:
            json.dump(MOE, f)
        a.dir = mdir
        a.id = "shard-newmoe-1"; a.quant = "nvfp4"; a.adapter = "generic-vllm"
        a.worker_model = "NewMoE-1"; a.engine_path = "/root/models/newmoe"
        a.tokenizer = "org/NewMoE-1"; a.chat_template = None; a.ctx = 16384
        a.src = src_path; a.key = key_path; a.out = out_path

        # capture stdout json
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            am.cmd_add(a)
        result = json.loads(buf.getvalue())
        ok(result["ok"] is True, "cmd_add reports ok")
        ok(result["version"] == 6, "version bumped 5 -> 6", f"({result['version']})")

        # the signed registry verifies against the printed pubkey, and BOTH models resolve
        pub = result["publisher_pubkey"]
        signed = reg.load_registry(out_path, expected_pubkey=pub)   # raises on failure
        by = reg.models_by_id(signed)
        ok(set(by) == {"shard-glm-5.2", "shard-newmoe-1"}, "both models in signed registry")
        ok(by["shard-newmoe-1"]["layerCount"] == 78 and by["shard-newmoe-1"]["adapter"] == "generic-vllm",
           "M4 acceptance: new model serving-ready in registry with ZERO engine code")

        # tamper the new row post-sign -> signature rejects (drift guard still holds)
        bad = copy.deepcopy(signed); bad["models"][-1]["layerCount"] = 999
        try:
            reg.verify_registry(bad, expected_pubkey=pub)
            ok(False, "tampered added-row rejected")
        except reg.RegistryError:
            ok(True, "tampered added-row rejected -> RegistryError")

    print(f"\n{'ALL' if failed == 0 else 'SOME FAILURES:'} {passed} PASS"
          + (f", {failed} FAIL" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
