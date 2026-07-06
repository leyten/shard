"""registry_test — $0 proof of the signed model registry (M1).

  python3 shard/registry_test.py

Proves: sign -> verify round-trip; pin mismatch rejected; a tampered signed field
(layerCount) rejected; an unknown adapter rejected; a duplicate id rejected; get_spec /
layer_count resolve. No GPU, no network — pure ed25519 over the canonical bytes.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard import manifest as mf
from shard import registry as reg

passed = failed = 0


def ok(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}{(' ' + detail) if detail else ''}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def raises(fn, name):
    try:
        fn()
    except reg.RegistryError as e:
        ok(True, name, f"-> {type(e).__name__}")
        return
    except Exception as e:  # wrong exception type is still a fail
        ok(False, name, f"raised {type(e).__name__}, expected RegistryError")
        return
    ok(False, name, "did not raise")


SRC = {
    "schema": reg.SCHEMA,
    "version": 7,
    "models": [
        {"id": "shard-glm-5.2", "hfArch": "Glm4MoeForCausalLM", "workerModel": "GLM-5.2",
         "enginePath": "/root/models/GLM-5.2", "layerCount": 78, "gbPerLayer": 1.05,
         "kvGbPerLayer": 0.04, "quant": "nvfp4", "adapter": "glm-nvfp4",
         "tokenizerId": "zai-org/GLM-5.2", "chatTemplate": None, "weightManifestCid": "",
         "defaults": {"K": 4, "depth": 2, "draftCtx": 16384}},
        {"id": "shard-gpt-oss-120b", "hfArch": "GptOssForCausalLM", "workerModel": "gpt-oss-120b",
         "enginePath": "/root/models/gpt-oss-120b", "layerCount": 36, "gbPerLayer": 0.95,
         "kvGbPerLayer": 0.03, "quant": "mxfp4", "adapter": "generic-vllm",
         "tokenizerId": "openai/gpt-oss-120b", "chatTemplate": None, "weightManifestCid": "",
         "defaults": {"K": 4, "depth": 2, "draftCtx": 16384}},
    ],
}


def main():
    priv = mf.gen_key()
    pub = mf.pub_b64(priv)
    signed = reg.sign_registry(copy.deepcopy(SRC), priv)

    # round-trip
    reg.verify_registry(signed, expected_pubkey=pub)
    ok(True, "sign->verify round-trip")
    ok(reg.layer_count("shard-glm-5.2", registry=signed) == 78, "glm layerCount=78")
    ok(reg.layer_count("shard-gpt-oss-120b", registry=signed) == 36, "gpt-oss layerCount=36")
    ok(reg.get_spec("not-a-model", registry=signed) is None, "unknown id -> None")
    glm = reg.get_spec("shard-glm-5.2", registry=signed)
    ok(glm is not None and glm["adapter"] == "glm-nvfp4", "glm adapter")

    # pin mismatch
    other = mf.pub_b64(mf.gen_key())
    raises(lambda: reg.verify_registry(signed, expected_pubkey=other), "pin mismatch rejected")

    # tamper a SIGNED field: flip gpt-oss layerCount back to the buggy 120 after signing.
    # This is exactly the gpt-oss drift the registry exists to stop — the signature must catch it.
    tampered = copy.deepcopy(signed)
    tampered["models"][1]["layerCount"] = 120
    raises(lambda: reg.verify_registry(tampered, expected_pubkey=pub),
           "tampered layerCount rejected")

    # unknown adapter (sign a bad source -> _validate_rows fires after sig passes)
    bad_adapter = copy.deepcopy(SRC)
    bad_adapter["models"][0]["adapter"] = "make-believe"
    s2 = reg.sign_registry(bad_adapter, priv)
    raises(lambda: reg.verify_registry(s2, expected_pubkey=pub), "unknown adapter rejected")

    # duplicate id
    dup = copy.deepcopy(SRC)
    dup["models"][1]["id"] = "shard-glm-5.2"
    s3 = reg.sign_registry(dup, priv)
    raises(lambda: reg.verify_registry(s3, expected_pubkey=pub), "duplicate id rejected")

    # missing required field
    miss = copy.deepcopy(SRC)
    del miss["models"][0]["enginePath"]
    s4 = reg.sign_registry(miss, priv)
    raises(lambda: reg.verify_registry(s4, expected_pubkey=pub), "missing field rejected")

    # unsigned
    raises(lambda: reg.verify_registry(copy.deepcopy(SRC), expected_pubkey=pub),
           "unsigned registry rejected")

    print(f"\n{'ALL' if failed == 0 else 'SOME FAILURES:'} {passed} PASS"
          + (f", {failed} FAIL" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
