"""shard-models/1 — the single signed model registry (M1: kills the drift class of bug).

A model is defined EXACTLY ONCE here. Both repos read this file:
  - shard (Python): phase0/plan_ring.py + scheduler_svc fit math, the engine launcher.
  - c0mpute (TypeScript): lib/orchestrator/modelRegistry.ts -> getShardModelSpec / isShardModel.

Before M1 the per-model facts (layer count, bytes/layer, engine path, quant) were smeared
across three places — plan_ring.MODEL_LAYERS, types.ts SHARD_MODELS, and a third
getLayerCountForModel() helper — and they drifted (gpt-oss 120-vs-36). One signed registry,
verified before use, makes that class of bug structurally impossible: editing a layer count
in one file changes behaviour in both repos, and a CI test asserts every id resolves.

Trust model (same as shard/manifest.py): the registry is signed with the publisher's
ed25519 key; consumers pin `expected_pubkey` and fail closed on any mismatch. This reuses
manifest.py's canonical() / sign / verify primitives byte-for-byte so the TS port only has
to match one canonicalization.

Schema (`schema: "shard-models/1"`):

    {
      "schema": "shard-models/1",
      "version": 3,                       # bump on every change; consumers cache by this
      "models": [
        {
          "id": "shard-glm-5.2",          # user-facing model id (job:submit `model`)
          "hfArch": "Glm4MoeForCausalLM", # config.architectures[0] — keys the generic adapter
          "workerModel": "GLM-5.2",       # the `model` string shard workers register with
          "enginePath": "/root/models/GLM-5.2",   # path the specpipe stages load on the box
          "layerCount": 78,               # transformer layers — receipts MUST tile [0:layerCount]
          "gbPerLayer": 1.05,             # model bytes/layer at the served quant (VRAM fit)
          "kvGbPerLayer": 0.04,           # KV bytes/layer at target ctx (VRAM fit)
          "quant": "nvfp4",               # served quant format: nvfp4 | mxfp4 | fp8 | ...
          "adapter": "glm-nvfp4",         # ModelRuntime impl (shard/node.py): glm-nvfp4 | generic-vllm | generic-hf
          "tokenizerId": "zai-org/GLM-5.2",       # tokenizer the coordinator formats/detokenizes with
          "chatTemplate": null,           # null = use the tokenizer's built-in chat template
          "weightManifestCid": "",        # CID of the signed weight manifest (manifest.py); "" until hosted
          "defaults": { "K": 4, "depth": 2, "draftCtx": 16384 }
        }
      ],
      "publisher_pubkey": "<base64 raw ed25519>",
      "signature":        "<base64 over canonical(registry \\ signature)>"
    }

Boundary law (docs/INTEGRATION.md): pure engine/control-plane. Knows layers/quant/paths,
nothing about accounts or $ZERO. c0mpute pins the pubkey and reads the rows; it never writes here.
"""
import json
import os

from shard import manifest as mf  # reuse canonical()/sign/verify/key helpers — one trust path

SCHEMA = "shard-models/1"

# Default on-box location of the signed registry. Override with SHARD_MODELS_JSON.
# NB: a dir literally named `models/` is gitignored (weights), so the committed registry
# lives in registry/ — the filename stays models.json (that's the schema's name).
DEFAULT_PATH = os.environ.get(
    "SHARD_MODELS_JSON",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry", "models.json"),
)

# Adapters the runtime knows how to instantiate (one ModelRuntime impl per id, shard/node.py).
# The CI cross-repo test asserts every registry row names one of these AND that this set is
# identical to c0mpute's KNOWN_ADAPTERS, so a typo'd adapter fails loudly (not at spawn time).
#   glm-nvfp4    — the tuned GLM-NVFP4 specialized path
#   generic-vllm — vLLM loader for production quant/MoE kernels (VllmRuntime, MODEL_RUNTIME.md)
#   generic-hf / hf — universal HF/Transformers fallback (phase0/hf_runtime.GenericHFRuntime)
KNOWN_ADAPTERS = {"glm-nvfp4", "generic-vllm", "generic-hf", "hf"}

# Required fields on every model row. Missing any => RegistryError (fail closed, not a silent 0).
REQUIRED_FIELDS = (
    "id", "hfArch", "workerModel", "enginePath",
    "layerCount", "gbPerLayer", "kvGbPerLayer",
    "quant", "adapter", "tokenizerId",
)


class RegistryError(Exception):
    """The registry failed to load/verify — bad signature, unknown schema, or a malformed
    row. Always raised (never a silent default) so callers fail closed."""


def canonical(registry: dict) -> bytes:
    """Bytes signed over: the registry minus its signature, sorted keys, compact. Delegates
    to manifest.canonical so the signing convention is identical across manifests + registry
    and the TS port only matches ONE canonicalization."""
    return mf.canonical(registry)


def sign_registry(registry: dict, priv) -> dict:
    """Stamp publisher_pubkey + signature into a copy and return it (manifest.sign_manifest)."""
    return mf.sign_manifest(registry, priv)


def _validate_rows(registry: dict) -> None:
    """Structural checks the signature can't give you: schema tag, unique ids, every required
    field present and sane, adapter known, layerCount a positive int."""
    if registry.get("schema") != SCHEMA:
        raise RegistryError(f"unknown registry schema {registry.get('schema')!r}")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise RegistryError("registry has no models")
    seen = set()
    for m in models:
        mid = m.get("id")
        if not mid:
            raise RegistryError("model row missing id")
        if mid in seen:
            raise RegistryError(f"duplicate model id {mid!r}")
        seen.add(mid)
        for f in REQUIRED_FIELDS:
            if m.get(f) in (None, ""):
                raise RegistryError(f"model {mid!r} missing required field {f!r}")
        lc = m.get("layerCount")
        if not isinstance(lc, int) or lc <= 0:
            raise RegistryError(f"model {mid!r} layerCount must be a positive int, got {lc!r}")
        if m.get("adapter") not in KNOWN_ADAPTERS:
            raise RegistryError(
                f"model {mid!r} names unknown adapter {m.get('adapter')!r}; known: {sorted(KNOWN_ADAPTERS)}")


def verify_registry(registry: dict, expected_pubkey: str | None = None) -> None:
    """Fail closed: raise RegistryError unless the signature is valid (and matches the pinned
    pubkey if given) AND every row is structurally sound. Same ed25519 + pin semantics as
    shard/manifest.py, but checked against OUR schema tag (shard-models/1)."""
    pub_b64 = registry.get("publisher_pubkey")
    sig_b64 = registry.get("signature")
    if not pub_b64 or not sig_b64:
        raise RegistryError("registry is unsigned")
    if expected_pubkey is not None and pub_b64 != expected_pubkey:
        raise RegistryError("publisher pubkey does not match the pinned key")
    if registry.get("schema") != SCHEMA:
        raise RegistryError(f"unknown registry schema {registry.get('schema')!r}")
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    import base64
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64))
        pub.verify(base64.b64decode(sig_b64), canonical(registry))
    except (InvalidSignature, ValueError, Exception) as e:  # noqa: B014 — fail closed on anything
        raise RegistryError(f"signature verification failed: {type(e).__name__}") from e
    _validate_rows(registry)


def load_registry(path: str | None = None, expected_pubkey: str | None = None,
                  verify: bool = True) -> dict:
    """Read + (by default) verify the signed registry from disk. Returns the parsed dict.

    expected_pubkey pins the publisher (recommended in prod). verify=False is for the
    signing tool only (it builds the unsigned dict, then signs)."""
    p = path or DEFAULT_PATH
    try:
        with open(p) as f:
            reg = json.load(f)
    except FileNotFoundError as e:
        raise RegistryError(f"registry not found at {p}") from e
    except json.JSONDecodeError as e:
        raise RegistryError(f"registry is not valid JSON: {e}") from e
    if verify:
        verify_registry(reg, expected_pubkey)
    return reg


def models_by_id(registry: dict) -> dict:
    """{id -> model row} for O(1) lookup."""
    return {m["id"]: m for m in registry.get("models", [])}


def get_spec(model_id: str, path: str | None = None, expected_pubkey: str | None = None,
             registry: dict | None = None) -> dict | None:
    """The model row for `model_id`, or None if it's not a registered ring model. Pass a
    pre-loaded `registry` to avoid re-reading/re-verifying per lookup."""
    reg = registry if registry is not None else load_registry(path, expected_pubkey)
    return models_by_id(reg).get(model_id)


def layer_count(model_id: str, **kw) -> int | None:
    """Convenience: just the layer count for a model id (the field that drifted)."""
    spec = get_spec(model_id, **kw)
    return spec["layerCount"] if spec else None
