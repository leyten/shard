"""Per-model chat rendering + tool parsing: the plugin seam.

docs/MODEL_RUNTIME.md lists "tokenizer / chat template / tool parse" as an INHERITED concern and
names the leak this closes: `m25_tools.py` hardcodes MiniMax's XML, and m25_pipe / m25_gateway import
it by name, so onboarding a second model meant editing both. Kimi-K3 makes that concrete — it shares
nothing with M2.5's format (no ChatML, no <think>, no chat_template.jinja at all; see k3_tools.py).

The seam is a lookup, not a framework. A family module exports the names the call sites already
import — `render_ids`, `parse_completion`, `to_openai_message`, `THINK_BEGIN`, `THINK_END`,
`TOOLCALL_BEGIN` — and a call site binds them once from `for_model(<model id>)` instead of from
`m25_tools`. Nothing about m25_tools changed; it just stopped being the only option.

  from model_tools import for_model
  T = for_model(os.environ.get("M25_MODEL_ID", "minimax-m2.5"))
  prompt_ids = T.render_ids(tok, messages, tools=tools)

Selection is by substring on the model id, longest rule first, because that id arrives from a
config, an env var or an OpenAI request body and is never normalized. An id matching no rule falls
back to m25: every ring in existence today either passes an M2.5 id or none, and this file must not
be the reason one of them changes behavior. `SHARD_TOOLS_FAMILY` overrides everything, for a box
serving a checkpoint under a name we did not anticipate.

Import is LAZY. k3_tools pulls in the vendored reference renderer, which a box serving M2.5 has no
reason to load, and a future family may want a dependency M2.5 does not have.
"""
import importlib
import os

# family -> (module name, id substrings that select it). Ordered longest-substring-first at match
# time, so "minimax-m2.5" cannot be stolen by a shorter rule.
FAMILIES = {
    "m25": ("m25_tools", ("minimax", "m2.5", "m25")),
    "k3": ("k3_tools", ("kimi-k3", "kimi_k3", "kimik3", "k3")),
}
DEFAULT_FAMILY = "m25"

_CACHE = {}


def family_for(model_id):
    """Model id -> family key. Case-insensitive substring match, longest rule wins."""
    forced = os.environ.get("SHARD_TOOLS_FAMILY")
    if forced:
        if forced not in FAMILIES:
            raise ValueError(f"SHARD_TOOLS_FAMILY={forced!r} is not one of {sorted(FAMILIES)}")
        return forced
    mid = (model_id or "").lower()
    best = None
    for fam, (_, pats) in FAMILIES.items():
        for p in pats:
            if p in mid and (best is None or len(p) > best[0]):
                best = (len(p), fam)
    return best[1] if best else DEFAULT_FAMILY


def for_model(model_id=None):
    """The tools module for a model id. Cached — call sites bind its names once at import."""
    fam = family_for(model_id)
    if fam not in _CACHE:
        _CACHE[fam] = importlib.import_module(FAMILIES[fam][0])
    return _CACHE[fam]
