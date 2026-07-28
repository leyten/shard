"""Which tensors of a checkpoint belong to a stage's block — read off the checkpoint's OWN names.

A pipeline stage owns a contiguous layer block [lo:hi), plus the embedding at the head and the
final norm + output head at the tail. Three sites have to answer "does this tensor belong to my
stage?": the verified fetch's FILE selection (shard/fetch.py), the MLX range loader's TENSOR
selection (shard/mlx_runtime.py), and the transformers device_map's MODULE selection
(phase0/pipeline.py). All three used to answer it by hardcoding one family's key namespace
("model.layers.{j}.", "model.embed_tokens", "model.norm", "lm_head"), which made onboarding a
model a code change and failed on a checkpoint that namespaces its tensors differently (a
multimodal text stack under "language_model.model.", a text_config inner model): the fetch
matched nothing and pulled an empty block, the loaders matched nothing and refused to load.

The number a checkpoint gives a decoder layer lives INSIDE the tensor name; only the namespace
around it is the checkpoint's own. So match the numbered component, tolerate any prefix, and
share one classification between the site that fetches bytes and the sites that load them —
they must agree by construction, or a node pulls one set of files and materializes another.

Accepted cost (documented with the fetch change): a checkpoint carrying a SECOND numbered tower
(a vision or audio encoder with its own `layers.<n>.`) over-selects — its blocks are attributed
to the text stack's ranges. That direction costs bytes; the other direction is a MISSING weight
behind a valid receipt, so selection takes the union. Be honest about where that lands: the fetch
pulls extra files, while a LOADER that materializes extra tensors trips its own completeness
audit — such a checkpoint fetches, but placing its towers is still a per-model decision. A
device_map can't union at all (it must name ONE stack), which is what `namespace` resolves.
"""
import re

# "model.layers.0.", "text_model.layers.0." and "language_model.model.layers.0." are all in the
# wild. Match the numbered component and tolerate any prefix; a number in another component
# ("...experts.3.w1.weight") is not a layer number.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Boundary tensors carry no layer number: the same prefix-tolerant match on the components the
# loader places on the head (embedding) and on the tail (final norm, output head).
_EMBED_RE = re.compile(r"(?:^|\.)embed_tokens(?:\.|$)")
_NORM_RE = re.compile(r"(?:^|\.)norm(?:\.|$)")
_HEAD_RE = re.compile(r"(?:^|\.)lm_head(?:\.|$)")

LAYER_PATTERN = _LAYER_RE.pattern     # for the errors that name what a checkpoint failed to match

# The HF-standard layout, i.e. the module paths the loaders used to hardcode — still the answer
# for every checkpoint in that namespace, now as a fallback rather than an assumption.
HF_LAYOUT = {"inner": "model", "layers": "model.layers", "embed": "model.embed_tokens",
             "norm": "model.norm", "lm_head": "lm_head"}


def layer_of(tensor: str) -> int | None:
    """The decoder layer a tensor belongs to per the checkpoint's own naming, or None for a
    boundary/auxiliary tensor (embedding, final norm, output head, rotary cache). Numbers in
    other components ("...experts.3.w1.weight") are not layer numbers — only `layers.<n>.` is."""
    m = _LAYER_RE.search(tensor)
    return int(m.group(1)) if m else None


def boundary_of(tensor: str) -> str | None:
    """Which stage BOUNDARY a tensor belongs to — "embed" (the head's embedding, which a tied
    tail reads its logits through), "norm" or "lm_head" (the tail's) — or None for a tensor that
    carries a layer number or sits at no boundary (a rotary cache, a vision tower's stem).

    A layer tensor is never a boundary tensor (a per-layer "input_layernorm" belongs to its
    layer, not to the tail), and the order here is the precedence: one classification, so the
    fetch's files and the loaders' tensors cannot disagree about an edge weight."""
    if _LAYER_RE.search(tensor):
        return None
    if _EMBED_RE.search(tensor):
        return "embed"
    if _NORM_RE.search(tensor):
        return "norm"
    if _HEAD_RE.search(tensor):
        return "lm_head"
    return None


def _path_at(tensor: str, m: re.Match, drop_number: bool = False) -> str:
    """The MODULE path a matched component names: the tensor name truncated after the match,
    without its trailing dot ("language_model.model.embed_tokens.weight" -> ".embed_tokens" ->
    "language_model.model.embed_tokens"). `drop_number` also drops the layer index, leaving the
    layer LIST ("...layers.3." -> "...layers")."""
    path = tensor[:m.end()].rstrip(".")
    return path.rsplit(".", 1)[0] if drop_number else path


def _decoder_tower(towers: dict, edges: dict) -> str:
    """Which numbered tower is the DECODER, when a checkpoint has more than one (a vision or
    audio encoder numbers its blocks `layers.<n>.` too). The decoder is the tower the token
    boundaries hang off: score each by how many of the embedding / final norm live under it,
    and only on a tie fall back to depth. Depth alone gets it WRONG on real checkpoints —
    Qwen2-Audio's Whisper encoder is 32 layers against a 28-layer text decoder."""
    def parent(path):
        return path.rsplit(".", 1)[0] if "." in path else ""

    def score(path):
        inner = parent(path)
        votes = sum(any(parent(c) == inner for c in edges[k]) for k in ("embed", "norm"))
        return votes, len(towers[path])
    return max(towers, key=score)


def namespace(names) -> dict:
    """Where a checkpoint puts the pieces a stage is split on, as MODULE paths derived from its
    own tensor names: the decoder-layer list ("layers"), the module holding it ("inner" — where
    the embedding, final norm and rotary cache live), and the boundary modules ("embed", "norm",
    "lm_head"). This is what a device_map is keyed by.

    Empty `names` (no index to read) -> HF_LAYOUT, i.e. exactly the names the loaders hardcoded;
    names that carry no `layers.<n>.` tensor at all -> ValueError, because a device_map derived
    from a namespace we could not read would silently place NOTHING on the GPU.

    A boundary the checkpoint does not carry (no lm_head.* on a tied model) falls back to its
    conventional name under the DERIVED stack, not under the hardcoded one. A second numbered
    tower can't be unioned here the way the fetch unions files — a device_map must name ONE
    stack — so the one the token boundaries hang off wins (see _decoder_tower).

    Caveat: transformers remaps checkpoint keys for some multimodal classes, so a derived module
    path can differ from the tensor name it came from. A dict device_map is taken as given (no
    coverage check), so the mismatch surfaces at the first module lookup rather than as a device
    error — loud either way, and no worse than the hardcoded names, which named nothing at all."""
    names = list(names)
    if not names:
        return dict(HF_LAYOUT)
    towers: dict[str, set] = {}
    edges: dict[str, set] = {"embed": set(), "norm": set(), "lm_head": set()}
    for n in names:
        m = _LAYER_RE.search(n)
        if m:
            towers.setdefault(_path_at(n, m, drop_number=True), set()).add(int(m.group(1)))
            continue
        b = boundary_of(n)
        if b == "embed":
            edges[b].add(_path_at(n, _EMBED_RE.search(n)))
        elif b == "norm":
            edges[b].add(_path_at(n, _NORM_RE.search(n)))
        elif b == "lm_head":
            edges[b].add(_path_at(n, _HEAD_RE.search(n)))
    if not towers:
        raise ValueError(f"no tensor name matches the layer pattern {LAYER_PATTERN!r} "
                         f"(e.g. {sorted(names)[:3]}) — cannot locate the decoder layers")
    ns = {"layers": _decoder_tower(towers, edges)}
    ns["inner"] = ns["layers"].rsplit(".", 1)[0] if "." in ns["layers"] else ""
    outer = ns["inner"].rsplit(".", 1)[0] if "." in ns["inner"] else ""
    for key, under, default in (("embed", ns["inner"], f"{ns['inner']}.embed_tokens"),
                                ("norm", ns["inner"], f"{ns['inner']}.norm"),
                                ("lm_head", outer, f"{outer}.lm_head")):
        # A boundary INSIDE the chosen stack wins over one a second tower happens to have
        # (a ViT's own final norm is not the tail's), then any the checkpoint does carry,
        # and only if it carries none (a tied model has no lm_head tensor) the conventional
        # name under the DERIVED stack. Sorting keeps the pick independent of key order.
        found = sorted(edges[key])
        inside = [c for c in found if c.startswith(f"{under}." if under else "")]
        ns[key] = (inside or found or [default.lstrip(".")])[0]
    return ns
