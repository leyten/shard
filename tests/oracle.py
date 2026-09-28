"""The model-agnostic half of the CPU fake-ring harness: the position oracle, the target-sequence
builders and the fake tokenizers. Nothing here knows a model, an engine or a wire protocol, so a
fake mesh or a runtime-conformance suite can hold itself to the same bar the M2.5 fake ring
(tests/fake_ring.py, which re-exports all of it) does, without importing m25_pipe.

THE ORACLE: a fixed target token sequence T indexed by ABSOLUTE position. For any verify slot at
absolute position p the reply token is T[p+1] — the model's greedy argmax depends only on the
position, never on the (possibly wrong) draft token in that slot. This preserves speculative-accept
accounting exactly: a draft token is accepted iff it equals the true continuation, and every
committed stream must therefore be a prefix of T after the prompt. Losslessness ==
`output_ids == T[P:P+n]`, assertable to the token.

Stdlib only: importing this bootstraps nothing.
"""


def oracle_token(T, p):
    """The target's greedy argmax AT absolute position p: T[p+1], clamped past the end so deep
    in-flight speculation never crashes the ring — those replies get discarded anyway."""
    return T[p + 1] if p + 1 < len(T) else T[-1]


class FakeTok:
    """The minimal tokenizer surface coordinate_pipe touches: eos (an id NOT in T so generation never
    stops early), decode() for the result text, and apply_chat_template() so render_ids() yields
    exactly the chosen prompt prefix of T."""

    def __init__(self, prompt_ids, eos_id=10 ** 6):
        self._ids = [int(t) for t in prompt_ids]
        self.eos_token_id = eos_id

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, return_dict=True, **kw):
        return {"input_ids": list(self._ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False):   # render_ids' reasoning=False path (unused but cheap)
        return {"input_ids": []}


class FakeTokB:
    """FakeTok for the rows coordinator: ONE tokenizer, per-stream prompts — a message's content is
    the stream index and apply_chat_template returns that stream's prompt prefix."""

    def __init__(self, prompts, eos_id=10 ** 6):
        self._prompts = [list(map(int, p)) for p in prompts]
        self.eos_token_id = eos_id

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, return_dict=True, **kw):
        return {"input_ids": list(self._prompts[int(messages[-1]["content"])])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": []}


# ---- target-sequence builders ------------------------------------------------------------------
# Token ids stay < 100 (the synthetic EAGLE head's target vocab) so the real EagleDrafter's embed
# lookups are always in range on every path.

def novel_T(n, seed=1234):
    """Pseudo-random tokens — no n-gram structure, the drafter is blind."""
    import random
    rng = random.Random(seed)
    return [rng.randrange(2, 100) for _ in range(n)]


def repetitive_T(n, seed=99, phrase_len=30):
    """A ~30-token phrase repeated verbatim — the n-gram drafter matches long runs (rag-quote-like)."""
    import random
    rng = random.Random(seed)
    phrase = [rng.randrange(2, 100) for _ in range(phrase_len)]
    out = []
    while len(out) < n:
        out.extend(phrase)
    return out[:n]


def trap_T(n, seed=7, break_at=168, novel_len=30):
    """Repetition that BREAKS once: phrase A repeats through `break_at`, then a novel run, then a
    different phrase B repeats. The n-gram drafter confidently speculates A's continuation across the
    break -> a mid-stream divergence with chunks in flight (the discard-bookkeeping trap)."""
    import random
    rng = random.Random(seed)
    a = [rng.randrange(2, 100) for _ in range(26)]
    b = [rng.randrange(2, 100) for _ in range(26)]
    out = []
    while len(out) < break_at:
        out.extend(a)
    out = out[:break_at]
    out += [rng.randrange(2, 100) for _ in range(novel_len)]
    while len(out) < n:
        out.extend(b)
    return out[:n]
