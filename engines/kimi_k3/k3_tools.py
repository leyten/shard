"""Kimi-K3 chat rendering + tool-call parsing — the K3 half of the per-model tools seam.

The K3 analogue of m25_tools.py. Same four exports the call sites already import (`render_ids`,
`parse_completion`, `to_openai_message`, and the THINK/TOOLCALL markers), so phase0/model_tools.py
can hand either family to m25_pipe / m25_gateway unchanged.

WHY THIS FILE IS A WRAPPER, NOT A PORT
K3 ships NO chat_template.jinja. Rendering is a Python file in the checkpoint, `encoding_k3.py`, and
that file IS the definition of the XTML wire format — there is no spec to implement against. So the
render half here RENTS it (phase0/kimi_k3_ref/encoding_k3.py, vendored verbatim; docs/MODEL_RUNTIME.md
"tokenizer / chat template / tool parse -> inherited"). Every structural decision below — attribute
escaping, argument typing, which channel wraps what — is read off that file, never re-derived. The
only halves that are ours are the ones the reference does not contain: turning segments into ids
(which the tokenizer does, and we must not do differently), and parsing the model's OUTPUT back
(which nothing upstream does at all).

THE WIRE FORMAT, as the reference emits it
An assistant turn in history:

    <|open|>message role="assistant"<|sep|>
      <|open|>think<|sep|>REASONING<|close|>think<|sep|>
      <|open|>response<|sep|>CONTENT<|close|>response<|sep|>
      <|open|>tools<|sep|>
        <|open|>call tool="NAME" index="1"<|sep|>
          <|open|>argument key="K" type="string"<|sep|>VALUE<|close|>argument<|sep|>
        <|close|>call<|sep|>
      <|close|>tools<|sep|>
    <|close|>message<|sep|><|end_of_msg|>

(newlines added here for reading — the real render has none). The generation prompt is the first two
lines' openers with nothing after them, so a completion begins INSIDE the think channel and the
parser starts in phase "reasoning". Argument values are TYPED: `type` is one of
boolean/null/number/string/object/array, a string value is the raw text and every other type is its
JSON encoding (`encoding_k3._xtml_type` / `_xtml_value`). An `arguments` string the renderer could
not parse as JSON is emitted instead as `<|open|>json type="object"<|sep|>RAW<|close|>json<|sep|>`
(`_xtml_json_block`); parse_completion preserves that raw block so render -> parse -> render is
stable. Attribute values escape `&` then `"` (`_escape_attr_value`), so unescaping must undo
`&quot;` BEFORE `&amp;` or a literal `&quot;` in a tool name round-trips wrong.

THINKING IS ALWAYS ON, AND HISTORY MUST CARRY IT
K3 is trained with preserved thinking history: a multi-turn request has to echo back each assistant
turn's `reasoning_content` (and `tool_calls`), or quality degrades silently — the reference still
renders, it just renders an EMPTY think channel. `to_openai_message` therefore always emits
`reasoning_content`, so a client that replays our own messages satisfies the contract for free.
`check_thinking_history` reports the assistant turns that do not; `strict_thinking_history=True`
turns that into a hard error. The default is permissive because a caller seeding few-shot assistant
turns by hand is a legitimate (if degraded) request, and refusing it is a product call, not this
file's.

WHAT THIS FORMAT CANNOT EXPRESS (read before trusting a parse)
XTML escapes ATTRIBUTE values (`&` then `"`) but not element BODIES, so the format is not
self-delimiting and no parser of it can be lossless. An argument value that contains the literal
characters `<|close|>argument<|sep|>` ends its own argument early, and can go on to open another
argument or another whole `<|open|>call ...` — the rendered bytes are then indistinguishable from a
model that really made two calls. Likewise a visible answer that quotes `<|open|>tools<|sep|>` ends
the answer there and everything after it reads as a tool call. Both are properties of the wire, not
of this parser; they are pinned as tests so a future change cannot quietly make them worse.

What that does NOT include is the case that matters most: a USER cannot do this. Their text is a
non-special segment, so those characters become ordinary bytes and never a control token. The
exposure is a model persuaded to emit its own control markers, which is why a call is only accepted
when its own `<|close|>call<|sep|>` arrives before the next opener, and why the raw-json escape
hatch is honoured only when it is the entire call body. A tool name or argument key containing
`<|sep|>` is unrepresentable and its call (or argument) is dropped.

Two request-shaped gaps are the gateway's, not this file's: `reasoning_effort` is collapsed to a
boolean before it reaches render_ids (so every K3 request currently renders `thinking_effort=max`),
and `parse_completion`'s `reasoning` flag needs threading from the request to disambiguate a
`reasoning=False` completion truncated before any marker.

ASSISTANT PREFILL IS NOT IMPLEMENTED UPSTREAM
Moonshot's hosted API supports it; `encoding_k3.py` does not — `build_chat_segments` always closes
the last assistant message and opens a fresh one (HF discussion #99, open, unanswered). Rather than
invent a rendering the model was never trained on, `continue_final_message=True` raises. TODO: drop
the raise and pass through once the reference implements it.

  self-test:  python3 phase0/k3_tools.py
"""
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_k3_ref.encoding_k3 import (        # noqa: E402 — the vendored reference IS the renderer
    CLOSE_TOKEN, END_OF_MSG_TOKEN, OPEN_TOKEN, SEP_TOKEN,
    _VALID_THINKING_EFFORTS, _escape_attr_value, build_chat_segments,
)

OPEN, CLOSE, SEP, END_OF_MSG = OPEN_TOKEN, CLOSE_TOKEN, SEP_TOKEN, END_OF_MSG_TOKEN

# Channel markers. Named to match m25_tools' exports so the gateway's imports are family-agnostic;
# K3 additionally wraps visible content in its own channel, which M2.5 has no equivalent of.
THINK_BEGIN = f"{OPEN}think{SEP}"
THINK_END = f"{CLOSE}think{SEP}"
RESPONSE_BEGIN = f"{OPEN}response{SEP}"
RESPONSE_END = f"{CLOSE}response{SEP}"
TOOLCALL_BEGIN = f"{OPEN}tools{SEP}"
TOOLCALL_END = f"{CLOSE}tools{SEP}"
MESSAGE_END = f"{CLOSE}message{SEP}"

# Control-token ids of the pinned checkpoint (tokenizer_config.json @ 9f62e4e9). Recorded for the
# stage's benefit — nothing here hardcodes them; ids come from the tokenizer that is actually loaded.
CONTROL_IDS = {"[BOS]": 163584, "[EOS]": 163585, END_OF_MSG_TOKEN: 163586, OPEN_TOKEN: 163587,
               CLOSE_TOKEN: 163588, SEP_TOKEN: 163589, "[UNK]": 163838, "[PAD]": 163839}
VOCAB_SIZE = 163840                          # config.json text_config.vocab_size == n_words


class K3VocabError(ValueError):
    """A token id was produced that the embedding table cannot hold. See check_ids()."""


# ---------------------------------------------------------------- render


def render_segments(messages, tools=None, *, add_generation_prompt=True, thinking=True,
                    reasoning_effort="max", strict_thinking_history=False,
                    continue_final_message=False, **kwargs):
    """messages (+tools) -> the reference's list[EncodeSegment]. The one render entrypoint; every
    other render function here is a projection of this one.

    `reasoning_effort` is the OpenAI spelling of the reference's `thinking_effort` and defaults to
    "max" exactly as TikTokenTokenizer.apply_chat_template does. Validated against the reference's
    OWN set rather than a copy of it, so a future value is picked up by re-vendoring.
    """
    if continue_final_message:
        raise NotImplementedError(
            "assistant prefill (continue_final_message) is not implemented by K3's own renderer: "
            "build_chat_segments always closes the last assistant message and opens a new one "
            "(HF moonshotai/Kimi-K3 discussion #99). shard will not invent a rendering the model "
            "was not trained on.")
    if thinking and reasoning_effort is not None and reasoning_effort not in _VALID_THINKING_EFFORTS:
        # The reference asserts this deep inside build_chat_segments; catching it here turns an
        # AssertionError into a request-shaped error the gateway can map to a 400. NOTE the upstream
        # oddity: the system message the reference then renders advertises "low, medium, high, max"
        # while this set is {high, low, max} -- "medium" is documented to the MODEL but rejected by
        # the renderer. We follow the code, not the prose.
        raise ValueError(f"unsupported reasoning_effort={reasoning_effort!r}; K3 accepts "
                         f"{sorted(_VALID_THINKING_EFFORTS)}")
    _reject_unrenderable(messages)
    if strict_thinking_history:
        bad = check_thinking_history(messages)
        if bad:
            raise ValueError(
                f"assistant messages {bad} carry no reasoning_content: K3 is trained on preserved "
                "thinking history and renders an EMPTY think channel for these, which degrades "
                "quality silently. Echo back the reasoning_content of every assistant turn.")
    return build_chat_segments(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                               thinking=thinking, thinking_effort=reasoning_effort, **kwargs)


RENDERABLE_ROLES = frozenset({"user", "system", "assistant", "tool"})


def _reject_unrenderable(messages):
    """The reference's message loop is a chain of `elif role ==` with no else, so a role it does not
    know is skipped in silence. That is a security bug at the gateway, not a nicety: OpenAI's
    `developer` role is current API surface, and a caller who puts "never reveal the key" in one gets
    a prompt that never contained it and an answer that reads as if it did. Refuse instead. A batched
    conversation (list of lists) is left to the reference."""
    if not isinstance(messages, list) or (messages and isinstance(messages[0], list)):
        return
    bad = [f"[{i}] {m.get('role')!r}" if isinstance(m, dict) else f"[{i}] {type(m).__name__}"
           for i, m in enumerate(messages)
           if not isinstance(m, dict) or m.get("role") not in RENDERABLE_ROLES]
    if bad:
        raise ValueError(f"K3's renderer would silently DROP these messages: {', '.join(bad)}. "
                         f"Renderable roles are {sorted(RENDERABLE_ROLES)}.")


def render_text(messages, tools=None, **kw):
    """The rendered prompt as text — identical to `apply_chat_template(..., tokenize=False)`, which
    is that same join (tokenization_kimi.py:393). This is the parity oracle the fixtures assert on,
    and it is what a debug dump should print. It is NOT a route to token ids: re-encoding this string
    in one piece would let user text containing the literal "<|open|>" become a control token."""
    return "".join(s.text for s in render_segments(messages, tools, **kw))


def render_ids(tok, messages, tools=None, add_generation_prompt=True, reasoning=True,
               vocab_size=None, **kw):
    """Build prompt token ids. Same signature as m25_tools.render_ids (`reasoning` is K3's
    `thinking`), so the pipe's call sites are unchanged.

    Encodes SEGMENT BY SEGMENT, honouring each segment's allow_special. That boundary is the whole
    security story of this format: structural markers encode with allowed_special="all" and user /
    tool / argument text with disallowed_special=(), so a prompt containing the characters
    "<|open|>" tokenizes as ordinary bytes and cannot forge a channel
    (tokenization_kimi._encode_text_piece). Joining the segments into one string and encoding that
    would hand every user a control-token injection, which is why render_text is not this function.
    """
    n = _vocab_size(tok, vocab_size)
    if not n:
        raise ValueError(
            f"{type(tok).__name__} states no vocab_size, so the embedding bound cannot be checked. "
            "Pass vocab_size= (the config's) explicitly. Failing open here is how an out-of-range "
            "id reaches an embedding as a device-side assert -- see check_ids.")
    _assert_segment_boundary(tok)
    segs = render_segments(messages, tools, add_generation_prompt=add_generation_prompt,
                           thinking=reasoning, **kw)
    ids = []
    for s in segs:
        ids.extend(_encode_segment(tok, s.text, s.allow_special))
    return check_ids(ids, n, where="prompt")


def _assert_segment_boundary(tok):
    """Prove the tokenizer HONOURS allow_special_tokens rather than merely accepting the keyword.

    Encode a control marker both ways: as structure it is one id, as text it is several. A shim that
    forwards allowed_special="all" both times — an easy thing to write, since tiktoken raises on an
    unexpected special otherwise — returns the same ids, and every user gets a control-token
    injection while this module's docstring promises the opposite. A signature check cannot tell the
    two apart, and two encodes of eight characters is nothing next to rendering a prompt."""
    if _encode_segment(tok, OPEN, False) == _encode_segment(tok, OPEN, True):
        raise TypeError(
            f"{type(tok).__name__} ignores allow_special_tokens: it encodes {OPEN!r} identically as "
            "structure and as user text, so a prompt containing those characters could forge a "
            "channel. Use the checkpoint's TikTokenTokenizer, or a shim that passes "
            "disallowed_special=() for text segments.")


def _encode_segment(tok, text, allow_special):
    """Encode one segment. Prefers the reference tokenizer's own segment encoder; accepts anything
    that duck-types its `encode(text, allow_special_tokens=...)`. A tokenizer with neither is
    REFUSED rather than fallen back on: a plain HF `encode` cannot express the boundary above, so
    silently using it would turn a security property into a lie."""
    piece = getattr(tok, "_encode_text_piece", None)
    if piece is not None:
        return list(piece(text, allow_special_tokens=allow_special))
    enc = getattr(tok, "encode", None)
    if enc is not None and _takes_allow_special(enc):
        return list(enc(text, allow_special_tokens=allow_special))
    raise TypeError(
        f"{type(tok).__name__} cannot encode K3 chat segments: rendering needs an encoder that "
        "takes allow_special_tokens (Kimi's TikTokenTokenizer, or a shim over tiktoken). A plain "
        "HuggingFace tokenizer cannot separate structural markers from user text.")


def _takes_allow_special(fn):
    try:
        return "allow_special_tokens" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _vocab_size(tok, override):
    if override is not None:
        return int(override)
    n = getattr(tok, "vocab_size", None)
    return int(n) if n else None


def check_ids(ids, vocab_size, where="ids"):
    """Reject any id the embedding table cannot hold, with the offending values named.

    This exists because K3's shipped tokenizer can produce them. `tokenization_kimi.py:78-88`
    hardcodes a default `additional_special_tokens` list carried over from Kimi K2 —
    <|im_end|>, <|im_user|>, <|im_assistant|>, <|start_header_id|>, <|end_header_id|>,
    <|im_system|>, <|im_middle|> — none of which is in K3's vocab. When HuggingFace registers those
    as added tokens they land at 163840-163846, one past the declared vocab_size of 163840, and
    `tokenizer(text)` then returns an id that `tokenizer.encode(text)` does not (HF discussion #65,
    open). An out-of-range id is a bare CUDA device-side assert at the embedding — this turns it
    into a sentence naming the token. Cheap on a prompt, and the guard is version-independent: our
    own render path does not hit it (segments encode through tiktoken, which has never heard of
    those tokens), but a tokenizer we did not build is not ours to trust.
    """
    if vocab_size:
        bad = sorted({i for i in ids if not 0 <= i < vocab_size})
        if bad:
            raise K3VocabError(
                f"{where}: token ids {bad[:8]}{'...' if len(bad) > 8 else ''} are outside "
                f"[0, {vocab_size}) and would index past the embedding table. K3's tokenizer "
                "registers Kimi-K2 special tokens past its declared vocab_size (HF discussion #65); "
                "load it with the checkpoint's own additional_special_tokens.")
    return ids


def audit_tokenizer(tok, vocab_size=None):
    """The standing regression check for the above: every ADDED token must be addressable. Returns
    the offenders as [(surface, id)] — empty is the healthy answer, which is why an unknown
    vocab_size RAISES rather than returning the same empty list a clean tokenizer does."""
    n = _vocab_size(tok, vocab_size)
    if not n:
        raise ValueError(f"{type(tok).__name__} states no vocab_size; pass vocab_size= so a clean "
                         "audit cannot be confused with an audit that could not run.")
    added = getattr(tok, "get_added_vocab", None)
    added = added() if callable(added) else getattr(tok, "added_tokens_encoder", {}) or {}
    return sorted(((s, i) for s, i in added.items() if n and i >= n), key=lambda x: x[1])


def check_thinking_history(messages):
    """Indices of assistant messages that would render an EMPTY think channel. See module docstring:
    K3 is trained on preserved thinking history, so these degrade quality without erroring."""
    return [i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "assistant"
            and not str(m.get("reasoning_content") or m.get("reasoning") or "").strip()]


# ---------------------------------------------------------------- parse

_ATTR_RE = re.compile(r'([A-Za-z_][\w.:-]*)="([^"]*)"')


def _blocks(text, tag):
    """Yield (attrs, body, start, end) for each COMPLETE <|open|>TAG a<|sep|>body<|close|>TAG<|sep|>.

    Scanned, not regexed. The obvious pattern for this has two non-greedy groups, and a regex engine
    answers it by trying every (attrs-end x body-end) pair from every start — so a run of openers
    that never close costs seconds of CPU: 600 unclosed <|open|>call headers in 22 KiB took ten
    seconds. That is reachable from a request. A user's prompt carries those characters through as
    ordinary bytes (that is what the segment boundary is for), the model echoes them as ordinary
    tokens, and the gateway parses the decoded output inside the request handler. Scanning is linear.

    An opener with no close of its own before the NEXT opener is dropped, not stretched to the next
    close — otherwise a truncated call absorbs the following call's arguments and reports them under
    the first call's name, which is precisely the "executes the wrong thing" outcome that dropping
    incomplete calls exists to prevent.
    """
    om, cm = f"{OPEN}{tag}", f"{CLOSE}{tag}{SEP}"
    i = 0
    while True:
        s = text.find(om, i)
        if s < 0:
            return
        h = text.find(SEP, s + len(om))                 # end of the header
        if h < 0:
            return
        e = text.find(cm, h + len(SEP))
        if e < 0:
            return                                       # never closed: nothing complete follows
        nxt = text.find(om, h + len(SEP))
        if 0 <= nxt < e:                                 # this one never closed; resume at the next
            i = nxt
            continue
        yield text[s + len(om):h], text[h + len(SEP):e], s, e + len(cm)
        i = e + len(cm)


def _unescape_attr(v):
    """Invert encoding_k3._escape_attr_value, which does & then ". Undo in the reverse order or a
    literal "&quot;" in a value comes back as a quote."""
    return v.replace("&quot;", '"').replace("&amp;", "&")


def _attrs(s):
    return {k: _unescape_attr(v) for k, v in _ATTR_RE.findall(s)}


def _coerce(raw, typ):
    """Invert _xtml_value: a string value is the raw text, every other type is its JSON encoding.
    An unparseable non-string falls back to the raw text rather than losing it (truncated output)."""
    if typ == "string":
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


_CONTENT_END = (RESPONSE_END, TOOLCALL_BEGIN, MESSAGE_END, END_OF_MSG)


def _earliest(text, markers):
    """Index of the first of several markers to occur, or -1."""
    best = -1
    for m in markers:
        i = text.find(m)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _trim_partial(s, markers=_CONTENT_END + (RESPONSE_BEGIN, THINK_BEGIN, THINK_END)):
    """Drop a trailing fragment that is the beginning of a marker.

    Generation stops on a token boundary, not a marker boundary, so a completion cut short lands
    mid-marker often enough to matter: "...answer<|open|" would otherwise be handed to the user as
    the answer. Applied ONLY to a channel that never closed — a channel terminated by its own marker
    keeps every byte, so an answer that legitimately ends in "<" survives. Longest fragment first,
    or trimming "<|open|" would leave "<|ope"."""
    for n in range(min(len(s), max(len(m) for m in markers) - 1), 0, -1):
        if any(m.startswith(s[-n:]) for m in markers):
            return s[:-n]
    return s


def _think_is_open(body, reasoning):
    """No <|close|>think<|sep|> in the completion. Either the model is still reasoning, or the think
    channel was never opened — with thinking=False the generation prompt opens the RESPONSE channel
    instead, so the same marker-less text is content, not reasoning. A caller that knows passes
    `reasoning`; otherwise infer from the markers that did arrive. The one case inference cannot
    settle is a thinking=False completion truncated before any marker at all, which reads as
    reasoning; pass reasoning=False to be certain."""
    if reasoning is not None:
        return bool(reasoning)
    return not (RESPONSE_BEGIN in body or RESPONSE_END in body or TOOLCALL_BEGIN in body)


def parse_completion(text, reasoning=None):
    """Decoded K3 completion -> {reasoning_content, content, tool_calls}, the same dict m25_tools
    returns so the gateway's assembly is unchanged. tool_calls entries carry an extra `json` field:
    the raw <|open|>json<|sep|> block when the model emitted one, else None (see module docstring).

    Neither channel is stripped. M2.5 has to guess where its visible answer begins and trims
    whitespace to do it; K3 delimits both channels explicitly, so trimming would only lose bytes the
    model meant — and would put this out of step with StreamSplitter, which cannot retract a space
    it already streamed. The two agree exactly, and a test asserts it at every chunk size.

    Robust to what a real stream hands it: a leading <|open|>think<|sep|> echo or none; a missing
    <|close|>think<|sep|>; a response channel that never closed; multiple calls. A call whose
    <|close|>call<|sep|> has not arrived is DROPPED, never half-reported — a fragment of a tool call
    is worse than no tool call. See "WHAT THIS FORMAT CANNOT EXPRESS" in the module docstring for the
    shapes no parser of XTML can recover.
    """
    body = text
    reasoning_content = None
    closed_think = THINK_END in body
    if closed_think:
        head, _, body = body.partition(THINK_END)
        # ONE leading opener, not every occurrence: the echo can only be at position 0, and a global
        # replace silently deletes a <|open|>think<|sep|> the model wrote on purpose (asked to
        # explain its own chat format, it will).
        reasoning_content = head.removeprefix(THINK_BEGIN) or None
    elif _think_is_open(body, reasoning):
        reasoning_content = _trim_partial(body.removeprefix(THINK_BEGIN)) or None
        body = ""

    tail = body                              # everything after the think channel
    i = tail.find(RESPONSE_BEGIN)
    if i >= 0:
        content = tail[i + len(RESPONSE_BEGIN):]
    elif closed_think:
        content = ""                         # between the channels, or straight into tools: the
    else:                                    # gap carries only structure (StreamSplitter phase 1)
        content = tail                       # thinking=False: the prompt opened the response channel
    j = _earliest(content, _CONTENT_END)     # a response channel that never closed still ends at
    if j >= 0:                               # the tool block / message close -- never leak structure
        content = content[:j]
    else:
        content = _trim_partial(content)

    tool_calls = []
    k = tail.find(TOOLCALL_BEGIN)
    if k >= 0:
        for attrs, cbody, _s, _e in _blocks(tail[k:], "call"):
            name = _attrs(attrs).get("tool")
            if name:                         # a call with no tool attribute is not actionable --
                tool_calls.append({"name": name, **_call_arguments(cbody)})

    return {"reasoning_content": reasoning_content, "content": content, "tool_calls": tool_calls}


def _call_arguments(body):
    """One call's arguments.

    The raw-json escape hatch is honoured only when it is the ENTIRE call body. The renderer emits
    it alone (encoding_k3._render_assistant_segments takes that branch instead of the argument
    blocks), so a json block sitting beside real arguments is not a K3 render — it is a value that
    happens to contain those characters, and letting it win would discard every real argument.

    `json` is kept ONLY when the raw text is not valid JSON, which is exactly the condition under
    which the reference sets _xtml_json_block. That makes render -> parse -> render byte-stable
    through this path. Valid JSON is decoded instead, and re-renders as ordinary typed argument
    blocks; a valid non-object is dropped, because normalize_tool_arguments refuses to render one
    and there is nothing replayable to keep."""
    for attrs, raw, s, e in _blocks(body, "json"):
        if body[:s].strip() or body[e:].strip():
            break                            # not the whole body: a value, not the escape hatch
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {"arguments": {}, "json": raw}
        return {"arguments": parsed if isinstance(parsed, dict) else {}, "json": None}
    args = {}
    for attrs, val, _s, _e in _blocks(body, "argument"):
        a = _attrs(attrs)
        if "key" in a:
            args[a["key"]] = _coerce(val, a.get("type", "string"))
    return {"arguments": args, "json": None}


def to_openai_message(parsed, index_base=0):
    """parse_completion() -> (OpenAI `message`, finish_reason).

    reasoning_content is emitted even when empty. That is deliberate: replaying our own assistant
    messages must satisfy K3's preserved-thinking-history contract without the client knowing the
    contract exists. A raw json block is passed through as `arguments` verbatim rather than being
    re-serialized — OpenAI's arguments is a string, and forwarding the model's own bytes keeps
    render -> parse -> render stable through the reference's _xtml_json_block path."""
    msg = {"role": "assistant", "content": parsed["content"] or None,
           "reasoning_content": parsed["reasoning_content"] or ""}
    if parsed["tool_calls"]:
        msg["tool_calls"] = [
            {"id": f"call_{index_base + i}", "type": "function",
             "function": {"name": tc["name"],
                          "arguments": tc["json"] if tc.get("json") is not None
                          else json.dumps(tc["arguments"], ensure_ascii=False)}}
            for i, tc in enumerate(parsed["tool_calls"])
        ]
        return msg, "tool_calls"
    return msg, "stop"


# ---------------------------------------------------------------- streaming

class StreamSplitter:
    """Incremental (reasoning, content) split, the K3 counterpart of the gateway's _SSESplitter.

    Same contract: feed(delta) -> (reasoning_delta, content_delta) scanning only the new text, end()
    flushes at end-of-stream, and len(marker)-1 characters are held back so a marker straddling two
    commits is never half-emitted. K3 needs its own because it has one more channel than M2.5: the
    visible answer is WRAPPED in <|open|>response<|sep|>...<|close|>response<|sep|>, so a splitter
    built on M2.5's three markers would stream that opener to the user as content.

    Phases: 0 reasoning -> 1 between channels (swallowed) -> 2 content -> 3 done. Everything from
    the tool-call block onward is swallowed: tool calls surface through parse_completion at the end
    of the stream, never as content fragments.
    """
    _MARKERS = (THINK_BEGIN, THINK_END, RESPONSE_BEGIN, RESPONSE_END, TOOLCALL_BEGIN, MESSAGE_END,
                END_OF_MSG)
    _HOLD = max(len(m) for m in _MARKERS) - 1

    def __init__(self, reasoning_on=True):
        self.reasoning_on = reasoning_on
        self.phase = 0 if reasoning_on else 2   # thinking=False opens the response channel in the prompt
        self.buf = ""
        # A completion may repeat the <|open|>think<|sep|> the prompt already opened. It can only do
        # so at position 0, so the echo is resolved once, by holding the buffer until it is either
        # matched or ruled out. Stripping it with a replace() on each take instead made the split
        # depend on chunk size: with real (few-token) deltas the marker straddles takes and reaches
        # the client as text.
        self.echo = reasoning_on

    def _take(self, upto):
        s, self.buf = self.buf[:upto], self.buf[upto:]
        return s

    def _first(self, *markers):
        """(index, marker) of the earliest marker present, or (-1, None)."""
        best, hit = -1, None
        for m in markers:
            i = self.buf.find(m)
            if i >= 0 and (best < 0 or i < best):
                best, hit = i, m
        return best, hit

    def feed(self, delta):
        self.buf += delta
        r = c = ""
        while True:
            if self.echo:                            # resolve a leading opener before emitting
                if self.buf.startswith(THINK_BEGIN):
                    self.buf = self.buf[len(THINK_BEGIN):]
                    self.echo = False
                elif THINK_BEGIN.startswith(self.buf):
                    return r, c                      # still could become one: hold
                else:
                    self.echo = False
            if self.phase == 0:
                i = self.buf.find(THINK_END)
                if i >= 0:
                    r += self._take(i)
                    self.buf = self.buf[len(THINK_END):]
                    self.phase = 1
                    continue
                safe = len(self.buf) - self._HOLD
                if safe > 0:
                    r += self._take(safe)
                return r, c
            if self.phase == 1:
                # between the channels the model emits only structure; drop it, and jump straight to
                # the tool block if this turn is a pure tool call with no response channel at all
                i, hit = self._first(RESPONSE_BEGIN, TOOLCALL_BEGIN)
                if i >= 0:
                    self._take(i + len(hit))
                    self.phase = 2 if hit == RESPONSE_BEGIN else 3
                    continue
                safe = len(self.buf) - self._HOLD
                if safe > 0:
                    self._take(safe)
                return r, c
            if self.phase == 2:
                i, _ = self._first(RESPONSE_END, TOOLCALL_BEGIN, MESSAGE_END, END_OF_MSG)
                if i >= 0:
                    c += self._take(i)
                    self.buf = ""
                    self.phase = 3
                    return r, c
                safe = len(self.buf) - self._HOLD
                if safe > 0:
                    c += self._take(safe)
                return r, c
            return r, c                          # phase 3: structure + tool calls never stream

    def end(self):
        """End-of-stream: release the held-back tail (nothing more is coming). A tail that is the
        start of a marker is dropped, not flushed — reaching here in phase 0 or 2 means generation
        stopped mid-channel, so "<|open|" is a cut-off marker, never the answer's last characters."""
        r = c = ""
        if self.phase == 0:
            r = _trim_partial(self.buf.removeprefix(THINK_BEGIN) if self.echo else self.buf)
        elif self.phase == 2:
            c = _trim_partial(self.buf)
        self.buf = ""
        return r, c


def stream_splitter(reasoning_on=True):
    """Factory the gateway calls; m25 has no equivalent and keeps its own local class."""
    return StreamSplitter(reasoning_on)


def mock_completion(last, tools, reasoning=True):
    """A canned completion in K3's OWN shape, for the gateway's no-GPU mock (M25_GATEWAY_MOCK).
    Without this the mock emits MiniMax XML whatever model is selected, and the api-shape test would
    'pass' against a format the parser here has never seen."""
    head = f"The user asked: {last[:40]}.{THINK_END}" if reasoning else ""
    if tools:
        name = _escape_attr_value((tools[0].get("function") or tools[0])["name"])
        return (f"{head}{RESPONSE_BEGIN}Let me look that up.{RESPONSE_END}{TOOLCALL_BEGIN}"
                f'{OPEN}call tool="{name}" index="1"{SEP}'
                f'{OPEN}argument key="query" type="string"{SEP}{last[:30]}{CLOSE}argument{SEP}'
                f"{CLOSE}call{SEP}{TOOLCALL_END}{MESSAGE_END}{END_OF_MSG}")
    return (f"{head}{RESPONSE_BEGIN}Here is a concise answer to: {last[:60]}.{RESPONSE_END}"
            f"{MESSAGE_END}{END_OF_MSG}")


if __name__ == "__main__":
    demo = [{"role": "user", "content": "weather in Ghent?"},
            {"role": "assistant", "reasoning_content": "check the tool",
             "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "get_weather", "arguments": '{"city": "Ghent", "days": 2}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "18C, rain"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    print(render_text(demo, tools).replace(END_OF_MSG, END_OF_MSG + "\n"))
    print("\n--- parse ---")
    out = (f"reasoning here{THINK_END}{RESPONSE_BEGIN}It is 18C.{RESPONSE_END}"
           f"{TOOLCALL_BEGIN}{OPEN}call tool=\"get_weather\" index=\"1\"{SEP}"
           f"{OPEN}argument key=\"city\" type=\"string\"{SEP}Ghent{CLOSE}argument{SEP}"
           f"{CLOSE}call{SEP}{TOOLCALL_END}{MESSAGE_END}{END_OF_MSG}")
    print(json.dumps(parse_completion(out), indent=2, ensure_ascii=False))
