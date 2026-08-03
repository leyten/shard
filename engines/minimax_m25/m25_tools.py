"""MiniMax-M2.5 tool-calling + chat rendering (engine-side, pure — no GPU/transport deps).

Ground truth is the model's own chat_template.jinja (verified against nvidia/MiniMax-M2.5-NVFP4):

  * INPUT: apply_chat_template(messages, tools=[...]) injects a tools block —
        <tools>\n<tool>{function_json}</tool>\n...\n</tools>
    and instructs the model to answer tool calls in XML.

  * OUTPUT: the model emits interleaved thinking then an Anthropic-style XML call:
        <think>\n ...reasoning... \n</think>\n\n visible-content
        <minimax:tool_call>
        <invoke name="NAME">
        <parameter name="KEY">VALUE</parameter>
        </invoke>
        </minimax:tool_call>
    The generation prompt opens "<think>\n", so the decoded completion starts INSIDE the think
    block and the visible content/tool-call follow "</think>". A param VALUE is JSON-encoded unless
    it is already a plain string (template: `v | tojson if v is not string else v`) — so on parse we
    try json.loads(VALUE) and fall back to the raw string.

Verified locally (no GPU): the markers <minimax:tool_call>/<think> are added-vocab but NOT special,
so they survive coordinate_pipe's `tok.decode(out, skip_special_tokens=True)` — the parser runs on
that decoded text directly.

The coordinator calls render_ids() to build the prompt and parse_completion() on the decoded output;
the OpenAI gateway maps the result onto the chat-completions schema (arguments -> JSON string).
"""
import json
import re

TOOLCALL_BEGIN = "<minimax:tool_call>"
TOOLCALL_END = "</minimax:tool_call>"
THINK_END = "</think>"
THINK_BEGIN = "<think>"

_INVOKE_RE = re.compile(r'<invoke\s+name="(?P<name>[^"]*)"\s*>(?P<body>.*?)</invoke>', re.DOTALL)
_PARAM_RE = re.compile(r'<parameter\s+name="(?P<key>[^"]*)"\s*>(?P<val>.*?)</parameter>', re.DOTALL)


def _normalize_history(messages):
    """The M2.5 template renders assistant tool_calls by iterating `arguments.items()`, so it needs
    arguments as a DICT — but the OpenAI wire format carries it as a JSON string. Convert string
    arguments back to dicts (without mutating the caller's messages) so OpenAI-format multi-turn
    history templates cleanly."""
    out = []
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m); continue
        m = dict(m); new_tcs = []
        for tc in tcs:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except (ValueError, TypeError):
                    args = {}
                fn = dict(fn, arguments=args)
                tc = dict(tc, function=fn) if "function" in tc else fn
            new_tcs.append(tc)
        m["tool_calls"] = new_tcs
        out.append(m)
    return out


_NOTHINK_IDS = {}                                 # tokenizer id() -> the "</think>\n\n" token ids (cached per tokenizer)


def render_ids(tok, messages, tools=None, add_generation_prompt=True, reasoning=True):
    """Build prompt token ids from OpenAI-style messages [{role, content, ...}] and optional tools
    [{"type":"function","function":{...}}]. Returns a flat list[int] (no torch needed). The M2.5
    template emits a BatchEncoding under return_dict=True, so we unwrap input_ids and flatten.

    reasoning=False: the M2.5 chat template hardwires "<think>\n" at the generation prompt, so every
    request normally reasons first (novel tokens -> 0% n-gram accept -> WAN-floor slow). For
    latency-sensitive / high-overlap normal usage, append "</think>\n\n" to CLOSE the think block so
    the model answers directly (the answer of a summarize/code/RAG task reuses the context -> the
    n-gram drafter accepts it -> fast). This is the real-usage analogue of the benchmarks' think-skip,
    now a first-class flag instead of a copy-task-only hack. Default True (reason, max quality)."""
    enc = tok.apply_chat_template(_normalize_history(messages), tools=tools or None,
                                  add_generation_prompt=add_generation_prompt, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):  # [1, T] batch -> [T]
        ids = ids[0]
    ids = list(ids)
    if add_generation_prompt and not reasoning:
        key = id(tok)
        if key not in _NOTHINK_IDS:
            _NOTHINK_IDS[key] = list(tok("</think>\n\n", add_special_tokens=False)["input_ids"])
        ids = ids + _NOTHINK_IDS[key]              # close the forced <think> -> direct answer (no reasoning)
    return ids


def _coerce(raw):
    """Invert the template's value encoding: JSON for non-strings, raw for strings. Try json.loads;
    on failure the value was a bare string, so return it stripped of formatting whitespace."""
    v = raw.strip()
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def parse_completion(text):
    """Parse a decoded M2.5 completion into {reasoning_content, content, tool_calls}.

    tool_calls is a list of {"name": str, "arguments": dict}. Robust to: a leading "<think>" (forced
    by the gen prompt) or none; a missing "</think>"; multiple <invoke> blocks; and a truncated
    (missing) </minimax:tool_call> end token — invokes are matched by their own </invoke> bounds."""
    reasoning = None
    body = text
    if THINK_END in body:
        head, _, body = body.partition(THINK_END)
        reasoning = head.split(THINK_BEGIN)[-1].strip("\n").strip() or None

    tool_calls = []
    if TOOLCALL_BEGIN in body:
        idx = body.index(TOOLCALL_BEGIN)
        content = body[:idx].strip()
        for m in _INVOKE_RE.finditer(body[idx:]):
            args = {p.group("key"): _coerce(p.group("val")) for p in _PARAM_RE.finditer(m.group("body"))}
            tool_calls.append({"name": m.group("name"), "arguments": args})
    else:
        content = body.strip()

    return {"reasoning_content": reasoning, "content": content, "tool_calls": tool_calls}


def to_openai_message(parsed, index_base=0):
    """Map parse_completion() output onto an OpenAI chat-completion `message` dict + finish_reason.
    arguments are serialized to a JSON string per the OpenAI tool-call schema."""
    msg = {"role": "assistant", "content": parsed["content"] or None}
    if parsed["reasoning_content"]:
        msg["reasoning_content"] = parsed["reasoning_content"]
    if parsed["tool_calls"]:
        msg["tool_calls"] = [
            {"id": f"call_{index_base + i}", "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
            for i, tc in enumerate(parsed["tool_calls"])
        ]
        return msg, "tool_calls"
    return msg, "stop"
