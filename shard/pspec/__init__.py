"""shard v2 pipelined speculation, model-agnostic (ARCHITECTURE-v2 §4.3, BUILD-PLAN S1).

accept  the accept rule: one rule for every frame length, the entry half (`accept`) and the tail
        half (`Frontier`), lifted from the V4 engine with `plan_verify_round` and `wants_block`.
coord   the single-session coordinator on shard-frame/2: `Session` (the per-session state machine,
        no I/O), `FrameSender` (the sender-side sep fence), `run` (a one-session driver) and the
        `Link` seam it talks through (`SocketLink` for the real node server and the fake mesh).
"""
