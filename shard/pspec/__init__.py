"""shard v2 pipelined speculation, model-agnostic (ARCHITECTURE-v2 §4.3, BUILD-PLAN S1).

accept  the accept rule: one rule for every frame length, the entry half (`accept`) and the tail
        half (`Frontier`), lifted from the V4 engine with `plan_verify_round` and `wants_block`.
"""
