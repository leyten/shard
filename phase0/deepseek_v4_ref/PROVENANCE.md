# Vendored reference: DeepSeek V4 Flash

**Source:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (snapshot 2026-07-31)
**License:** MIT (permits redistribution with attribution)

`inference/` and `encoding/` are copied **verbatim** from that repo, along with the
top-level `config.json`, `generation_config.json`, and the tokenizer files. Only the
`*.safetensors` weight shards are omitted (they are pulled per box at run time).

## Why this is vendored, not reimplemented

The shard V4 engine **rents** this math; it does not rewrite it. This mirrors the K3
precedent in `phase0/kimi_k3_ref/` and the runtime policy in `docs/MODEL_RUNTIME.md`.
`phase0/v4_stage.py` instantiates the reference `Block`s for a contiguous layer range
and drives them across the pipeline. It never reimplements:

- the hybrid attention (`Attention`: MLA latent + sliding window + the `Indexer`/`Compressor`
  top-512 compressed-sparse path),
- the manifold-constrained hyper-connections (`Block.hc_pre`/`hc_post`, `hc_split_sinkhorn`),
- the FP4/FP8 MoE and kernels (`kernel.py`),
- the DSpark drafter (`DSparkBlock`, `forward_spec`, the Markov + confidence heads).

Keeping these byte-identical means version skew and any upstream fixes are absorbed in
`v4_stage.py`, in one auditable place, rather than by editing code we do not own.

## The one contract that matters for the pipeline

`Transformer.forward` expands the hidden state to `hc_mult=4` streams
(`h.unsqueeze(2).repeat(1,1,4,1)`) that **persist across every layer** and collapse to one
only at `hc_head` before the final norm/head. So the **inter-stage payload is `h: [b, s, 4, dim]`**,
~4x a plain hidden state. The DSpark drafter reads `main_hidden` = the mean-over-streams of the
hidden at layers `dspark_target_layer_ids = [40, 41, 42]` (all on the tail stage), so drafting
is local to the tail box.
