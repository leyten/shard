"""Kimi-K3's OWN reference modules, vendored verbatim — the engine RENTS this math, never rewrites it.

Provenance (re-sync by re-downloading; the files below are byte-identical to these URLs):

  repo      huggingface.co/moonshotai/Kimi-K3
  revision  9f62e4e9fffbd0a83ddd60e1c209d828994b3569  (weights live 2026-07-27T16:29:18Z)
  files     https://huggingface.co/moonshotai/Kimi-K3/raw/main/modeling_kimi_linear.py
            sha256 9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a
            https://huggingface.co/moonshotai/Kimi-K3/raw/main/configuration_kimi_k3.py
            sha256 735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae
            https://huggingface.co/moonshotai/Kimi-K3/raw/main/encoding_k3.py
            sha256 b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333
            https://huggingface.co/moonshotai/Kimi-K3/raw/main/tokenization_kimi.py
            sha256 f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944

These ship in the checkpoint as `trust_remote_code` modules, so the alternative to vendoring them is
downloading 1.4 TiB of weights (or hitting the network) before a CPU test can construct one decoder
layer. They carry their own headers: the MLA / MoE-gate / sparse-MoE parts are Apache-2.0 (adapted
from DeepSeek-V3), the rest is the Kimi K3 License. None is modified here — this file is the only
addition, and it exists so `from .configuration_kimi_k3 import KimiLinearConfig` resolves as a
package. `modeling_kimi_linear` additionally hard-imports `fla` (Triton, GPU-only) and `einops`; see
phase0/k3_kda_cpu.py for the CPU backend that satisfies the first on a box with no GPU.

`encoding_k3` is the chat renderer. K3 ships NO chat_template.jinja — rendering is this Python file,
and it is the ONLY definition of the XTML wire format that exists, so shard wraps it (phase0/
k3_tools.py) rather than porting it. It is pure stdlib, which is why the round-trip tests need
neither a tokenizer nor a network. `tokenization_kimi` is the TikTokenTokenizer that consumes it;
it is vendored for the encode contract it defines (see below) and hard-imports tiktoken, tokenizers
and transformers, so nothing on the test path imports it.

THE VOCAB FILE IS NOT VENDORED. `tiktoken.model` is a 2,795,286-byte binary BPE table:

  https://huggingface.co/moonshotai/Kimi-K3/resolve/main/tiktoken.model
  sha256 b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103

A box serving K3 already downloads it with the rest of the checkpoint, so carrying a copy in git
would buy nothing but permanent repo weight. Tests build a small synthetic tiktoken vocab instead
(tests/test_k3_tools.py `_toy_tokenizer`), which is enough because the thing worth testing about
encoding is not the BPE merges but the SEGMENT boundary: `TikTokenTokenizer._encode_text_piece`
encodes structural markers with allowed_special="all" and user/tool text with disallowed_special=(),
so a prompt containing the literal characters "<|open|>" tokenizes as ordinary text and can never
forge a control token. That is a security property, and it is testable with 20 merges.

The vision tower (`modeling_kimi_k3.py`) and the processors are deliberately NOT vendored: a pipeline
stage runs the text decoder, and the tower is a head-side concern we have not scoped. `encoding_k3`
renders image placeholders regardless — that path is reachable but untested here.
"""
