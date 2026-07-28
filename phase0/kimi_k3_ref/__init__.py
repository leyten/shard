"""Kimi-K3's OWN reference modules, vendored verbatim — the engine RENTS this math, never rewrites it.

Provenance (re-sync by re-downloading; the files below are byte-identical to these URLs):

  repo      huggingface.co/moonshotai/Kimi-K3
  revision  9f62e4e9fffbd0a83ddd60e1c209d828994b3569  (weights live 2026-07-27T16:29:18Z)
  files     https://huggingface.co/moonshotai/Kimi-K3/raw/main/modeling_kimi_linear.py
            sha256 9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a
            https://huggingface.co/moonshotai/Kimi-K3/raw/main/configuration_kimi_k3.py
            sha256 735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae

These ship in the checkpoint as `trust_remote_code` modules, so the alternative to vendoring them is
downloading 1.4 TiB of weights (or hitting the network) before a CPU test can construct one decoder
layer. They carry their own headers: the MLA / MoE-gate / sparse-MoE parts are Apache-2.0 (adapted
from DeepSeek-V3), the rest is the Kimi K3 License. Neither is modified here — this file is the only
addition, and it exists so `from .configuration_kimi_k3 import KimiLinearConfig` resolves as a
package. `modeling_kimi_linear` additionally hard-imports `fla` (Triton, GPU-only) and `einops`; see
phase0/k3_kda_cpu.py for the CPU backend that satisfies the first on a box with no GPU.

The vision tower (`modeling_kimi_k3.py`) is deliberately NOT vendored: a pipeline stage runs the text
decoder, and the tower is a head-side concern we have not scoped.
"""
