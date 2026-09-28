"""The CPU reference backend (torch on the host CPU): the canonical-numerics definition of each
adapted model (mimo_v2) and a StageRuntime over it (runtime.CpuRuntime). Every other backend
(CUDA, MLX) is tested against it; unlike shard/runtime/{api,toy}.py it needs torch.

Comparing another numerics class against it: exact-trajectory checks cannot pass, even for a
correct backend. MoE routing has near-ties, and one expert flipped by a last-bit difference moves
a row by tens of bf16 ulps and can change later sampled tokens. So compare
teacher-forced, one layer at a time on the reference's own inputs, within tolerance, and judge rows
whose reference routing gap or logit margin is below epsilon separately (design-runtime §3.5, §7 T5).
"""
