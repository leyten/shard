"""The CPU reference backend (torch on the host CPU): the canonical-numerics definition of each
adapted model (mimo_v2) and a StageRuntime over it (runtime.CpuRuntime). Every other backend
(CUDA, MLX) is tested against it; unlike shard/runtime/{api,toy}.py it needs torch."""
