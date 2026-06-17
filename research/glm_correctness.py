"""GLM-5.2-NVFP4 full-model correctness + tok/s on 16x RTX 5090 (Blackwell), via vLLM.
confirms the whole model serves coherent tokens at NVFP4 and measures decode throughput,
with and without the native MTP speculative-decoding draft.
run under /root/vllm: python glm_correctness.py
"""
import time, torch
from vllm import LLM, SamplingParams

MODEL = "/root/glm52nvfp4"
COMMON = dict(model=MODEL, quantization="modelopt", kv_cache_dtype="fp8",
              tensor_parallel_size=16, trust_remote_code=True,
              max_model_len=4096, gpu_memory_utilization=0.90, enforce_eager=False)

def run(llm, tag):
    print(f"\n===== {tag} =====", flush=True)
    # correctness: coherent output, greedy
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    prompts = ["Explain what decentralized computing is, in two sentences.",
               "Write a short poem about the ocean at midnight."]
    t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    for p, o in zip(prompts, outs):
        print(f"\n>>> {p}\n{o.outputs[0].text.strip()}", flush=True)
    print(f"\n[batch] {ntok} tok in {dt:.1f}s = {ntok/dt:.1f} tok/s aggregate", flush=True)
    # single-stream decode tok/s (the headline number)
    sp1 = SamplingParams(temperature=0.0, max_tokens=256)
    _ = llm.generate(["Hi."], SamplingParams(temperature=0, max_tokens=8))  # warm
    t0 = time.time(); o = llm.generate(["Explain how speculative decoding speeds up LLM inference."], sp1)
    dt = time.time() - t0; n = len(o[0].outputs[0].token_ids)
    print(f"[single-stream] {n} tok in {dt:.1f}s = {n/dt:.1f} tok/s", flush=True)
    return n / dt


print("loading GLM-5.2-NVFP4 across 16x 5090 (this takes a few min)...", flush=True)
base = LLM(**COMMON)
base_tps = run(base, "NVFP4 base (no draft)")
del base
import gc; gc.collect(); torch.cuda.empty_cache()

# native MTP speculative decoding
try:
    print("\nloading with MTP spec-decode...", flush=True)
    spec = LLM(**COMMON, speculative_config={"method": "mtp", "num_speculative_tokens": 5})
    spec_tps = run(spec, "NVFP4 + MTP spec-decode (k=5)")
    print(f"\nMTP speedup: {spec_tps/base_tps:.2f}x ({base_tps:.1f} -> {spec_tps:.1f} tok/s)", flush=True)
except Exception as e:
    print(f"\nMTP path failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("(base NVFP4 number stands; MTP needs the draft layer in the checkpoint + vLLM support)", flush=True)
