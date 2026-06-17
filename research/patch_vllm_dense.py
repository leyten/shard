"""Patch vLLM 0.23 deepseek_v2.py to run GLM-5.2 (glm_moe_dsa) DENSE on consumer sm_120
(RTX 5090), bypassing the missing FLASHMLA_SPARSE kernel. Two edits:
  1. is_v32 = False  -> dense MLA (TRITON_MLA works on sm_120); valid because at decode
     lengths << index_topk (2048) the sparse indexer selects everything anyway.
  2. skip "indexer" weights in load_weights -> the dense model has no indexer module, so
     the checkpoint's self_attn.indexer.* weights have no home (else KeyError).
"""
f = "/root/vllm/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py"
src = open(f).read()
n_v32 = src.count('self.is_v32 = hasattr(config, "index_topk")')
src = src.replace('self.is_v32 = hasattr(config, "index_topk")', "self.is_v32 = False")
out, n_skip = [], 0
for ln in src.split("\n"):
    out.append(ln)
    if ln.strip().startswith("for name, loaded_weight in weights:"):
        ind = len(ln) - len(ln.lstrip())
        out.append(" " * (ind + 4) + 'if "indexer" in name: continue  # dense: no indexer module')
        n_skip += 1
open(f, "w").write("\n".join(out))
print(f"patched: is_v32 sites={n_v32} -> False, indexer-skip inserted in {n_skip} load loops")
