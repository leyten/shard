"""De-risk probe (one node): does the graphed fixed-topology tree forward
(FastVerify.tree_decode) reproduce the eager static tree forward bit-for-bit?

This isolates the CUDA-graph mechanics for the tree (shapes, the tree mask buffer,
the scratch-slot KV layout, capture/replay) on a real gpt-oss-120b stage, the same
way fastverify_graph.py de-risked the linear path -- so we don't debug graph capture
over the WAN. Tree LOGIC (ancestor mask + accept + gather) is validated end-to-end
on the swarm afterwards.
"""
import torch
from pipeline import load_stage
from fastverify import FastVerify

MODEL = "/root/models/gpt-oss-120b"
dev = "cuda"
parts = load_stage(MODEL, 0, 4, device=dev)          # head stage: embed + layers 0-8
embed = parts["embed"]
torch.manual_seed(0)

L = 24
prompt = torch.randint(0, 150000, (1, L), device=dev)

# fixed w=2,d=4 tree (build_tree shape): root + 2 chains of length 4 -> 9 nodes
par = [-1, 0, 1, 2, 3, 0, 5, 6, 7]
dep = [0, 1, 2, 3, 4, 1, 2, 3, 4]
M = len(par)
ttok = torch.randint(0, 150000, (1, M), device=dev)

with torch.no_grad():
    fv = FastVerify(parts, dev=dev)
    # eager static tree forward (no graph)
    fv.reset(); fv.prefill(embed(prompt), 0)
    fv._tbuild(M, par, dep); fv._tset(embed(ttok), L, par, dep)
    he = fv._tbody().clone()
    # graphed tree forward (capture + replay) on the same prefix + tree
    fv.reset(); fv.prefill(embed(prompt), 0)
    fv.tgraph = None
    hg = fv.tree_decode(embed(ttok), L, par, dep).clone()

d = (he.float() - hg.float()).abs()
print(f"tree forward graphed-vs-eager-static | shape {tuple(hg.shape)} | "
      f"max|diff|={d.max().item():.6f} mean|diff|={d.mean().item():.7f}")
print("PROBE_OK" if d.max().item() < 1e-2 else "PROBE_DIVERGE")
