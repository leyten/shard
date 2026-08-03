"""Selective HF puller (stdlib only, no deps) — grabs the small config files + only the
safetensors shard(s) that hold a given layer's tensors, for the single-GPU S1 probe.

  python3 m25_pull.py --repo nvidia/MiniMax-M2.5-NVFP4 --dir /root/m25 --layer 30
"""
import os, json, argparse, urllib.request

TOKEN = open("/root/.hf_token").read().strip()


def dl(repo, fname, out):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    return os.path.getsize(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/MiniMax-M2.5-NVFP4")
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()

    for f in ["config.json", "hf_quant_config.json", "generation_config.json", "model.safetensors.index.json"]:
        try:
            print(f"{f}: {dl(a.repo, f, f'{a.dir}/{f}')} bytes", flush=True)
        except Exception as e:
            print(f"{f}: ERR {e}", flush=True)

    idx = json.load(open(f"{a.dir}/model.safetensors.index.json"))["weight_map"]
    shards = sorted({s for k, s in idx.items() if k.startswith(f"model.layers.{a.layer}.")})
    print(f"shards holding layer {a.layer}: {shards}", flush=True)
    for s in shards:
        print(f"downloading {s} ...", flush=True)
        try:
            print(f"  {s}: {dl(a.repo, s, f'{a.dir}/{s}')/1e9:.2f} GB", flush=True)
        except Exception as e:
            print(f"  {s}: ERR {e}", flush=True)
    print("PULL DONE", flush=True)
