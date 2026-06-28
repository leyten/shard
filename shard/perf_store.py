"""perf_store — the 'dynamic, learns from real runs' half of throughput-aware scheduling.

throughput.best_ring needs three numbers it cannot know a priori:
  - ms_per_layer[node]      how fast each GPU pushes a round's tokens through one layer
  - rtt[a][b]               measured wire latency between two specific volunteer boxes
  - accept_rate[model]      speculative-decode accept fraction (draft quality vs target)

All three are PROPERTIES OF THE ACTUAL FLEET, not constants — so we learn them. After every
completed ring job the orchestrator posts a telemetry record (per-stage compute ms, observed
edge RTTs, tokens emitted / rounds run); we fold each into an EWMA keyed by node / edge /
model. Cold entries fall back to GPU-class priors (a 4090 is ~x ms/layer) so the very first
ring is a decent guess, and every subsequent ring is tuned to what the fleet really did.

EWMA (not a flat mean) so the estimate tracks drift — a node that thermal-throttles, a peering
path that degrades — without a unbounded history. alpha ~0.3: ~half-life of 2 samples, fast
enough to follow real change, damped enough to ignore one noisy job.

Stdlib only; JSON-file backed so it survives an orchestrator restart. Boundary law holds:
this stores layers/ms/rtt/accept — never $, accounts, or receipts.
"""
import json
import os
import threading


# GPU-class priors: ms to push one speculative round's token batch through ONE layer.
# Rough, measured-order-of-magnitude seeds; the EWMA corrects them after the first real run.
# Keyed by a normalized class string (see classify_gpu).
GPU_MS_PER_LAYER_PRIOR = {
    "h100":  0.18,
    "a100":  0.30,
    "4090":  0.42,
    "3090":  0.70,
    "4080":  0.55,
    "a6000": 0.50,
    "l40s":  0.40,
    "unknown": 0.80,        # conservative: assume slow until proven fast
}

DEFAULT_ACCEPT_RATE = 0.62       # typical greedy spec-decode accept for a tuned draft
DEFAULT_K = 4
DEFAULT_RTT_MS = 80.0            # cold edge: assume a mediocre cross-region link
EWMA_ALPHA = 0.3


def classify_gpu(name: str) -> str:
    """map a raw nvidia-smi name onto a prior bucket. best-effort substring match."""
    s = (name or "").lower()
    for key in ("h100", "a100", "4090", "4080", "3090", "a6000", "l40s"):
        if key in s:
            return key
    return "unknown"


def _ewma(old, new, alpha=EWMA_ALPHA):
    return new if old is None else (alpha * new + (1 - alpha) * old)


class PerfStore:
    """thread-safe, JSON-backed EWMA store of fleet performance.

    keys:
      ms_per_layer : node_id -> ms/layer (seeded from GPU class, refined per run)
      rtt          : "a|b" -> ms (directional; a->b)
      accept       : model -> accept_rate
    """

    def __init__(self, path=None):
        self.path = path
        self._lock = threading.Lock()
        self.ms_per_layer = {}
        self.gpu_class = {}             # node_id -> class string (for cold prior)
        self.rtt = {}
        self.accept = {}
        if path and os.path.exists(path):
            self._load()

    # ---- reads (return a usable number even when cold) ----
    def ms_for(self, node_id):
        with self._lock:
            v = self.ms_per_layer.get(node_id)
            if v is not None:
                return v
            cls = self.gpu_class.get(node_id, "unknown")
            return GPU_MS_PER_LAYER_PRIOR.get(cls, GPU_MS_PER_LAYER_PRIOR["unknown"])

    def rtt_for(self, a, b):
        if a == b:
            return 0.0
        with self._lock:
            return self.rtt.get(f"{a}|{b}", DEFAULT_RTT_MS)

    def accept_for(self, model):
        with self._lock:
            return self.accept.get(model, DEFAULT_ACCEPT_RATE)

    def ms_map(self, node_ids):
        """ms_per_layer dict for a set of nodes, priors filled in. For throughput.best_ring."""
        return {nid: self.ms_for(nid) for nid in node_ids}

    def rtt_mesh(self, node_ids):
        """full directional mesh for a node set, cold edges = DEFAULT_RTT_MS."""
        return {a: {b: self.rtt_for(a, b) for b in node_ids if b != a} for a in node_ids}

    # ---- writes ----
    def seed_gpu(self, node_id, gpu_name):
        """record a node's GPU class so its cold ms/layer prior is sensible."""
        with self._lock:
            self.gpu_class[node_id] = classify_gpu(gpu_name)

    def observe_run(self, record: dict):
        """fold one completed ring job into the EWMAs.

        record = {
          "model": "GLM-5.2",
          "stages": [{"node_id": "A", "n_layers": 30, "compute_ms": 11.2}, ...],
          "edges":  [{"from": "A", "to": "B", "rtt_ms": 34.1}, ...],     # optional
          "tokens": 64, "rounds": 20,                                    # for accept_rate
          "K": 4                                                          # the K used
        }
        compute_ms is the stage's per-round compute; we divide by n_layers to get ms/layer.
        """
        with self._lock:
            for s in record.get("stages", []):
                nid = s["node_id"]
                nl = max(int(s.get("n_layers", 0)), 1)
                if "compute_ms" in s and s["compute_ms"] is not None:
                    per_layer = float(s["compute_ms"]) / nl
                    self.ms_per_layer[nid] = _ewma(self.ms_per_layer.get(nid), per_layer)
            for e in record.get("edges", []):
                k = f"{e['from']}|{e['to']}"
                self.rtt[k] = _ewma(self.rtt.get(k), float(e["rtt_ms"]))
            rounds = record.get("rounds")
            toks = record.get("tokens")
            K = record.get("K", DEFAULT_K)
            if rounds and toks and K:
                # accepted tokens per round / K = accept fraction (clamped to [0,1])
                acc = min(max((toks / rounds) / K, 0.0), 1.0)
                m = record.get("model", "")
                self.accept[m] = _ewma(self.accept.get(m), acc)
            if self.path:
                self._save_locked()

    # ---- persistence ----
    def _snapshot(self):
        return {"ms_per_layer": self.ms_per_layer, "gpu_class": self.gpu_class,
                "rtt": self.rtt, "accept": self.accept}

    def _save_locked(self):
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self._snapshot(), f)
        os.replace(tmp, self.path)        # atomic — never a half-written store

    def _load(self):
        with open(self.path) as f:
            d = json.load(f)
        self.ms_per_layer = d.get("ms_per_layer", {})
        self.gpu_class = d.get("gpu_class", {})
        self.rtt = d.get("rtt", {})
        self.accept = d.get("accept", {})
