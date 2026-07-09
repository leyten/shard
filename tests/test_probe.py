"""Capability probe (shard/probe.py) — the admission function + the network probe.

The role half is the ADMISSION_SPEC.md table as executable assertions (torch-free,
fabricated measured-cap dicts, like test_plan.py):
  * the marquee verdicts: 24 GB marlin NEVER anchors interactive but IS a batched
    filler; a 32 GB 5090 anchors on a tight pool and loses it on wide scatter;
    peak-gating (the admit-then-OOM fix) costs layers,
  * every gate can deny on its own: kernel, VRAM, RTT-hops, uplink, NAT, compute,
  * the LIVING-spec lever: overriding a v0 threshold flips the verdict,
  * `binding` names exactly the constraints that denied the next-higher role.

The network half runs for real over loopback (stdlib only): receiver-timed upload,
ping RTT, and the nonce dial-back. Plus the `python3 -m shard.probe` stdio seam.

Run: python3 -m pytest tests/test_probe.py -q
"""
import json
import os
import socket
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.probe import ADMISSION_MODEL_V0, derive_layers, derive_role, measure_net, serve  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cap(**kw):
    """A healthy tight-pool 32 GB Blackwell cutlass card; tests override per scenario."""
    base = {
        "total_vram_mb": 32768.0, "footprint_mb_per_layer": 1700.0,
        "load_peak_extra_mb": 4300.0, "has_fast_kernel": True, "layer_ms": 0.75,
        "can_recompute_block": True, "uplink_mbps": 500.0, "rtt_to_pool_ms": 19.0,
        "nat_dialable": True, "disk_free_gb": 200.0,
    }
    base.update(kw)
    return base


def _failed(verdict, role):
    for b in verdict["binding"]:
        if b["role"] == role:
            return b["failed"]
    return []


# ---- the spec table's marquee verdicts (numbers from ADMISSION_SPEC.md, v0) ----

def test_32gb_5090_tight_pool_is_interactive():
    v = derive_role(_cap())
    assert v["role"] == "interactive-anchor"
    assert v["layers"] == 12                     # the spec's load-tested cap (12, NOT plan.py's 13 —
    assert v["n_single"] == 6                    # admission is conservative until the probe settles it)
    assert v["predicted_tok_s"] > 20             # the tight-ring receipt regime


def test_wide_scatter_denies_what_tight_admits():
    # the receipt's ACTUAL scatter RTT (30 ms, 13-15 tok/s measured) — not a softer number.
    v = derive_role(_cap(rtt_to_pool_ms=30.0))
    assert v["role"] == "batched-filler"
    assert "hops_vs_rtt" in _failed(v, "interactive-anchor")
    assert v["predicted_tok_s"] < 20             # the model agrees with the 13-15 receipt's verdict


def test_24gb_marlin_never_anchors_but_fills_batched():
    cap = _cap(total_vram_mb=24576.0, footprint_mb_per_layer=4250.0,
               load_peak_extra_mb=900.0, layer_ms=1.6)
    v = derive_role(cap)
    assert v["layers"] == 5                      # the spec's marlin row
    assert v["n_single"] == 13                   # 13 hops — the physics that kills anchoring
    assert v["role"] == "batched-filler"
    assert "hops_vs_rtt" in _failed(v, "interactive-anchor")
    # even a perfect network doesn't save it: the hop count is VRAM-born
    v = derive_role({**cap, "rtt_to_pool_ms": 19.0, "uplink_mbps": 10000.0})
    assert v["role"] == "batched-filler"
    assert v["predicted_agg_tok_s"] > 20         # but aggregate clears the bar — ROUTED, not rejected


def test_48gb_blackwell_is_the_comfortable_anchor():
    v = derive_role(_cap(total_vram_mb=49152.0))
    assert v["role"] == "interactive-anchor"
    assert v["n_single"] == 4                    # density-scaled cap: 48 GB must NOT collapse to the
    assert v["n_single"] < derive_role(_cap())["n_single"]   # 32 GB verdict — the spec's core distinction


def test_peak_gating_costs_layers():
    lo = derive_layers(_cap(load_peak_extra_mb=0.0), ADMISSION_MODEL_V0)
    hi = derive_layers(_cap(load_peak_extra_mb=12000.0), ADMISSION_MODEL_V0)
    assert hi < lo                               # the swizzle peak is a real capacity cost
    assert derive_layers(_cap(footprint_mb_per_layer=None), ADMISSION_MODEL_V0) == 0
    assert derive_layers(_cap(total_vram_mb=0), ADMISSION_MODEL_V0) == 0


# ---- every gate denies on its own ----

def test_no_fast_kernel_relegates_to_verifier():
    v = derive_role(_cap(has_fast_kernel=False))
    assert v["role"] == "verifier"
    assert "fast_kernel" in _failed(v, "interactive-anchor")
    assert "fast_kernel" in _failed(v, "batched-filler")


def test_cpu_box_with_granted_recompute_is_verifier():
    v = derive_role({"can_recompute_block": True, "layer_ms": 4000.0, "uplink_mbps": 50.0,
                     "rtt_to_pool_ms": 40.0, "nat_dialable": False, "disk_free_gb": 100.0})
    assert v["role"] == "verifier"
    assert v["layers"] == 0 and v["n_single"] is None and v["predicted_tok_s"] is None


def test_undialable_node_cannot_hold_a_stage():
    v = derive_role(_cap(nat_dialable=False))
    assert v["role"] == "verifier"               # a stage must accept inbound; a verifier dials out
    assert "nat_dialable" in _failed(v, "interactive-anchor")


def test_residential_uplink_denies_interactive_only():
    v = derive_role(_cap(uplink_mbps=120.0))
    assert v["role"] == "batched-filler"         # decode is trivial; 16k prefill TTFT is the gate
    assert "uplink" in _failed(v, "interactive-anchor")


def test_throttled_compute_denies_via_c():
    v = derive_role(_cap(layer_ms=9.0))          # broken/throttled card: C balloons past the budget
    assert v["role"] != "interactive-anchor"
    assert "hops_vs_rtt" in _failed(v, "interactive-anchor")


def test_seeder_and_reject_tail():
    v = derive_role({"uplink_mbps": 50.0, "disk_free_gb": 100.0, "nat_dialable": False})
    assert v["role"] == "seeder"                 # no GPU, no recompute — still earns its keep
    v = derive_role({"uplink_mbps": 5.0, "disk_free_gb": 2.0})
    assert v["role"] == "reject"
    assert len(v["binding"]) == 4                # every role was considered and denied


def test_unreachable_pool_still_yields_a_role():
    # measure_net reports rtt_to_pool_ms=None when no pool peer answered (firewalled
    # candidate, pool down) — a common real case that must map to a role, not a crash.
    v = derive_role({"rtt_to_pool_ms": None, "uplink_mbps": None, "nat_dialable": False,
                     "can_recompute_block": True, "layer_ms": 500.0, "disk_free_gb": 50.0})
    assert v["role"] == "verifier"


# ---- the LIVING-spec lever: numbers are inputs, not law ----

def test_spec_override_flips_the_verdict():
    cap = _cap(uplink_mbps=120.0)
    assert derive_role(cap)["role"] == "batched-filler"
    v = derive_role(cap, spec={"uplink_interactive_mbps": 100.0})
    assert v["role"] == "interactive-anchor"     # measurement said 200 was too strict? revise it.
    # a lower bar admits the 24 GB marlin card as a (slow) anchor — the bar is a role tag
    marlin = _cap(total_vram_mb=24576.0, footprint_mb_per_layer=4250.0,
                  load_peak_extra_mb=900.0, layer_ms=1.6)
    assert derive_role(marlin, spec={"bar_interactive_tok_s": 8.0})["role"] == "interactive-anchor"


def test_model_override_reparameterizes():
    # a ~30B model (24 layers) fits a 24 GB marlin card in 5 layers -> N=5: the same card
    # that can NEVER anchor M2.5 anchors the smaller model — self-organizing multi-model.
    marlin = _cap(total_vram_mb=24576.0, footprint_mb_per_layer=4250.0,
                  load_peak_extra_mb=900.0, layer_ms=1.6)
    v = derive_role(marlin, model={"n_layers": 24})
    assert v["n_single"] == 5
    assert v["role"] == "interactive-anchor"


# ---- the network probe, for real, over loopback ----

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start_peer():
    port = _free_port()
    ready = threading.Event()
    threading.Thread(target=serve, args=(port,), kwargs={"ready_evt": ready}, daemon=True).start()
    assert ready.wait(5)
    return port


def test_net_probe_loopback_roundtrip():
    port = _start_peer()
    res = measure_net([f"127.0.0.1:{port}"], upload_mb=1, dialback_port=_free_port())
    assert res["rtt_to_pool_ms"] is not None and res["rtt_to_pool_ms"] < 50
    assert res["uplink_mbps"] > 1                # receiver-timed, loopback is fast
    assert res["nat_dialable"] is True           # the peer dialed back and the nonce echoed


def test_dialback_fails_closed_without_listener():
    port = _start_peer()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.settimeout(15)
    s.sendall((json.dumps({"op": "dialback", "port": _free_port(), "nonce": "x"}) + "\n").encode())
    rep = json.loads(s.makefile().readline())
    s.close()
    assert rep["ok"] is False                    # nothing listening == CGNAT == not dialable


def test_dead_pool_reports_unreachable():
    res = measure_net([f"127.0.0.1:{_free_port()}"], upload_mb=1)
    assert res["rtt_to_pool_ms"] is None and res["uplink_mbps"] == 0.0
    assert res["nat_dialable"] is False
    # ...and the role function accepts that cap verbatim (None RTT -> a role, not a crash)
    assert derive_role(res)["role"] == "reject"


def test_malformed_peer_is_skipped_not_fatal():
    port = _start_peer()
    res = measure_net(["no-colon-here", f"127.0.0.1:{port}"], upload_mb=1)
    assert res["rtt_ms"][0] is None              # bad peer skipped
    assert res["rtt_ms"][1] is not None          # good peer still measured


def test_upload_size_lie_is_not_counted():
    # a sender that claims N bytes but sends fewer gets ok=false from the receiver —
    # a short-send can never inflate (or register) an uplink number.
    port = _start_peer()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.settimeout(15)
    s.sendall((json.dumps({"op": "upload", "bytes": 4096}) + "\n").encode())
    s.sendall(b"x" * 100)
    s.shutdown(socket.SHUT_WR)
    rep = json.loads(s.makefile().readline())
    s.close()
    assert rep["ok"] is False


# ---- the stdio seam c0mpute drives ----

def test_cli_roundtrip():
    req = {"cap": _cap(), "spec": {"bar_interactive_tok_s": 20.0}}
    r = subprocess.run([sys.executable, "-m", "shard.probe"], input=json.dumps(req),
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    v = json.loads(r.stdout)
    assert v["role"] == "interactive-anchor" and v["spec"] == "v0"


def test_cli_bad_request_is_json_error():
    r = subprocess.run([sys.executable, "-m", "shard.probe"], input="not json",
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2
    assert "error" in json.loads(r.stdout)
