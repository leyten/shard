"""Multi-GPU-per-box K3 rings: pin each stage to a local GPU, chain them over LOOPBACK inside a box,
and cross the WAN sidecar only at box boundaries — so a ring of B boxes x G GPUs is B*G stages but
only B WAN hops.

These are the PURE plumbing proofs (no GPU, no network): the per-stage device pin (a) is a string
the launcher emits, the planner GPU-split (b) and the launch-spec topology (c) are pure tiling/wiring
builders, and (d) locks the single-GPU / M2.5-shape path byte-identical. Real multi-GPU execution
(loopback handoff + the CUDA device pin actually taking) validates later on a rented 4x5090 box.

Run: python3 -m pytest tests/test_k3_multigpu.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

pytest.importorskip("torch")                               # k3_pipe imports torch + k3_stage at module scope
pytest.importorskip("safetensors.torch")
KP = pytest.importorskip("k3_pipe")


def _node_stages(ranges, ids=None):
    """A head-first NODE plan (plan_ring's shape) from contiguous (lo, hi) box blocks."""
    ids = ids or [f"box{k}" for k in range(len(ranges))]
    n = len(ranges)
    return [{"id": ids[k], "index": k, "lo": lo, "hi": hi, "head": k == 0, "tail": k == n - 1,
             "layers": hi - lo} for k, (lo, hi) in enumerate(ranges)]


# ── (a) device-routing: the launch command pins the intended local GPU ──────────────────────────────

def test_stage_cmd_pins_cuda_visible_devices_and_a_distinct_moe_port():
    """gpu=<idx> => CUDA_VISIBLE_DEVICES=<idx> (the process sees its card as cuda:0, so the engine's
    K3_DEV=cuda + set_device(0) target it with no code change) and a DISTINCT K3_MOE_PORT so
    co-located vLLM MoE groups don't collide on MASTER_PORT."""
    cmd = KP.stage_launch_cmd(5, 8, 24, 30, gpu=2, port=KP.local_eng_port(2),
                              nxt_addr="127.0.0.1:29623")
    assert "CUDA_VISIBLE_DEVICES=2 " in cmd
    assert f"K3_MOE_PORT={KP.K3_MOE_PORT_BASE + 2} " in cmd                 # 29559
    assert f"--port {KP.local_eng_port(2)} " in cmd                        # 29622, a loopback-only port
    assert "--next 127.0.0.1:29623" in cmd                                 # loopback to the next GPU
    assert "--stage 5 --nstages 8 --lo 24 --hi 30" in cmd                  # GLOBAL ring indices


def test_each_local_gpu_gets_its_own_visible_device_and_moe_port():
    cmds = [KP.stage_launch_cmd(j, 4, j, j + 1, gpu=j, port=KP.local_eng_port(j)) for j in range(4)]
    assert [c.count("CUDA_VISIBLE_DEVICES=") for c in cmds] == [1, 1, 1, 1]
    devs = sorted(int(c.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0]) for c in cmds)
    ports = sorted(int(c.split("K3_MOE_PORT=")[1].split(" ")[0]) for c in cmds)
    assert devs == [0, 1, 2, 3]                                            # one card each
    assert len(set(ports)) == 4                                            # distinct MASTER_PORTs


def test_single_gpu_stage_cmd_is_the_knobs_off_form_detached_via_setsid():
    """The device knob is additive: with gpu/port/nxt_addr unset the command is the single-stage-
    per-box form (no CUDA_VISIBLE_DEVICES, binds ENG_IN, forwards to the local sidecar FWD_RING leg).
    In every shape the command is fully DETACHED via setsid + fd redirect (Bug 1) so a parallel
    over-ssh launch returns the session instantly instead of hanging on the engine's child fds."""
    got = KP.stage_launch_cmd(1, 3, 31, 62, model_dir="/root/k3")
    inner = (f"python3 /root/k3_pipe.py stage --stage 1 --nstages 3 --lo 31 --hi 62 "
             f"--port {KP.ENG_IN} --next 127.0.0.1:{KP.FWD_RING} --dir /root/k3 "
             f"> /root/k3_stage_{KP.ENG_IN}.log 2>&1")
    expect = (f"K3_DIR=/root/k3 K3_DEV=cuda M25_ENGINE_BIND=127.0.0.1 "
              f"setsid bash -c '{inner}' </dev/null >/dev/null 2>&1 &")
    assert got == expect
    assert "CUDA_VISIBLE_DEVICES" not in got
    tail = KP.stage_launch_cmd(2, 3, 62, 93)                               # the tail: no --next
    assert "--next" not in tail and f"--port {KP.ENG_IN}" in tail


def test_fourteen_stage_launch_spec_fully_detaches_every_stage():
    """The launch race that aborted the live 14-stage ring (2026-07-29) had two halves: a stage gave
    up dialing its neighbour too early (fixed by the _dial window, see test_k3_pipe), AND `nohup
    <cmd> &` over ssh HUNG on the engine's child fds so the parallel launch never returned. So every
    stage command in a real B*G launch spec must be fully detached (setsid + fd redirect + &), and
    co-located stages on one box must log to DISTINCT files so their output never collides."""
    nodes = _node_stages(KP.even_tiling(93, 2))            # 2 boxes ...
    stages = KP.box_ring_launch(nodes, 7)["stages"]        # ... x 7 GPUs = 14 stages
    assert len(stages) == 14
    for s in stages:
        cmd = s["cmd"]
        assert "setsid bash -c " in cmd                    # its own detached process group
        assert cmd.rstrip().endswith("</dev/null >/dev/null 2>&1 &")   # fds closed, backgrounded
        assert f"/root/k3_stage_{s['eng_port']}.log" in cmd            # per-port log
    for k in range(2):                                     # per box: every stage logs to a distinct file
        logs = [f"/root/k3_stage_{s['eng_port']}.log" for s in stages if s["box_index"] == k]
        assert len(set(logs)) == len(logs)


# ── (b) planner: split node blocks to (node, gpu) with contiguity + B WAN hops ───────────────────────

def test_mixed_fleet_tiles_93_layers_to_node_gpu_blocks_contiguously():
    """Two 4-GPU nodes + one 1-GPU node. plan_ring places layers on NODES; split_stages_to_gpus tiles
    each node's block across its GPUs. The (node, gpu) blocks tile [0, 93) with no gap/overlap, each
    box's GPUs are 0..G-1, and there are still only B=3 WAN hops."""
    nodes = _node_stages(KP.even_tiling(93, 3))            # 3 boxes, ~31 layers each
    subs = KP.split_stages_to_gpus(nodes, [4, 4, 1])

    assert len(subs) == 4 + 4 + 1                          # sum of the per-box GPU counts
    # global head/tail are the ring ends, exactly one of each
    assert subs[0]["head"] and subs[0]["lo"] == 0
    assert subs[-1]["tail"] and subs[-1]["hi"] == 93
    assert sum(s["head"] for s in subs) == 1 and sum(s["tail"] for s in subs) == 1
    # contiguous tiling across the whole flattened ring
    cursor = 0
    for g, s in enumerate(subs):
        assert s["global_index"] == g and s["nstages"] == len(subs)
        assert s["lo"] == cursor, "the (node, gpu) blocks must tile with no gap/overlap"
        cursor = s["hi"]
    assert cursor == 93
    # per box: local GPU indices 0..G-1, one box_head + one box_tail, blocks stay within the node
    for k, G in enumerate((4, 4, 1)):
        box = [s for s in subs if s["box_index"] == k]
        assert [s["gpu"] for s in box] == list(range(G))
        assert sum(s["box_head"] for s in box) == 1 and sum(s["box_tail"] for s in box) == 1
        assert box[0]["lo"] == nodes[k]["lo"] and box[-1]["hi"] == nodes[k]["hi"]

    # B WAN hops: the sidecar sees only the 3 boxes (intra-box handoffs are loopback) — B-1 ring
    # forwards + 1 head->tail return = B.
    maddrs = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(3)]
    wan = sum(len(KP.ring_sidecar_spec(b, 3, maddrs)[1]) for b in range(3))
    assert wan == 3


def test_single_gpu_split_returns_the_node_blocks_unchanged():
    """gpus=1 everywhere => one sub-block per node, identical layer ranges — a single-GPU ring is
    byte-identical to the node plan."""
    nodes = _node_stages(KP.even_tiling(93, 5))
    subs = KP.split_stages_to_gpus(nodes, 1)
    assert [(s["lo"], s["hi"]) for s in subs] == [(nd["lo"], nd["hi"]) for nd in nodes]
    assert all(s["box_head"] and s["box_tail"] and s["nlocal"] == 1 for s in subs)


def test_plan_layer_ranges_per_gpu_consumes_plan_ring_then_splits():
    """The consumption path end to end: a feasible homogeneous multi-GPU pool -> plan_ring node plan
    -> per-GPU split. per_gpu=False (default) stays the node plan (unchanged), per_gpu=True expands
    it to (node, gpu) stages that still tile [0, 93)."""
    n = 22
    nodes = [{"id": f"box{i}", "free_vram_mb": 96000.0, "subnet": f"10.{i}.0.0/24",
              "total_vram_mb": 98304.0, "gpus": 2} for i in range(n)]
    rtt = [[0.0 if i == j else 12.0 for j in range(n)] for i in range(n)]

    node_plan = KP.plan_layer_ranges(nodes, rtt)                           # default: node granularity
    assert node_plan is not None and node_plan[0]["lo"] == 0 and node_plan[-1]["hi"] == 93

    gpu_plan = KP.plan_layer_ranges(nodes, rtt, per_gpu=True)
    assert gpu_plan is not None
    assert len(gpu_plan) == 2 * len(node_plan)                            # 2 GPUs per placed box
    cursor = 0
    for s in gpu_plan:
        assert s["lo"] == cursor
        cursor = s["hi"]
    assert cursor == 93
    assert gpu_plan[0]["head"] and gpu_plan[-1]["tail"]


# ── (c) launch-spec: 2 boxes x 4 GPUs = 8 stages, loopback intra-box + 1 WAN forward per boundary ────

def test_two_box_four_gpu_launch_spec_wires_loopback_and_wan():
    """8 stage specs. Every intra-box handoff is loopback (127.0.0.1 to the next local GPU); only the
    box's LAST stage crosses the WAN (the sidecar FWD_RING leg). 2 boxes => 6 loopback links + 1 WAN
    forward + 1 tail, and the sidecar carries B=2 WAN hops total (1 ring forward + 1 head->tail
    return)."""
    nodes = _node_stages(KP.even_tiling(93, 2))            # 2 boxes, ~47 layers each
    maddrs = [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(2)]
    plan = KP.box_ring_launch(nodes, 4, maddrs)
    stages = plan["stages"]

    assert len(stages) == 8                                # B*G
    links = [s["link"] for s in stages]
    # 8 stages in a line = 7 forward links + the tail terminator. 3 loopback per box (4 GPUs) x 2
    # boxes = 6 loopback, and exactly 1 crosses the box boundary over the WAN.
    assert links.count("loopback") == 6
    assert links.count("wan") == 1
    assert links.count("tail") == 1
    # the one WAN forward is the FIRST box's last GPU handing off to the next box
    wan = next(s for s in stages if s["link"] == "wan")
    assert wan["box_index"] == 0 and wan["box_tail"] and wan["next"] == f"127.0.0.1:{KP.FWD_RING}"
    # the tail is the LAST box's last GPU, and it does not forward
    tail = next(s for s in stages if s["link"] == "tail")
    assert tail["box_index"] == 1 and tail["tail"] and tail["next"] is None

    # each box's first local stage binds ENG_IN (the sidecar ingress / coordinator dial); ports are
    # distinct within a box and every stage is pinned to its own local GPU.
    for k in range(2):
        box = [s for s in stages if s["box_index"] == k]
        assert box[0]["eng_port"] == KP.ENG_IN and box[0]["box_head"]
        assert len({s["eng_port"] for s in box}) == 4
        assert [s["gpu"] for s in box] == [0, 1, 2, 3]
        for s in box:
            assert f"CUDA_VISIBLE_DEVICES={s['gpu']} " in s["cmd"]
            assert f"--stage {s['global_index']} --nstages 8 " in s["cmd"]

    # sidecar wiring is at BOX granularity: B=2 boxes => 2 WAN hops (1 ring forward + 1 return).
    assert plan["sidecars"] is not None and len(plan["sidecars"]) == 2
    assert sum(len(fwd) for _, fwd, _ in plan["sidecars"]) == 2
    inb_head, fwd_head, allow_head = plan["sidecars"][0]
    assert inb_head == "" and allow_head is None                          # head box: no inbound, open
    assert any(x.startswith(f"127.0.0.1:{KP.FWD_RET}=") for x in fwd_head)  # head tunnels the coord-return


def test_box_ring_launch_topology_only_without_maddrs():
    """box_maddrs omitted => stage specs but no sidecar wiring (the topology alone, e.g. for a unit
    check or a co-located single-box ring)."""
    nodes = _node_stages(KP.even_tiling(20, 1))
    plan = KP.box_ring_launch(nodes, 4)
    assert plan["sidecars"] is None
    assert len(plan["stages"]) == 4
    # a single box: 3 loopback links + tail, NO WAN forward (nothing leaves the box)
    links = [s["link"] for s in plan["stages"]]
    assert links.count("loopback") == 3 and links.count("wan") == 0 and links.count("tail") == 1


# ── (e) the RETURN path: a G>1 tail box bridges its ingress <-> its last-GPU stage ───────────────────

def test_multigpu_tail_box_ingress_carries_a_return_relay_to_the_box_tail():
    """The coordinator-return tunnel lands at the tail box's ENG_IN (its ingress), but the token is
    produced on the box's LAST GPU. So ONLY that box's ingress gets a --ret-relay pointing at the box
    tail's loopback port; every other stage (including the global tail itself) has none."""
    nodes = _node_stages(KP.even_tiling(93, 2))            # 2 boxes
    plan = KP.box_ring_launch(nodes, 4, [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(2)])
    stages = plan["stages"]

    relays = [s for s in stages if s["ret_relay"] is not None]
    assert len(relays) == 1, "exactly one relay: the multi-GPU tail box's ingress"
    relay = relays[0]
    assert relay["box_index"] == 1 and relay["box_head"] and not relay["tail"]   # tail box ingress
    # it bridges to the box tail = the box's last local GPU (local index G-1 = 3 => ENG_LOCAL_BASE+3)
    assert relay["ret_relay"] == f"127.0.0.1:{KP.local_eng_port(3)}"
    assert relay["eng_port"] == KP.ENG_IN                                        # the ingress binds ENG_IN
    assert f"--ret-relay {relay['ret_relay']} " in relay["cmd"]
    # the global tail is NOT a relay (it samples + is dialed BY the relay), and neither is any other
    tail = next(s for s in stages if s["tail"])
    assert tail["ret_relay"] is None and "--ret-relay" not in tail["cmd"]
    assert sum("--ret-relay" in s["cmd"] for s in stages) == 1


def test_return_relay_target_is_the_box_tail_port_across_geometries():
    """box_return_relay marks the ingress of the tail box (nlocal>1) and points it at the box tail's
    loopback port, whatever G — and marks nothing else."""
    for G in (2, 3, 4):
        subs = KP.split_stages_to_gpus(_node_stages(KP.even_tiling(30, 2)), [1, G])
        tail_box = subs[-1]["box_index"]
        marked = [(s["global_index"], KP.box_return_relay(s, tail_box)) for s in subs]
        hits = [gi for gi, r in marked if r is not None]
        ingress = next(s for s in subs if s["box_index"] == tail_box and s["box_head"])
        assert hits == [ingress["global_index"]]              # only the tail box ingress
        assert KP.box_return_relay(ingress, tail_box) == f"127.0.0.1:{KP.local_eng_port(G - 1)}"


def test_single_gpu_tail_box_needs_no_return_relay():
    """A 1-GPU tail box (or an all-single-GPU ring) has box_head == box_tail == the global tail, which
    serves the coordinator-return directly — no relay, and no --ret-relay anywhere. Byte-identical to
    the pre-relay launch spec."""
    # tail box is 1 GPU, head box is 4 GPUs: the relay is a TAIL-box property, so still none
    nodes = _node_stages(KP.even_tiling(93, 2))
    plan = KP.box_ring_launch(nodes, [4, 1], [f"/ip4/10.0.0.{k}/tcp/29600/p2p/PEER{k}" for k in range(2)])
    assert all(s["ret_relay"] is None for s in plan["stages"])
    assert all("--ret-relay" not in s["cmd"] for s in plan["stages"])
    # an all-single-GPU ring: no relay either
    single = KP.box_ring_launch(_node_stages(KP.even_tiling(93, 5)), 1)
    assert all(s["ret_relay"] is None for s in single["stages"])
