"""Device-capability records for mesh tests and vectors: one DeviceInfo per device family.

These are what `StageRuntime.describe().device` would report (shard/runtime/api.py), not card
names placement ever looks at. free_mem_bytes is what the backend can still allocate: roughly
torch's view of a discrete card at idle, the wired limit (raised) on a Mac, total minus the OS on a
DGX Spark or CPU box. Bandwidths are spec-sheet values (mem_bw_source "spec"). Format support
follows the hardware: Ampere has no FP8/FP4 units (emulated), Ada adds FP8, Blackwell adds FP4,
RDNA3 has neither and no MXFP4 kernel in our stack yet.
"""
GiB = 1 << 30

_AMPERE = (("fp32", "bf16", "fp16", "int8", "int4"), ("fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"))
_ADA = (("fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"), ("mxfp4", "nvfp4"))
_BLACKWELL = (("fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4", "int8"), ("int4",))
_APPLE = (("fp32", "bf16", "fp16"), ("int8", "int4", "mxfp4", "fp8_e4m3"))
_CPU = (("fp32",), ("bf16", "fp16", "int8", "int4", "fp8_e4m3", "mxfp4"))
_RDNA3 = (("fp32", "bf16", "fp16", "int8"), ("int4",))

# kind: (backend, device_name, arch_class, host_arch, total GiB, free GiB, unified, bw GB/s, formats)
DEVICES = {
    "4060-8g":     ("cuda", "RTX 4060", "sm_89", "x86_64", 8, 7.6, False, 272, _ADA),
    "3060-12g":    ("cuda", "RTX 3060", "sm_86", "x86_64", 12, 11.6, False, 360, _AMPERE),
    "4060ti-16g":  ("cuda", "RTX 4060 Ti 16GB", "sm_89", "x86_64", 16, 15.5, False, 288, _ADA),
    "3090":        ("cuda", "RTX 3090", "sm_86", "x86_64", 24, 23.3, False, 936, _AMPERE),
    "4090":        ("cuda", "RTX 4090", "sm_89", "x86_64", 24, 23.3, False, 1008, _ADA),
    "7900xtx":     ("rocm", "RX 7900 XTX", "gfx1100", "x86_64", 24, 23.5, False, 960, _RDNA3),
    "5090":        ("cuda", "RTX 5090", "sm_120", "x86_64", 32, 31.0, False, 1792, _BLACKWELL),
    "a6000-48g":   ("cuda", "RTX A6000", "sm_86", "x86_64", 48, 47.2, False, 768, _AMPERE),
    "pro6000-96g": ("cuda", "RTX PRO 6000", "sm_120", "x86_64", 96, 94.5, False, 1792, _BLACKWELL),
    "spark-128g":  ("cuda", "DGX Spark GB10", "sm_121", "aarch64", 128, 112.0, True, 273, _BLACKWELL),
    "mac-64g":     ("mlx", "Mac M4 Max 64GB", "apple_m4", "arm64", 64, 52.0, True, 546, _APPLE),
    "mac-192g":    ("mlx", "Mac M2 Ultra 192GB", "apple_m2", "arm64", 192, 176.0, True, 800, _APPLE),
    "mac-512g":    ("mlx", "Mac M3 Ultra 512GB", "apple_m3", "arm64", 512, 496.0, True, 819, _APPLE),
    "cpu-256g":    ("cpu", "EPYC 256GB", "cpu-x86_64", "x86_64", 256, 240.0, True, 100, _CPU),
}


def device(kind: str) -> dict:
    be, name, arch, host, total, free, unified, bw, (hw, emu) = DEVICES[kind]
    return {"backend": be, "device_name": name, "arch_class": arch, "host_arch": host, "n_devices": 1,
            "total_mem_bytes": total * GiB, "free_mem_bytes": int(free * GiB), "unified_memory": unified,
            "mem_bw_bytes_s": bw * 1e9, "mem_bw_source": "spec",
            "formats_hw": list(hw), "formats_emulated": list(emu)}


def node(nid: str, kind: str, cell: str = "eu-1", **kw) -> dict:
    """A node record as the directory receives it (placement.Node.from_dict)."""
    d = {"id": nid, "cell": cell, "subnet": kw.pop("subnet", f"s-{nid}"), "device": device(kind),
         "reliability": 1.0, "uplink_mbps": 200.0}
    d.update(kw)
    return d


# The catalog's small whole-server model (EST need: 9B at ~4.5 bit + KV + runtime).
DISTILL_9B = {"name": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "need_bytes": 7_000_000_000,
              "formats": ["int4", "bf16"]}
