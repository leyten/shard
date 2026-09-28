"""shard v2 mesh: many holders per layer range, per-session routed paths (ARCHITECTURE-v2 §4.2-§4.4).

Data plane:
  clock      injectable time (RealClock in production, VirtualClock in the fault harness)
  links      dial-by-PeerId stream pool + peer-tagged inbound streams over TCP, the loopback
             stand-in for sidecar v2 (§4.6)
  node       the node server that serves one layer span through a StageRuntime

Control plane:
  placement  which layers each device of a mixed fleet holds, its role, candidate paths (pure)
  lease      orchestrator-signed lease tokens and the node-side admission gate
  directory  reference directory: holder registry, load bumps, lease issuance, path query

The c0mpute orchestrator implements the same directory API in TypeScript; both must pass
tests/vectors/mesh_*.json.
"""
