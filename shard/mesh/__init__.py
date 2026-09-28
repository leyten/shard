"""shard v2 mesh: many holders per layer range, per-session routed paths (ARCHITECTURE-v2 §4.2-§4.4).

  placement  which layers each device of a mixed fleet holds, its role, candidate paths (pure)
  lease      orchestrator-signed lease tokens and the node-side admission gate
  directory  reference directory: holder registry, load bumps, lease issuance, path query

The c0mpute orchestrator implements the same directory API in TypeScript; both must pass
tests/vectors/mesh_*.json.
"""
