"""shard v2 mesh data plane (ARCHITECTURE-v2 §4.2-§4.4): node server, links, clocks.

`clock`: injectable time (RealClock in production, VirtualClock in the fault harness).
`links`: dial-by-PeerId stream pool + peer-tagged inbound streams over TCP, the loopback
stand-in for sidecar v2 (§4.6).
`node`:  the node server that serves one layer span through a StageRuntime.
"""
