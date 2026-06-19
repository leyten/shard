# sidecar — shard's libp2p transport daemon

A Go (`go-libp2p`) daemon that runs next to the Python engine on every node. It owns the
node's keypair identity (an ed25519 key → libp2p PeerId) and moves activation frames
between adjacent pipeline stages over authenticated, encrypted libp2p streams. Per the
boundary law ([../docs/INTEGRATION.md](../docs/INTEGRATION.md)) it knows only **peers and
bytes** — nothing about `$ZERO`, accounts, payments, or the orchestrator.

It's a single static binary (no CGO), so we build once and `scp` it to any linux/amd64
node — a node needs no Go toolchain to run it.

## Build

```sh
cd sidecar
GOTOOLCHAIN=local GOPROXY=https://goproxy.io,direct go build -o /tmp/sidecar .
```

Two load-bearing build pins (in the spirit of `phase0/setup_box.sh`'s `kernels` pin):

- **`go-libp2p v0.33.2`** — builds on Go 1.22 (our toolchain); newer releases want Go 1.23/1.25+.
  It still ships DCUtR + circuit-relay-v2, so step 2 (NAT traversal) isn't blocked by the pin.
- **`GOPROXY=goproxy.io`** — `proxy.golang.org` returns `403` on the `klauspost/compress` zip
  (and Go only falls back to `direct` on `404`, not `403`), which stalls the whole resolve.
  `goproxy.io` serves it fine.

## Run as a tunnel (how the engine uses it)

The sidecar is a transparent TCP↔libp2p tunnel. Pin the libp2p port to the node's
public-mapped port; the engine then dials/listens on localhost and the sidecar carries
each connection to/from the right ring neighbour:

```sh
# a stage: accept inbound streams -> the local engine, and carry the engine's
# next-hop connection to the downstream peer over libp2p
sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/29600 \
        -inbound 127.0.0.1:29610 \
        -forward 127.0.0.1:29611=/ip4/<peer_ip>/tcp/<peer_port>/p2p/<peer_id>
```

The engine runs unchanged except one import: `import wire` → `import shard.transport as
wire` (see `../shard/transport.py`). Proven: gpt-oss-120B across 4 scattered boxes over
libp2p, no `SHARD_PSK`, bit-identical to the trusted-wire receipt
(`../docs/receipts/gpt-oss-120b-libp2p-20260619.json`).

## Self-test — identity + connect-by-key + round-trip (connectivity check)

```sh
# terminal A (listener): prints its PeerId + dialable multiaddr, writes the addr to a file
/tmp/sidecar -key /tmp/a.key -addrfile /tmp/addrA

# terminal B (dialer): connects by key, round-trips a 2 MiB activation-sized frame
/tmp/sidecar -key /tmp/b.key -peer "$(cat /tmp/addrA)" -size 2097152
# -> ROUND-TRIP OK: 2097152 bytes echoed by <A's PeerId> ...
```

`ROUND-TRIP OK` with two distinct PeerIds = the transport is up, authenticated by key, and
intact. No `SHARD_PSK` anywhere.
