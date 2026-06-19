// Command sidecar is shard's libp2p transport daemon.
//
// One sidecar runs alongside the Python engine on every node. It owns the node's
// cryptographic identity (an ed25519 keypair -> libp2p PeerId) and carries the engine's
// inter-stage connections to/from its ring neighbours over authenticated, encrypted
// libp2p streams. It runs as a transparent TCP<->libp2p tunnel: the engine keeps its
// plain socket code and just talks to localhost; the sidecar does the network. This is
// what replaces the shared-SHARD_PSK TCP wire (phase0/wire.py).
//
// Per the boundary law (docs/INTEGRATION.md): pure engine plumbing. It knows nothing
// about $ZERO, accounts, payments, or the orchestrator — only peers and bytes.
//
// Modes:
//   -inbound HOST:PORT            tunnel: dial the local engine for each inbound stream
//   -forward LOCAL=PEER_MULTIADDR tunnel: listen LOCAL, carry each conn to PEER (repeatable)
//   -peer PEER_MULTIADDR          self-test: round-trip one frame to a listener (connectivity check)
//   (none)                        self-test listener: echo one frame back
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	relayclient "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	"github.com/multiformats/go-multiaddr"
)

// activationProto is the stream protocol carrying inter-stage traffic (and the self-test).
const activationProto = "/shard/activation/1.0.0"

// stringList is a repeatable string flag (used for -forward).
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// loadOrCreateKey returns a stable node identity, persisting it to keyPath so a node
// keeps the same PeerId across restarts. This per-node key is what replaces the shared
// SHARD_PSK: a node proves who it is by holding this key, not by knowing a secret.
func loadOrCreateKey(keyPath string) (crypto.PrivKey, error) {
	if keyPath != "" {
		if b, err := os.ReadFile(keyPath); err == nil {
			return crypto.UnmarshalPrivateKey(b)
		}
	}
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		return nil, err
	}
	if keyPath != "" {
		b, err := crypto.MarshalPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(keyPath, b, 0o600); err != nil {
			return nil, err
		}
	}
	return priv, nil
}

// writeFrame / readFrame: a 4-byte big-endian length prefix + payload (the self-test wire).
func writeFrame(w io.Writer, b []byte) error {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(b)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	b := make([]byte, binary.BigEndian.Uint32(hdr[:]))
	_, err := io.ReadFull(r, b)
	return b, err
}

type natOpts struct {
	quic         bool
	relayService bool
	announce     string
	staticRelays []peer.AddrInfo
}

// tcpToQuic derives a QUIC listen addr from a TCP one: /ip4/x/tcp/P -> /ip4/x/udp/P/quic-v1.
func tcpToQuic(maddr string) string {
	i := strings.Index(maddr, "/tcp/")
	if i < 0 {
		return ""
	}
	port := maddr[i+len("/tcp/"):]
	if j := strings.Index(port, "/"); j >= 0 {
		port = port[:j]
	}
	return maddr[:i] + "/udp/" + port + "/quic-v1"
}

func newHost(priv crypto.PrivKey, listen string, n natOpts) (host.Host, error) {
	// libp2p defaults give Noise/TLS encryption + a stream muxer; every link is
	// authenticated to the peer's key. The NAT stack (DCUtR hole-punching + circuit
	// relay) lets home GPUs behind NAT join; QUIC (udp) hole-punches more reliably.
	listens := []string{listen}
	if n.quic {
		if q := tcpToQuic(listen); q != "" {
			listens = append(listens, q)
		}
	}
	opts := []libp2p.Option{
		libp2p.Identity(priv),
		libp2p.ListenAddrStrings(listens...),
		libp2p.EnableHolePunching(), // DCUtR: punch a direct hole between two NAT'd peers
	}
	if n.announce != "" {
		// libp2p only sees container-internal addrs behind Vast's port mapping; advertise
		// the real public addr so reservations/circuit addrs others get are actually dialable.
		ann, err := multiaddr.NewMultiaddr(n.announce)
		if err != nil {
			return nil, err
		}
		opts = append(opts, libp2p.AddrsFactory(func(addrs []multiaddr.Multiaddr) []multiaddr.Multiaddr {
			return append([]multiaddr.Multiaddr{ann}, addrs...)
		}))
	}
	if n.relayService {
		// be a public relay (circuit-relay-v2) + an AutoNAT server (so NAT'd peers can
		// learn their reachability + observed address from us). Force public reachability
		// so the hop service activates immediately.
		opts = append(opts, libp2p.EnableRelayService(), libp2p.ForceReachabilityPublic(), libp2p.EnableNATService())
	}
	// NAT'd nodes reserve on relays explicitly (in main) and let AutoNAT + the
	// observed-address manager (from several observer peers) determine reachability — so
	// DCUtR can hole-punch when the NAT is cone-type. Forcing private here is wrong: it
	// leaves holepunch with no public address to offer ("waiting for a public address").
	return libp2p.New(opts...)
}

// fullAddrs returns this host's dialable /p2p multiaddrs (addr + /p2p/<peerid>).
func fullAddrs(h host.Host) []string {
	p2p := multiaddr.StringCast("/p2p/" + h.ID().String())
	out := make([]string, 0, len(h.Addrs()))
	for _, a := range h.Addrs() {
		out = append(out, a.Encapsulate(p2p).String())
	}
	return out
}

func main() {
	keyPath := flag.String("key", "", "path to persist the node key (keeps PeerId stable)")
	peerAddr := flag.String("peer", "", "self-test: dial this /p2p multiaddr and round-trip a frame")
	addrFile := flag.String("addrfile", "", "write this host's dial multiaddr here (for scripting)")
	listenAddr := flag.String("listen", "/ip4/0.0.0.0/tcp/0", "libp2p listen multiaddr; pin the port for cross-box reach, e.g. /ip4/0.0.0.0/tcp/29600")
	inbound := flag.String("inbound", "", "tunnel: dial this local engine addr (host:port) for each inbound libp2p stream")
	var forwards stringList
	flag.Var(&forwards, "forward", "tunnel: localAddr=peerMultiaddr — listen localAddr, carry each conn to the peer (repeatable)")
	size := flag.Int("size", 1<<20, "self-test frame size in bytes (default 1 MiB)")
	relaySvc := flag.Bool("relay", false, "run as a circuit-relay-v2 server (public rendezvous for NAT'd nodes)")
	relaysCSV := flag.String("relays", "", "comma-separated relay /p2p multiaddrs to use when behind NAT")
	useQuic := flag.Bool("quic", false, "also listen on QUIC (udp) — better hole-punching + lossy links")
	announce := flag.String("announce", "", "advertise this public multiaddr ahead of auto-detected ones (e.g. /ip4/PUBIP/tcp/PORT)")
	flag.Parse()

	priv, err := loadOrCreateKey(*keyPath)
	if err != nil {
		log.Fatalf("key: %v", err)
	}
	var staticRelays []peer.AddrInfo
	for _, s := range strings.Split(*relaysCSV, ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		ma, err := multiaddr.NewMultiaddr(s)
		if err != nil {
			log.Fatalf("bad -relays entry %q: %v", s, err)
		}
		ai, err := peer.AddrInfoFromP2pAddr(ma)
		if err != nil {
			log.Fatalf("bad -relays entry %q: %v", s, err)
		}
		staticRelays = append(staticRelays, *ai)
	}
	h, err := newHost(priv, *listenAddr, natOpts{quic: *useQuic, relayService: *relaySvc, announce: *announce, staticRelays: staticRelays})
	if err != nil {
		log.Fatalf("host: %v", err)
	}
	defer h.Close()
	log.Printf("peer id: %s", h.ID())
	addrs := fullAddrs(h)
	for _, a := range addrs {
		fmt.Printf("ADDR %s\n", a)
	}
	if *addrFile != "" {
		pick := addrs[0]
		for _, a := range addrs {
			if strings.Contains(a, "127.0.0.1") {
				pick = a
				break
			}
		}
		if err := os.WriteFile(*addrFile, []byte(pick), 0o644); err != nil {
			log.Fatalf("addrfile: %v", err)
		}
	}

	go monitorConns(h) // log RELAY vs DIRECT connections so we can watch DCUtR upgrade

	// NAT'd node: explicitly reserve a slot on each relay and keep the connection
	// protected, so the relay can forward inbound connections to us. Clear errors,
	// no autorelay guesswork.
	for _, relay := range staticRelays {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := h.Connect(ctx, relay); err != nil {
			log.Printf("relay connect %s: %v", relay.ID, err)
			cancel()
			continue
		}
		res, err := relayclient.Reserve(ctx, h, relay)
		cancel()
		if err != nil {
			log.Printf("relay reserve %s: %v", relay.ID, err)
			continue
		}
		h.ConnManager().Protect(relay.ID, "relay")
		log.Printf("RESERVED relay slot on %s (expires %s)", relay.ID, res.Expiration)
	}

	// Tunnel mode: a transparent TCP<->libp2p bridge. The engine keeps its own socket
	// code and just talks to localhost; the sidecar carries each connection to/from the
	// right ring neighbour over libp2p. This is what replaces wire.py's TCP.
	if *inbound != "" || len(forwards) > 0 {
		if *inbound != "" {
			runInbound(h, *inbound)
		}
		for _, f := range forwards {
			pp := strings.SplitN(f, "=", 2)
			if len(pp) != 2 {
				log.Fatalf("bad -forward %q (want localAddr=peerMultiaddr)", f)
			}
			go runForward(h, pp[0], pp[1])
		}
		log.Printf("tunnel up (inbound=%q forwards=%v)", *inbound, []string(forwards))
		select {}
	}

	// Self-test dialer: connect by multiaddr, round-trip a frame, verify it byte-for-byte.
	// libp2p's Noise handshake guarantees the peer holds the key in the multiaddr.
	if *peerAddr != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		maddr, err := multiaddr.NewMultiaddr(*peerAddr)
		if err != nil {
			log.Fatalf("bad -peer: %v", err)
		}
		info, err := peer.AddrInfoFromP2pAddr(maddr)
		if err != nil {
			log.Fatalf("bad -peer: %v", err)
		}
		if err := h.Connect(ctx, *info); err != nil {
			log.Fatalf("connect: %v", err)
		}
		s, err := h.NewStream(ctx, info.ID, activationProto)
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		defer s.Close()
		log.Printf("connected to %s (key-authenticated)", s.Conn().RemotePeer())

		blob := make([]byte, *size)
		if _, err := rand.Read(blob); err != nil {
			log.Fatalf("rand: %v", err)
		}
		start := time.Now()
		if err := writeFrame(s, blob); err != nil {
			log.Fatalf("send: %v", err)
		}
		got, err := readFrame(s)
		if err != nil {
			log.Fatalf("recv: %v", err)
		}
		rtt := time.Since(start)
		if !bytes.Equal(got, blob) {
			log.Fatalf("ROUND-TRIP MISMATCH: sent %d bytes, got %d", len(blob), len(got))
		}
		fmt.Printf("ROUND-TRIP OK: %d bytes echoed by %s in %v\n", len(blob), s.Conn().RemotePeer(), rtt)
		return
	}

	// Self-test listener: echo any frame back to the sender (connectivity check).
	h.SetStreamHandler(activationProto, func(s network.Stream) {
		defer s.Close()
		b, err := readFrame(s)
		if err != nil {
			log.Printf("recv: %v", err)
			return
		}
		log.Printf("recv %d bytes from %s", len(b), s.Conn().RemotePeer())
		if err := writeFrame(s, b); err != nil {
			log.Printf("send: %v", err)
		}
	})
	log.Printf("listening; start a second sidecar with -peer <one of the ADDR lines above>")
	select {} // serve until killed
}

// openStream dials a peer by multiaddr and opens an activation stream. libp2p's
// Noise handshake guarantees the peer holds the key named in the multiaddr.
func openStream(h host.Host, peerAddr string) (network.Stream, error) {
	maddr, err := multiaddr.NewMultiaddr(peerAddr)
	if err != nil {
		return nil, err
	}
	info, err := peer.AddrInfoFromP2pAddr(maddr)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := h.Connect(ctx, *info); err != nil {
		return nil, err
	}
	return h.NewStream(ctx, info.ID, activationProto)
}

// monitorConns logs each new connection and whether it's via a relay or DIRECT — so we
// can watch DCUtR upgrade a relay rendezvous into a direct hole-punched link.
func monitorConns(h host.Host) {
	seen := map[string]bool{}
	for {
		time.Sleep(5 * time.Second)
		for _, c := range h.Network().Conns() {
			a := c.RemoteMultiaddr().String()
			kind := "DIRECT"
			if strings.Contains(a, "p2p-circuit") {
				kind = "RELAY"
			}
			key := c.RemotePeer().String() + "|" + a
			if !seen[key] {
				seen[key] = true
				log.Printf("CONN %s via %s [%s]", c.RemotePeer(), a, kind)
			}
		}
	}
}

// pipe copies bytes bidirectionally between two streams until either side closes.
func pipe(a, b io.ReadWriteCloser) {
	done := make(chan struct{}, 2)
	cp := func(dst io.Writer, src io.Reader) { io.Copy(dst, src); done <- struct{}{} }
	go cp(a, b)
	go cp(b, a)
	<-done
	a.Close()
	b.Close()
}

// runForward listens on a local TCP addr; each accepted connection is carried to the
// peer over a fresh libp2p stream — so the engine dials localhost and reaches the peer.
func runForward(h host.Host, listenAddr, peerMaddr string) {
	// pre-establish the connection so DCUtR can upgrade relay->direct BEFORE data flows
	// (otherwise the engine's first stream lands on the slow relay connection).
	if ma, err := multiaddr.NewMultiaddr(peerMaddr); err == nil {
		if ai, err := peer.AddrInfoFromP2pAddr(ma); err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			if err := h.Connect(ctx, *ai); err != nil {
				log.Printf("forward pre-connect %s: %v", ai.ID, err)
			}
			cancel()
		}
	}
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("forward listen %s: %v", listenAddr, err)
	}
	log.Printf("forward %s -> %s", listenAddr, peerMaddr)
	for {
		c, err := ln.Accept()
		if err != nil {
			log.Printf("forward accept: %v", err)
			return
		}
		go func() {
			s, err := openStream(h, peerMaddr)
			if err != nil {
				log.Printf("forward dial: %v", err)
				c.Close()
				return
			}
			pipe(c, s)
		}()
	}
}

// runInbound pipes each inbound libp2p stream to a fresh connection to the local
// engine — so the engine accepts on localhost, fed by its ring neighbours.
func runInbound(h host.Host, engineAddr string) {
	h.SetStreamHandler(activationProto, func(s network.Stream) {
		c, err := net.Dial("tcp", engineAddr)
		if err != nil {
			log.Printf("inbound -> engine %s: %v", engineAddr, err)
			s.Reset()
			return
		}
		pipe(s, c)
	})
}
