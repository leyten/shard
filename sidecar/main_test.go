package main

import (
	"context"
	"crypto/rand"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
)

// newTestHost brings up a loopback-only libp2p host with a throwaway identity.
func newTestHost(t *testing.T) host.Host {
	t.Helper()
	priv, _, err := crypto.GenerateEd25519Key(rand.Reader)
	if err != nil {
		t.Fatalf("key: %v", err)
	}
	h, err := libp2p.New(libp2p.Identity(priv), libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("host: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

func connectHosts(t *testing.T, from, to host.Host) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := from.Connect(ctx, peer.AddrInfo{ID: to.ID(), Addrs: to.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}
}

// TestRunInboundAllowlist: a sidecar with -allow A rejects a stream from stranger C
// before the engine is ever dialed, and passes a stream from A through to the engine.
func TestRunInboundAllowlist(t *testing.T) {
	sidecar := newTestHost(t)
	peerA := newTestHost(t)
	peerC := newTestHost(t)

	// stand-in engine: counts accepted connections, echoes one byte
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("engine listen: %v", err)
	}
	defer ln.Close()
	var accepted atomic.Int32
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			accepted.Add(1)
			go func(c net.Conn) {
				defer c.Close()
				io.Copy(c, c)
			}(c)
		}
	}()

	runInbound(sidecar, ln.Addr().String(), map[peer.ID]bool{peerA.ID(): true})
	connectHosts(t, peerA, sidecar)
	connectHosts(t, peerC, sidecar)

	// stranger C: stream must be reset, engine must see ZERO connections from it
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	sc, err := peerC.NewStream(ctx, sidecar.ID(), activationProto)
	if err == nil {
		sc.Write([]byte{0x42})
		sc.SetReadDeadline(time.Now().Add(3 * time.Second))
		buf := make([]byte, 1)
		if _, err := sc.Read(buf); err == nil {
			t.Fatalf("stranger's stream was bridged to the engine (read succeeded)")
		}
		sc.Reset()
	}
	time.Sleep(200 * time.Millisecond) // let any (wrong) engine dial land
	if n := accepted.Load(); n != 0 {
		t.Fatalf("engine accepted %d connection(s) for a non-allowed peer", n)
	}

	// allowed A: stream reaches the engine and round-trips a byte
	sa, err := peerA.NewStream(ctx, sidecar.ID(), activationProto)
	if err != nil {
		t.Fatalf("allowed stream: %v", err)
	}
	defer sa.Close()
	if _, err := sa.Write([]byte{0x7}); err != nil {
		t.Fatalf("allowed write: %v", err)
	}
	sa.SetReadDeadline(time.Now().Add(5 * time.Second))
	buf := make([]byte, 1)
	if _, err := io.ReadFull(sa, buf); err != nil || buf[0] != 0x7 {
		t.Fatalf("allowed peer did not reach the engine: %v (buf=%v)", err, buf)
	}
	if n := accepted.Load(); n != 1 {
		t.Fatalf("engine accepted %d connection(s), want exactly 1 (the allowed peer)", n)
	}
}

// TestRunInboundNoAllowlistIsOpen: zero -allow flags preserves the legacy open behavior.
func TestRunInboundNoAllowlistIsOpen(t *testing.T) {
	sidecar := newTestHost(t)
	peerC := newTestHost(t)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("engine listen: %v", err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				io.Copy(c, c)
			}(c)
		}
	}()

	runInbound(sidecar, ln.Addr().String(), nil)
	connectHosts(t, peerC, sidecar)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s, err := peerC.NewStream(ctx, sidecar.ID(), activationProto)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer s.Close()
	if _, err := s.Write([]byte{0x9}); err != nil {
		t.Fatalf("write: %v", err)
	}
	s.SetReadDeadline(time.Now().Add(5 * time.Second))
	buf := make([]byte, 1)
	if _, err := io.ReadFull(s, buf); err != nil || buf[0] != 0x9 {
		t.Fatalf("legacy open mode broken: %v", err)
	}
}
