package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
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

	runInbound(sidecar, ln.Addr().String(), map[peer.ID]bool{peerA.ID(): true}, 0)
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

	runInbound(sidecar, ln.Addr().String(), nil, 0)
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

// --- H4: absolute per-frame deadline in the tunnel pipe ---

// frame8 builds an engine-style frame: 8-byte big-endian length prefix + body.
func frame8(body []byte) []byte {
	var hdr [8]byte
	binary.BigEndian.PutUint64(hdr[:], uint64(len(body)))
	return append(hdr[:], body...)
}

// tunnelPair wires pipeFramed between two net.Pipe pairs, returning the outer ends:
// client -> [p1a p1b] -> pipeFramed -> [p2a p2b] -> engine.
func tunnelPair(T time.Duration) (client, engine net.Conn) {
	p1a, p1b := net.Pipe()
	p2a, p2b := net.Pipe()
	go pipeFramed(p1b, p2a, T)
	return p1a, p2b
}

// TestFramedStallMidFrameClosesTunnel: a peer that sends part of a frame then stalls
// (slow-loris) must have the tunnel torn down at the absolute deadline.
func TestFramedStallMidFrameClosesTunnel(t *testing.T) {
	T := 100 * time.Millisecond
	client, engine := tunnelPair(T)
	defer client.Close()
	defer engine.Close()

	if _, err := client.Write([]byte{0x00}); err != nil { // 1 prefix byte, then silence
		t.Fatalf("write: %v", err)
	}
	engine.SetReadDeadline(time.Now().Add(5 * T))
	buf := make([]byte, 16)
	start := time.Now()
	_, err := engine.Read(buf)
	if err == nil {
		t.Fatalf("expected the tunnel to close, got data")
	}
	if nerr, ok := err.(net.Error); ok && nerr.Timeout() {
		t.Fatalf("tunnel still open after %v (read timed out instead of conn closing)", time.Since(start))
	}
	if elapsed := time.Since(start); elapsed > 3*T {
		t.Fatalf("tunnel closed after %v, want ~%v", elapsed, T)
	}
}

// TestFramedPreFrameIdleUnlimited: silence BETWEEN frames never trips the deadline —
// only mid-frame stalls do.
func TestFramedPreFrameIdleUnlimited(t *testing.T) {
	T := 100 * time.Millisecond
	client, engine := tunnelPair(T)
	defer client.Close()
	defer engine.Close()

	time.Sleep(3 * T) // pre-frame idle, 3x the frame deadline
	body := []byte("hello-after-idle")
	go client.Write(frame8(body))
	engine.SetReadDeadline(time.Now().Add(5 * time.Second))
	got := make([]byte, 8+len(body))
	if _, err := io.ReadFull(engine, got); err != nil {
		t.Fatalf("frame after long idle should pass: %v", err)
	}
	if !bytes.Equal(got, frame8(body)) {
		t.Fatalf("frame corrupted through tunnel")
	}
}

// TestFramedLegitFramePassesByteIdentical: a normal frame flows through untouched.
func TestFramedLegitFramePassesByteIdentical(t *testing.T) {
	T := 2 * time.Second
	client, engine := tunnelPair(T)
	defer client.Close()
	defer engine.Close()

	body := make([]byte, 64<<10)
	rand.Read(body)
	go client.Write(frame8(body))
	engine.SetReadDeadline(time.Now().Add(5 * time.Second))
	got := make([]byte, 8+len(body))
	if _, err := io.ReadFull(engine, got); err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, frame8(body)) {
		t.Fatalf("tunneled frame not byte-identical")
	}
}

// TestFramedOversizePrefixClosesTunnel: a lying length prefix (> maxFrame) must close
// the tunnel instead of making it copy up to 16 EB.
func TestFramedOversizePrefixClosesTunnel(t *testing.T) {
	T := 500 * time.Millisecond
	client, engine := tunnelPair(T)
	defer client.Close()
	defer engine.Close()

	var hdr [8]byte
	binary.BigEndian.PutUint64(hdr[:], uint64(maxFrame)+1)
	if _, err := client.Write(hdr[:]); err != nil {
		t.Fatalf("write: %v", err)
	}
	engine.SetReadDeadline(time.Now().Add(3 * time.Second))
	buf := make([]byte, 8)
	_, err := engine.Read(buf)
	if err == nil {
		t.Fatalf("oversize prefix was forwarded")
	}
	if nerr, ok := err.(net.Error); ok && nerr.Timeout() {
		t.Fatalf("tunnel still open after oversize prefix")
	}
}
