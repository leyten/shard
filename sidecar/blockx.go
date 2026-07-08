// Block exchange — the torrent half of the verified weight-fetch (step 8).
//
// A node that HOLDS manifest shards seeds them: it announces each shard's CID on the
// shard DHT (PROVIDE) and serves the bytes over the blockx protocol. A JOINING node
// fetches a shard by CID: FIND-PROVIDERS on the DHT, then block-exchange from the
// first peer that serves it, resuming by offset like the HTTP mirror path.
//
// Trust model (why an untrusted seeder is safe): the transfer carries ZERO trust.
// The Python fetcher (shard/fetch.py) re-hashes every byte of every file against the
// SIGNED manifest before the loader may touch it — a hostile seeder can waste time,
// never poison weights. That's why this file has no verification code: verification
// deliberately lives on the other side of the seam, where it also covers the mirror.
//
// Per the boundary law: peers and bytes. The manifest is read only to learn
// (cid -> local file) — signatures are not checked here (the fetcher checks them).
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ipfs/go-cid"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// blockxProto requests one manifest shard by CID and streams its bytes.
const blockxProto = "/shard/blockx/1.0.0"

// shardDHTPrefix isolates shard provider records from the public IPFS DHT.
const shardDHTPrefix = "/shard"

// maxCtrlFrame caps the JSON control frames (request + response header). readFrame
// otherwise make()s whatever a 4-byte length says — up to 4 GiB from 4 bytes, a remote
// OOM from one hostile packet in EITHER direction. The control frames are &lt;1 KiB.
const maxCtrlFrame = 1 << 20

// idleTimeout aborts a transfer whose bytes stop flowing (slowloris / dead peer). It's
// an IDLE deadline, reset each chunk, so a slow-but-live 5 GB pull is fine; only silence
// trips it. Enforced with SetDeadline because a stream honors deadlines, not ctx.
const idleTimeout = 60 * time.Second

// readCtrlFrame is readFrame with a size cap and a read deadline — for the small JSON
// control frames only (never the bulk body, which streams under its own idle deadline).
func readCtrlFrame(s network.Stream) ([]byte, error) {
	if err := s.SetReadDeadline(time.Now().Add(idleTimeout)); err != nil {
		return nil, err
	}
	var hdr [4]byte
	if _, err := io.ReadFull(s, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > maxCtrlFrame {
		return nil, fmt.Errorf("control frame too large: %d > %d", n, maxCtrlFrame)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(s, b)
	return b, err
}

// copyIdle streams up to `want` bytes src->dst, resetting an idle deadline each chunk so
// a stalled peer aborts instead of hanging past the fetch timeout. Returns bytes copied.
func copyIdle(s network.Stream, dst io.Writer, src io.Reader, want int64) (int64, error) {
	buf := make([]byte, 1<<20)
	var done int64
	for done < want {
		if err := s.SetReadDeadline(time.Now().Add(idleTimeout)); err != nil {
			return done, err
		}
		chunk := int64(len(buf))
		if r := want - done; r < chunk {
			chunk = r
		}
		n, err := src.Read(buf[:chunk])
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return done, werr
			}
			done += int64(n)
		}
		if err == io.EOF {
			return done, io.ErrUnexpectedEOF // short body: fewer bytes than declared
		}
		if err != nil {
			return done, err
		}
	}
	return done, nil
}

type blockxReq struct {
	Cid    string `json:"cid"`
	Offset int64  `json:"offset"` // resume point: serve bytes [Offset, size)
}

type blockxResp struct {
	Size int64  `json:"size"` // full file size (not remaining bytes)
	Err  string `json:"err,omitempty"`
}

// setupDHT joins the shard DHT (kad, /shard prefix) and connects the bootstrap peers.
// Seeders run in server mode (they store + answer provider records); a one-shot
// fetcher also runs as a server so two-node nets bootstrap symmetrically — provider
// lookups then work even when the whole "network" is one seeder and one fetcher.
// Returns the successfully connected bootstrap peers: the fetcher direct-dials them
// when a DHT lookup comes up dry (known peers beat record propagation, torrent-style).
func setupDHT(ctx context.Context, h host.Host, bootstrap []string) (*dht.IpfsDHT, []peer.AddrInfo, error) {
	d, err := dht.New(ctx, h, dht.Mode(dht.ModeServer), dht.ProtocolPrefix(shardDHTPrefix))
	if err != nil {
		return nil, nil, err
	}
	var peers []peer.AddrInfo
	for _, b := range bootstrap {
		if b = strings.TrimSpace(b); b == "" {
			continue
		}
		ma, err := multiaddr.NewMultiaddr(b)
		if err != nil {
			log.Printf("dht bootstrap addr %q: %v", b, err)
			continue
		}
		ai, err := peer.AddrInfoFromP2pAddr(ma)
		if err != nil {
			log.Printf("dht bootstrap addr %q: %v", b, err)
			continue
		}
		cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		if err := h.Connect(cctx, *ai); err != nil {
			log.Printf("dht bootstrap connect %s: %v", ai.ID, err)
		} else {
			peers = append(peers, *ai)
		}
		cancel()
	}
	if len(bootstrap) > 0 && len(peers) == 0 {
		// NOT fatal: a seeder must outlive a dead neighbour (its DHT still serves inbound
		// joiners), and a fetcher with no reachable peer just reports "no providers" and
		// exits nonzero — the Python chain falls back to the mirror either way.
		log.Printf("dht: no bootstrap peer reachable (%d tried); continuing isolated", len(bootstrap))
	}
	if err := d.Bootstrap(ctx); err != nil {
		return nil, nil, err
	}
	return d, peers, nil
}

// waitRoutingTable blocks until the DHT routing table has a peer (or the deadline).
// Bootstrap is async — a lookup fired before the table populates terminates instantly
// with zero results (the exact 2-peer race the local test caught).
func waitRoutingTable(d *dht.IpfsDHT, deadline time.Duration) {
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		if d.RoutingTable().Size() > 0 {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// manifestShards reads a manifest (shard/manifest.py schema) and maps shard_id (CID)
// -> local path for every shard PRESENT in modelDir at the manifest's exact size.
// Partial or missing files are simply not seeded; a size-mismatched file is never
// served (the fetcher's re-hash would reject it anyway — don't waste its time).
func manifestShards(manifestPath, modelDir string) (map[string]string, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}
	var m struct {
		Shards []struct {
			ShardID string `json:"shard_id"`
			Path    string `json:"path"`
			Size    int64  `json:"size"`
		} `json:"shards"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("manifest parse: %w", err)
	}
	held := map[string]string{}
	for _, s := range m.Shards {
		p := filepath.Join(modelDir, s.Path)
		if st, err := os.Stat(p); err == nil && st.Size() == s.Size {
			held[s.ShardID] = p
		}
	}
	return held, nil
}

// runSeeder serves blockx requests for the held shards and keeps their provider
// records alive on the DHT (records expire; re-provide well inside the window).
func runSeeder(ctx context.Context, h host.Host, d *dht.IpfsDHT, manifestPath, modelDir string) error {
	held, err := manifestShards(manifestPath, modelDir)
	if err != nil {
		return err
	}
	if len(held) == 0 {
		return fmt.Errorf("no complete manifest shards found under %s", modelDir)
	}
	h.SetStreamHandler(blockxProto, func(s network.Stream) { serveBlock(s, held) })
	go func() {
		for {
			ok := 0
			for cs := range held {
				c, err := cid.Decode(cs)
				if err != nil {
					log.Printf("seed: bad shard_id %q: %v", cs, err)
					continue
				}
				pctx, cancel := context.WithTimeout(ctx, 30*time.Second)
				// Provide stores the record locally FIRST, then announces to the
				// network — so a tiny net (seeder + one fetcher) still resolves even
				// while the announce part reports an empty routing table.
				if err := d.Provide(pctx, c, true); err != nil {
					log.Printf("provide %s…: %v", cs[:min(16, len(cs))], err)
				} else {
					ok++
				}
				cancel()
			}
			log.Printf("SEEDING %d shards (%d announced) from %s", len(held), ok, modelDir)
			// fast retry until the network announce lands once (a seeder often starts
			// before any peer exists), then the slow re-provide cadence keeps records live.
			wait := 30 * time.Minute
			if ok < len(held) {
				wait = 15 * time.Second
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(wait):
			}
		}
	}()
	return nil
}

// serveBlock answers one blockx request: a JSON header frame, then raw bytes from
// the requested offset. Only CIDs from OUR manifest map are served — the requester
// never names a path, so there is nothing to traverse.
func serveBlock(s network.Stream, held map[string]string) {
	defer s.Close()
	fail := func(msg string) {
		b, _ := json.Marshal(blockxResp{Err: msg})
		_ = writeFrame(s, b)
	}
	raw, err := readCtrlFrame(s)
	if err != nil {
		return
	}
	var req blockxReq
	if err := json.Unmarshal(raw, &req); err != nil {
		fail("bad request")
		return
	}
	path, ok := held[req.Cid]
	if !ok {
		fail("not held")
		return
	}
	f, err := os.Open(path)
	if err != nil {
		fail("open failed")
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || req.Offset < 0 || req.Offset > st.Size() {
		fail("bad offset")
		return
	}
	_ = s.SetWriteDeadline(time.Now().Add(idleTimeout))
	hdr, _ := json.Marshal(blockxResp{Size: st.Size()})
	if err := writeFrame(s, hdr); err != nil {
		return
	}
	if req.Offset > 0 {
		if _, err := f.Seek(req.Offset, io.SeekStart); err != nil {
			return
		}
	}
	// stream the body with an idle write deadline reset per chunk (a stalled reader
	// aborts instead of pinning the goroutine + open fd forever — slowloris guard).
	buf := make([]byte, 1<<20)
	var sent int64
	for {
		nr, rerr := f.Read(buf)
		if nr > 0 {
			if err := s.SetWriteDeadline(time.Now().Add(idleTimeout)); err != nil {
				return
			}
			if _, werr := s.Write(buf[:nr]); werr != nil {
				log.Printf("blockx serve %s…: sent %d then %v", req.Cid[:min(16, len(req.Cid))], sent, werr)
				return
			}
			sent += int64(nr)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return
		}
	}
	log.Printf("SERVED %s… bytes[%d:%d] to %s", req.Cid[:min(16, len(req.Cid))], req.Offset, st.Size(), s.Conn().RemotePeer())
}

// runFetchCid is the one-shot fetch: find providers for the CID, pull the bytes from
// the first that serves them, write to `out`. NO verification here — the caller
// (shard/fetch.py) re-hashes against the signed manifest. Returns an error when no
// provider served the full file.
//
// Each peer gets its OWN partial file (out + ".p2p.<peer>.part"): a resume only ever
// appends a peer's own bytes to a peer's own prefix, so a hostile seeder that writes a
// bad prefix can never contaminate an honest peer's transfer (the earlier single shared
// .p2p.part let one hostile node per shard force every honest completion to hash-fail).
// pid in the name also isolates concurrent pulls of the same shard. All partials are
// cleaned on exit, so a failed/hostile transfer leaves nothing to poison a later run.
func runFetchCid(ctx context.Context, h host.Host, d *dht.IpfsDHT, known []peer.AddrInfo,
	cidStr, out string, size int64, timeout time.Duration) error {
	c, err := cid.Decode(cidStr) // the CID contract: reject garbage before touching the DHT
	if err != nil {
		return fmt.Errorf("bad cid: %w", err)
	}
	fctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	prefix := fmt.Sprintf("%s.p2p.%d.", out, os.Getpid())
	defer cleanupPartials(prefix) // no partial survives this call — nothing to resume-poison later

	waitRoutingTable(d, 8*time.Second) // lookups before the table populates return nothing
	// Candidate order: DHT providers first, then the known bootstrap peers DIRECTLY —
	// torrent-style: a peer we were handed beats waiting on record propagation, and a
	// non-holder answers "not held" in one round trip. Dedup by peer id.
	seen := map[peer.ID]bool{h.ID(): true}
	lastErr := fmt.Errorf("no providers for %s", cidStr)
	tried := 0
	attempt := func(ai peer.AddrInfo) (done bool, err error) {
		if seen[ai.ID] {
			return false, nil
		}
		seen[ai.ID] = true
		tried++
		part := prefix + shortPeer(ai.ID) + ".part" // this peer's own partial namespace
		if err := fetchFromPeer(fctx, h, ai, cidStr, part, size); err != nil {
			log.Printf("provider %s: %v", ai.ID, err)
			os.Remove(part) // drop this peer's (possibly hostile) bytes; never reused
			lastErr = err
			return false, nil
		}
		if err := os.Rename(part, out); err != nil {
			return true, err
		}
		fmt.Printf("FETCHED %s -> %s\n", cidStr, out)
		return true, nil
	}
	for ai := range d.FindProvidersAsync(fctx, c, 8) {
		if done, err := attempt(ai); done {
			return err
		}
	}
	for _, ai := range known {
		if done, err := attempt(ai); done {
			return err
		}
	}
	if tried == 0 {
		return lastErr // distinguishable in the log: "no providers" vs a transfer error
	}
	return fmt.Errorf("all %d providers failed, last: %w", tried, lastErr)
}

// cleanupPartials removes every partial file this fetch created (prefix*.part).
func cleanupPartials(prefix string) {
	matches, _ := filepath.Glob(prefix + "*.part")
	for _, m := range matches {
		os.Remove(m)
	}
}

// shortPeer is a filesystem-safe fragment of a peer id (its tail is the high-entropy part).
func shortPeer(p peer.ID) string {
	s := p.String()
	if len(s) > 12 {
		s = s[len(s)-12:]
	}
	return s
}

// fetchFromPeer pulls one shard from ONE peer into `part` — a fresh, peer-owned file, so
// there is no cross-peer prefix to resume (each attempt starts clean). A short/stalled
// body errors and the caller drops `part`. The bulk body streams under an idle deadline.
func fetchFromPeer(ctx context.Context, h host.Host, ai peer.AddrInfo, cidStr, part string, size int64) error {
	os.Remove(part) // this peer's partial is fresh each attempt — no inherited prefix
	cctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	err := h.Connect(cctx, ai)
	cancel()
	if err != nil {
		return err
	}
	s, err := h.NewStream(ctx, ai.ID, blockxProto)
	if err != nil {
		return err
	}
	defer s.Close()
	_ = s.SetWriteDeadline(time.Now().Add(idleTimeout))
	req, _ := json.Marshal(blockxReq{Cid: cidStr, Offset: 0})
	if err := writeFrame(s, req); err != nil {
		return err
	}
	raw, err := readCtrlFrame(s)
	if err != nil {
		return err
	}
	var resp blockxResp
	if err := json.Unmarshal(raw, &resp); err != nil {
		return fmt.Errorf("bad response header: %w", err)
	}
	if resp.Err != "" {
		return fmt.Errorf("peer refused: %s", resp.Err)
	}
	if size > 0 && resp.Size != size {
		return fmt.Errorf("peer size %d != manifest %d", resp.Size, size)
	}
	if resp.Size < 0 {
		return fmt.Errorf("peer declared negative size %d", resp.Size)
	}
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	n, err := copyIdle(s, f, s, resp.Size)
	f.Close()
	if err != nil {
		return fmt.Errorf("transfer stopped at %d/%d: %w", n, resp.Size, err)
	}
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
