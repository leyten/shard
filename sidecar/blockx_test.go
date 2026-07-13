package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ipfs/go-cid"
	mh "github.com/multiformats/go-multihash"
)

// testCid computes the CIDv1(raw, sha2-256) of content — byte-identical to
// shard/manifest.py's cidv1_raw().
func testCid(t *testing.T, content []byte) string {
	t.Helper()
	sum := sha256.Sum256(content)
	h, err := mh.Encode(sum[:], mh.SHA2_256)
	if err != nil {
		t.Fatalf("multihash: %v", err)
	}
	return cid.NewCidV1(cid.Raw, h).String()
}

// writeManifest writes a shard/manifest.py-schema manifest naming one shard.
func writeManifest(t *testing.T, dir, shardID, path string, size int64) string {
	t.Helper()
	m := map[string]any{"shards": []map[string]any{
		{"shard_id": shardID, "path": path, "size": size},
	}}
	raw, _ := json.Marshal(m)
	mp := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(mp, raw, 0o644); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	return mp
}

func openRoot(t *testing.T, dir string) *os.Root {
	t.Helper()
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatalf("open root: %v", err)
	}
	t.Cleanup(func() { root.Close() })
	return root
}

// TestManifestShardsRejectsTraversal: a manifest whose path climbs out of the model
// dir must fail the whole scan — nothing from it may be seeded, even when the file
// it points at exists outside the dir at the declared size.
func TestManifestShardsRejectsTraversal(t *testing.T) {
	outer := t.TempDir()
	modelDir := filepath.Join(outer, "model")
	if err := os.Mkdir(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := []byte("host-secret-outside-model-dir")
	if err := os.WriteFile(filepath.Join(outer, "secret.bin"), secret, 0o644); err != nil {
		t.Fatal(err)
	}

	for _, evil := range []string{"../secret.bin", "/etc/passwd", "a/../../secret.bin"} {
		mp := writeManifest(t, t.TempDir(), testCid(t, secret), evil, int64(len(secret)))
		held, err := manifestShards(mp, openRoot(t, modelDir))
		if err == nil {
			t.Fatalf("path %q: want error, got held=%v", evil, held)
		}
	}
}

// TestManifestShardsRejectsBadCid: a shard_id that isn't a CID fails the manifest —
// it could never be provided on the DHT and signals corruption.
func TestManifestShardsRejectsBadCid(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "w.bin"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	mp := writeManifest(t, t.TempDir(), "not-a-cid", "w.bin", 1)
	if _, err := manifestShards(mp, openRoot(t, modelDir)); err == nil {
		t.Fatalf("want error for a non-CID shard_id")
	}
}

// TestManifestShardsSkipsSymlinkEscape: a file that is a symlink pointing outside the
// model dir is never held (the root refuses to resolve it), while an honest sibling
// in the same manifest still seeds.
func TestManifestShardsSkipsSymlinkEscape(t *testing.T) {
	outer := t.TempDir()
	modelDir := filepath.Join(outer, "model")
	if err := os.Mkdir(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := []byte("outside-bytes")
	outside := filepath.Join(outer, "outside.bin")
	if err := os.WriteFile(outside, secret, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(modelDir, "link.bin")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	honest := []byte("honest-weights")
	if err := os.WriteFile(filepath.Join(modelDir, "w.bin"), honest, 0o644); err != nil {
		t.Fatal(err)
	}

	m := map[string]any{"shards": []map[string]any{
		{"shard_id": testCid(t, secret), "path": "link.bin", "size": int64(len(secret))},
		{"shard_id": testCid(t, honest), "path": "w.bin", "size": int64(len(honest))},
	}}
	raw, _ := json.Marshal(m)
	mp := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(mp, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	held, err := manifestShards(mp, openRoot(t, modelDir))
	if err != nil {
		t.Fatalf("manifestShards: %v", err)
	}
	if _, ok := held[testCid(t, secret)]; ok {
		t.Fatalf("escaping symlink was held for seeding")
	}
	if _, ok := held[testCid(t, honest)]; !ok {
		t.Fatalf("honest sibling shard was not held")
	}
}

// --- fetch-side integration: two in-process hosts, seeder <-> fetcher ---

// seedFetch stands up a seeder over modelDir/manifest and a fetcher, optionally
// tampers with the model dir AFTER the seeder scanned it, then runs a one-shot fetch
// of cidStr. Returns the fetch error and the output path.
func seedFetch(t *testing.T, modelDir, manifestPath, cidStr string, size int64, tamper func()) (string, error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	seeder := newTestHost(t)
	fetcher := newTestHost(t)

	dSeed, _, err := setupDHT(ctx, seeder, nil)
	if err != nil {
		t.Fatalf("seed dht: %v", err)
	}
	if err := runSeeder(ctx, seeder, dSeed, manifestPath, modelDir); err != nil {
		t.Fatalf("runSeeder: %v", err)
	}
	if tamper != nil {
		tamper()
	}

	seedMaddr := seeder.Addrs()[0].String() + "/p2p/" + seeder.ID().String()
	dFetch, known, err := setupDHT(ctx, fetcher, []string{seedMaddr})
	if err != nil {
		t.Fatalf("fetch dht: %v", err)
	}
	out := filepath.Join(t.TempDir(), "out.bin")
	return out, runFetchCid(ctx, fetcher, dFetch, known, cidStr, out, size, 30*time.Second)
}

// TestFetchHappyPath: an honest seeder's bytes verify against the CID in Go and land
// at the output path, mode 0600 (unverified-peer bytes stay private).
func TestFetchHappyPath(t *testing.T) {
	content := []byte("the-real-shard-bytes-0123456789")
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "w.bin"), content, 0o644); err != nil {
		t.Fatal(err)
	}
	cs := testCid(t, content)
	mp := writeManifest(t, t.TempDir(), cs, "w.bin", int64(len(content)))

	out, err := seedFetch(t, modelDir, mp, cs, int64(len(content)), nil)
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read out: %v", err)
	}
	if !bytes.Equal(got, content) {
		t.Fatalf("fetched bytes differ")
	}
	st, _ := os.Stat(out)
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("fetched file mode %v, want 0600", st.Mode().Perm())
	}
}

// TestFetchRejectsWrongBytes: a size-complete transfer whose bytes don't hash to the
// CID is rejected INSIDE Go — the fetch fails (with only this provider) instead of
// accepting the poison and leaving Python to discover it after Go gave up.
func TestFetchRejectsWrongBytes(t *testing.T) {
	real := []byte("the-real-shard-bytes-0123456789")
	poison := []byte("EVIL-bytes-same-length-01234567")
	if len(real) != len(poison) {
		t.Fatalf("test setup: sizes must match")
	}
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "w.bin"), poison, 0o644); err != nil {
		t.Fatal(err)
	}
	cs := testCid(t, real) // manifest names the REAL bytes; the seeder holds poison
	mp := writeManifest(t, t.TempDir(), cs, "w.bin", int64(len(real)))

	out, err := seedFetch(t, modelDir, mp, cs, int64(len(real)), nil)
	if err == nil {
		t.Fatalf("size-complete wrong bytes were accepted")
	}
	if !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("want a content-hash error, got: %v", err)
	}
	if _, serr := os.Stat(out); serr == nil {
		t.Fatalf("poison bytes were renamed into place")
	}
}

// TestServeRefusesPostScanSymlinkSwap (TOCTOU): the file passes the scan as a regular
// file, then is swapped for a symlink to an OUTSIDE file with the exact bytes the
// manifest promises. The pre-os.Root seeder followed the link and served it happily;
// the root-confined open must refuse, so the fetch fails instead of leaking a read
// path outside the model dir.
func TestServeRefusesPostScanSymlinkSwap(t *testing.T) {
	content := []byte("the-real-shard-bytes-0123456789")
	outer := t.TempDir()
	modelDir := filepath.Join(outer, "model")
	if err := os.Mkdir(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	inside := filepath.Join(modelDir, "w.bin")
	if err := os.WriteFile(inside, content, 0o644); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(outer, "outside.bin")
	if err := os.WriteFile(outside, content, 0o644); err != nil {
		t.Fatal(err)
	}
	cs := testCid(t, content)
	mp := writeManifest(t, t.TempDir(), cs, "w.bin", int64(len(content)))

	_, err := seedFetch(t, modelDir, mp, cs, int64(len(content)), func() {
		if err := os.Remove(inside); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, inside); err != nil {
			t.Skipf("symlink: %v", err)
		}
	})
	if err == nil {
		t.Fatalf("seeder served through an escaping symlink swapped in after the scan")
	}
}
