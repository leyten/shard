package main

import (
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

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
