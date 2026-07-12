# Tree-frame CUDA graphs — the eager-tax lever (designed 2026-07-12, build next)

## Why (measured, perstream-trees ring 2026-07-12)
Per-stream trees under de-lockstep raised g +15-70% on every arm (reasoning 2.31→3.92,
mix 3.13→4.27, receipts valid) but per-stream tok/s LOST on most content: the round
decomposition (M25_STAGE_TIMING, reasoning-B4) shows the tree frame's summed stage compute at
**154ms vs 45ms for chain frames — 3.4×, pure eager tax** (chain rows replay RowGraphRunner
graphs; run_block_tree_row runs 62 layers eager at N≈13-21). Round 328ms vs 169ms. If tree
frames replayed at chain-like stage cost (~50-70ms), tree rounds ≈ 200-240ms → the g lift nets
+20-40% per stream instead of −13%. The lever is mechanical, not physics.

## Design sketch (the _RGraphState pattern, padded)
- Capture `run_block_tree_row` per (Npad, bucket): pad every tree frame to a FIXED Npad
  (e.g. 16 or 24 ≥ trunk+M) with dummy nodes; static buffers refreshed per replay: pos_ids [Npad],
  additive mask [1,1,Npad,alen], row index, KV write indices.
- Mask: dummy ROWS fully masked (attend nothing but themselves → output garbage, ignored);
  dummy COLUMNS -inf for every real row (real nodes never attend padding). The tail's argmax
  reads only the first N_real rows (coordinator knows N_real; tail needs it — send "n" on the
  frame or slice by len(token_ids) BEFORE padding server-side; prefer stage-side padding so the
  wire and coordinator stay unchanged).
- ⚠️ THE KV HAZARD (found at design time — this is the silent-corruption class): dummy nodes
  MUST NOT write k/v into live row slots. A write at col c is readable by ANY later frame whose
  context covers c (chain masks by cp ≤, but col c < total passes). Fix: point the dummy rows'
  static write indices at a SACRIFICIAL slot (reserve col M25_KV_MAXLEN-1; bounds-check all real
  frames to total ≤ MAXLEN-1 so the trash column is never attended). The index tensors are
  static-refreshed per replay, so real nodes' indices stay exact.
- Mask/pos/index refresh per replay = _RGraphState.set() extended; capture cost ~1 graph per
  (Npad, bucket) — Npad fixed makes it ONE shape; falls under M25_GRAPH_MAX + the free-VRAM
  capture guard like every runner.
- Hatch: M25_TREE_GRAPH=0. Gates: graph-vs-eager bit-equality on the tree kernel (the
  graph_aux_check pattern), fake-ring rows+tree equivalence unchanged (stage-side change only),
  the trash-slot invariant unit-tested (no dummy write below MAXLEN-1; real writes exact), THE
  no-poison test: run a tree frame then chain frames over the same row and assert attention
  output equals the never-padded eager reference.
- Also worth folding in: aux slimming for tree frames (return-leg aux rows = accepted path only
  — today the full [N,H]×3 rides back; _aux_keep_lens's tree analog from tree_greedy_walk is
  coordinator-known only post-walk, so slim to the frame's node count ceiling or skip).

## Bar to re-check after the build (same ring shape)
reasoning-B4 ≥ 16-19/stream (from 11.7), mix-B4 tree-on ≥ 16 (from 11.3), prose ≥ 20 with
content routing (K=5) + trees. Kill stays: tree-on < +10% vs K-tuned chains per stream.
