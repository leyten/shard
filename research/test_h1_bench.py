"""Offline (NO network, NO GPU) validation of h1_bench's compare/aggregation — the load-bearing logic
that turns dumped target + draft argmaxes into the per-context acceptance table. Mock dumps only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_bench import compare_dumps

ok = 0
def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    print(f"  [ok] {name}"); ok += 1

TGT_VOCAB = 100

# target dump: two context labels, two seqs each
target = {
    ("code_1024", 0): [10, 20, 30, 40],
    ("code_1024", 1): [11, 21, 31],
    ("code_8192", 0): [50, 60, 70, 80, 95],   # 95 will be OOV for a small-vocab draft
}

# MTP-ish draft: matches target everywhere (perfect) -> p=1.0
mtp = {k: list(v) for k, v in target.items()}
# 9B-ish draft: smaller vocab (90), diverges on some + a target id (95) it can't represent
nineb = {
    ("code_1024", 0): [10, 99, 30, 40],       # pos1 differs -> 3/4
    ("code_1024", 1): [11, 21, 31],           # 3/3
    ("code_8192", 0): [50, 60, 70, 80, 88],   # target 95 >= 90 -> OOV (forced miss); others match -> 4/5
}

recs = compare_dumps(target, {"MTP": (mtp, TGT_VOCAB), "9B": (nineb, 90)}, TGT_VOCAB)
by = {(r["draft"], r["context"]): r for r in recs}

check("MTP code_1024 perfect p=1.0", by[("MTP", "code_1024")]["p_accept"] == 1.0)
check("MTP code_1024 n=7 (4+3)", by[("MTP", "code_1024")]["n_positions"] == 7)
check("MTP code_8192 perfect p=1.0", by[("MTP", "code_8192")]["p_accept"] == 1.0)
check("MTP no OOV", by[("MTP", "code_8192")]["oov_target"] == 0)

# 9B code_1024: 3/4 + 3/3 = 6/7
r = by[("9B", "code_1024")]
check("9B code_1024 matches=6", r["matches"] == 6)
check("9B code_1024 n=7", r["n_positions"] == 7)
check("9B code_1024 p=6/7", abs(r["p_accept"] - 6/7) < 1e-4)
check("9B code_1024 no OOV (all target ids < 90)", r["oov_target"] == 0)

# 9B code_8192: target 95 OOV (forced miss), 50/60/70/80 match -> 4/5, oov=1
r = by[("9B", "code_8192")]
check("9B code_8192 matches=4", r["matches"] == 4)
check("9B code_8192 oov_target=1", r["oov_target"] == 1)
check("9B code_8192 p=4/5", abs(r["p_accept"] - 0.8) < 1e-9)
check("9B code_8192 draft_vocab recorded", r["draft_vocab"] == 90)

# accept_len monotonic with p (sanity): MTP(p=1) accept_len > 9B accept_len at code_1024
check("accept_len: MTP > 9B at code_1024",
      by[("MTP", "code_1024")]["expected_accept_len"] > by[("9B", "code_1024")]["expected_accept_len"])

# unmatched seqs (target missing) are skipped, not errored
recs2 = compare_dumps(target, {"D": ({("nope", 9): [1, 2, 3]}, TGT_VOCAB)}, TGT_VOCAB)
check("missing-target seq skipped -> no records", recs2 == [])

print(f"\n[OK] all {ok} h1_bench compare/aggregation checks passed (split-dump comparison + OOV + per-context)")
