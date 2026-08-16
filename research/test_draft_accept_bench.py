"""Offline (NO GPU, NO model, NO torch) validation of draft_accept_bench's load-bearing logic:
the acceptance math (p = matches/n, expected accept-len ~ 1/(1-p)), the per-position argmax compare,
and the vocab-mismatch handling that makes GLM-4-9B (vocab 151552) pay for tokens it can't predict
while GLM-4.7-Flash (matched vocab 154880) does not.

We import the PURE functions from draft_accept_bench (compare_argmax / acceptance_from_counts /
expected_accept_length). They have no torch/GPU dependency — the heavy imports in the module are all
deferred inside the on-box drivers, so importing the module on a CPU-only box is safe. We exercise the
functions on MOCK draft/target argmax id lists, exactly mirroring how run_onbox aggregates per-position
counts across sequences. Mirrors test_coordring_relayback.py: pure stdlib, ~instant, exits nonzero on fail.

Run:  python3 test_draft_accept_bench.py
"""
import os, sys, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_accept_bench import compare_argmax, acceptance_from_counts, expected_accept_length

TGT_VOCAB = 154880     # GLM-5.2 target tokenizer
V_47F = 154880         # GLM-4.7-Flash: matched vocab
V_9B = 151552          # GLM-4-9B: smaller vocab -> OOV penalty on target ids >= 151552

FAIL = []
def check(name, cond, detail=""):
    if cond:
        print(f"  [ok] {name}")
    else:
        FAIL.append(name); print(f"  [FAIL] {name}: {detail}")


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    print("test_draft_accept_bench")

    # --- 1) expected_accept_length: the geometric model 1/(1-p) + edge clamps ---
    check("accept_len p=0", approx(expected_accept_length(0.0), 1.0))
    check("accept_len p=0.5", approx(expected_accept_length(0.5), 2.0))
    check("accept_len p=0.8", approx(expected_accept_length(0.8), 5.0))
    check("accept_len p=0.9", approx(expected_accept_length(0.9), 10.0))
    check("accept_len p=1 -> inf", math.isinf(expected_accept_length(1.0)))

    # --- 2) acceptance_from_counts: matches/n, n==0 guard ---
    check("p = matches/n", approx(acceptance_from_counts(3, 4), 0.75))
    check("p n==0 -> 0", approx(acceptance_from_counts(0, 0), 0.0))

    # --- 3) compare_argmax: exact match, all in-vocab (matched-vocab draft, e.g. 4.7-Flash) ---
    # draft predicts perfectly except position 2; all ids in both vocabs.
    drf = [10, 20, 999, 40]
    tgt = [10, 20, 30, 40]
    c = compare_argmax(drf, tgt, V_47F, TGT_VOCAB)
    check("47F exact-ish: 3/4 match", c["matches"] == 3 and c["n"] == 4, c)
    check("47F no OOV", c["oov_target"] == 0 and c["oob_draft"] == 0, c)

    # --- 4) vocab-mismatch penalty: GLM-4-9B can't represent target ids >= 151552 ---
    # target picks two high-vocab tokens (152000, 154000) the 9B can never emit -> forced non-match,
    # counted in oov_target. The draft "guesses" something in-vocab there; must NOT coincidentally match.
    drf9 = [10, 11, 12, 13, 14]
    tgt9 = [10, 152000, 12, 154000, 14]      # positions 1 and 3 are 9B-OOV target tokens
    c9 = compare_argmax(drf9, tgt9, V_9B, TGT_VOCAB)
    check("9B oov counted", c9["oov_target"] == 2, c9)
    # only positions 0,2,4 are eligible and all match -> 3 matches, oov positions are non-matches
    check("9B matches exclude oov", c9["matches"] == 3 and c9["n"] == 5, c9)
    check("9B p reflects penalty", approx(acceptance_from_counts(c9["matches"], c9["n"]), 0.6), c9)

    # --- 5) the SAME target stream scored by both drafts: matched-vocab draft must win when the target
    #        favors high-vocab tokens (the core thesis). Construct a target with several >=151552 picks
    #        that the matched-vocab draft predicts correctly but the 9B cannot even represent. ---
    tgt_mix = [10, 152500, 11, 153000, 12, 154500]
    drf_match = [10, 152500, 11, 153000, 12, 154500]   # 4.7-Flash nails all (matched vocab)
    drf_small = [10, 99, 11, 99, 12, 99]               # 9B: gets the low-vocab ones, OOV on high ones
    cm = compare_argmax(drf_match, tgt_mix, V_47F, TGT_VOCAB)
    cs = compare_argmax(drf_small, tgt_mix, V_9B, TGT_VOCAB)
    pm = acceptance_from_counts(cm["matches"], cm["n"])
    ps = acceptance_from_counts(cs["matches"], cs["n"])
    check("matched-vocab p == 1.0", approx(pm, 1.0), cm)
    check("small-vocab p == 0.5 (3/6, high-vocab OOV)", approx(ps, 0.5), cs)
    check("matched-vocab draft wins", pm > ps, (pm, ps))
    check("accept_len matched > small", expected_accept_length(pm) > expected_accept_length(ps))

    # --- 6) aggregation across sequences mirrors run_onbox (sum matches + n, then divide ONCE) ---
    seqs_drf = [[1, 2, 3], [4, 5, 6]]
    seqs_tgt = [[1, 9, 3], [4, 5, 9]]      # seq0: 2/3, seq1: 2/3 -> total 4/6
    M = N = 0
    for d, t in zip(seqs_drf, seqs_tgt):
        c = compare_argmax(d, t, V_47F, TGT_VOCAB)
        M += c["matches"]; N += c["n"]
    check("aggregate 4/6", M == 4 and N == 6, (M, N))
    check("aggregate p == 2/3", approx(acceptance_from_counts(M, N), 2.0 / 3.0))

    # --- 7) unequal lengths truncate to the common length (defensive against off-by-one) ---
    c_un = compare_argmax([1, 2, 3, 4], [1, 2, 3], V_47F, TGT_VOCAB)
    check("unequal len truncates to min", c_un["n"] == 3 and c_un["matches"] == 3, c_un)

    if FAIL:
        print(f"\n{len(FAIL)} FAILED: {FAIL}")
        sys.exit(1)
    print("\n[OK] all draft_accept_bench logic checks passed (acceptance math + argmax compare + vocab mismatch)")


if __name__ == "__main__":
    main()
