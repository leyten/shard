"""Offline suffix-tree drafter replay — the no-spend gate for lever A (suffix drafter).

Decides ×1.2-vs-×1.4 (or kill) BEFORE any engine work, per the verified lever stack
(docs/research/m25-lever-stack-verified-20260716.md). Method = SuffixDecoding
(arXiv 2411.04975) replayed over OUR saved greedy M2.5 traces: a suffix drafter only
PROPOSES verbatim continuations and the target greedy-commits the matching prefix, so
acceptance against a fixed greedy trace is EXACT offline — no model, no GPU, no ring.

Faithful to the engine (phase0/m25_pipe.py coordinate_pipe + phase0/eagle_draft.py
HybridDrafter): per round the CPU-half either MATCHES (use its draft; committed =
accepted+1) or misses (EAGLE drafts; we charge the arm's MEASURED committed/round,
g_off from the 07-12 K-tuned receipt — we cannot re-run EAGLE offline). A false match
starves EAGLE for that round; the replay measures exactly that tradeoff.

The honest band (the central bias of any offline hybrid sim): suffix routes on easy
rounds where EAGLE is also above its mean, so we report BOTH
  * multiplier_opt  — EAGLE counterfactual = flat measured mean everywhere;
  * multiplier_pess — EAGLE full-accepts (K_arm+1 committed) on every routed round.
Truth lives between them.

Matchers:
  * LOCAL  — this request's prompt + committed-so-far: 4..1-token anchor table +
             backward longest-match extension, most-recent-longest occurrence wins,
             continuation = verbatim copy (NgramDrafter's mechanics, unified-suffix
             strength). No future leakage by construction.
  * GLOBAL — depth-capped suffix trie over PRIOR requests in the arm (chronological
             accumulation = deployment order), continuation = frequency-greedy chain
             (cross-request scaffolding is where frequency counts matter).
"""

import argparse
import json
from bisect import bisect_left, insort
from collections import Counter, defaultdict


# ---------------------------------------------------------------- global trie

class SuffixTrie:
    """Depth-capped token trie with occurrence counts; built from whole finished
    sequences (never the in-flight request, so lookups cannot see the future)."""

    __slots__ = ("root", "depth")

    def __init__(self, depth=64):
        self.root = {}
        self.depth = depth

    def insert_seq(self, seq):
        n = len(seq)
        d = self.depth
        root = self.root
        for p in range(n):
            node = root
            for i in range(p, min(n, p + d)):
                t = seq[i]
                nxt = node.get(t)
                if nxt is None:
                    nxt = node[t] = {"#": 0}
                nxt["#"] += 1
                node = nxt

    def match(self, ctx, max_pat):
        """Longest suffix of ctx (<= max_pat tokens) that exists as a path.
        Returns (node, match_len); (root, 0) on no match."""
        n = len(ctx)
        for L in range(min(max_pat, n), 0, -1):
            node = self.root
            for i in range(n - L, n):
                node = node.get(ctx[i])
                if node is None:
                    break
            else:
                return node, L
        return self.root, 0

    def continuation(self, node, max_len, min_count):
        out = []
        while len(out) < max_len:
            best_t, best_c = None, min_count - 1
            for t, ch in node.items():
                if t != "#" and ch["#"] > best_c:
                    best_c, best_t = ch["#"], t
            if best_t is None:
                break
            out.append(best_t)
            node = node[best_t]
        return out


# ---------------------------------------------------------------- local matcher

class LocalMatcher:
    """Longest-suffix matcher over the committed sequence only. Anchor = the last
    `ng` tokens (ng tried 4->1); among earlier occurrences of the anchor, extend the
    match backwards and keep the longest (ties -> most recent). Continuation =
    verbatim copy of what followed that occurrence. Indexes positions incrementally;
    candidates are filtered to < t so the future is unreachable by construction."""

    NGS = (4, 3, 2, 1)

    def __init__(self, max_ext=64, max_cand=64):
        self.tables = {ng: defaultdict(list) for ng in self.NGS}
        self.indexed = 0
        self.max_ext = max_ext
        self.max_cand = max_cand

    def extend_index(self, seq):
        """Index anchor occurrences ending at positions (indexed, len(seq)]. An anchor
        keyed at position p means seq[p-ng:p] with continuation starting at p."""
        n = len(seq)
        for ng in self.NGS:
            tbl = self.tables[ng]
            for p in range(max(self.indexed, ng), n + 1):
                tbl[tuple(seq[p - ng:p])].append(p)
        self.indexed = n

    def match(self, seq, max_pat):
        """Longest suffix match against committed text. Returns (start_pos, match_len);
        (None, 0) when no anchor hits. match_len counts the full backward agreement
        (capped at max_pat) so routing thresholds compare with the global trie's."""
        n = len(seq)
        for ng in self.NGS:
            if n < ng:
                continue
            cands = self.tables[ng].get(tuple(seq[n - ng:n]))
            if not cands:
                continue
            hi = bisect_left(cands, n)  # occurrences strictly before the frontier
            if hi == 0:
                continue
            best_p, best_len = None, -1
            for p in cands[max(0, hi - self.max_cand):hi][::-1]:
                L = ng
                cap = min(max_pat, self.max_ext)
                while L < cap and p - L - 1 >= 0 and n - L - 1 >= 0 and seq[p - L - 1] == seq[n - L - 1]:
                    L += 1
                if L > best_len:
                    best_len, best_p = L, p
                    if L >= cap:
                        break
            if best_p is not None:
                return best_p, best_len
        return None, 0


# ---------------------------------------------------------------- replay core

def replay_request(prompt_ids, gen_ids, global_trie, policy, eagle_committed, k_arm):
    """Simulate hybrid rounds over one request's fixed greedy generation."""
    theta = policy["theta_route"]
    spec_max = policy["spec_max"]
    min_count = policy["min_count"]
    max_pat = policy["max_pat"]

    seq = list(prompt_ids)
    local = LocalMatcher(max_ext=max_pat)
    local.extend_index(seq)

    n_gen = len(gen_ids)
    t = 0
    pos_f = 0.0                       # fractional frontier (EAGLE rounds advance by a mean)
    rounds = 0
    routed = 0
    routed_committed = 0
    tokens_routed = 0
    acc_hist = Counter()
    while t < n_gen:
        lp, lL = local.match(seq, max_pat)
        gnode, gL = global_trie.match(seq, max_pat) if global_trie is not None else (None, 0)
        rounds += 1
        use = None
        if max(lL, gL) >= theta:
            if gL > lL:
                prop = global_trie.continuation(gnode, spec_max, min_count)
                use = "g"
            else:
                prop = seq[lp:lp + spec_max]
                use = "l"
            if not prop:
                use = None
        if use is not None:
            acc = 0
            lim = min(len(prop), n_gen - t - 1)
            for j in range(lim):
                if prop[j] == gen_ids[t + j]:
                    acc += 1
                else:
                    break
            adv = acc + 1
            routed += 1
            routed_committed += adv
            tokens_routed += adv
            acc_hist[acc] += 1
            pos_f = t + adv
        else:
            pos_f += eagle_committed
        new_t = min(n_gen, max(t + 1, int(pos_f)))
        seq.extend(gen_ids[t:new_t])
        local.extend_index(seq)
        t = new_t
    return {
        "rounds": rounds,
        "routed": routed,
        "routed_committed": routed_committed,
        "tokens_routed": tokens_routed,
        "n_gen": n_gen,
        "acc_hist": dict(acc_hist),
    }


def replay_arm(requests, policy, eagle_committed, k_arm, cross_request=True):
    """Replay an arm chronologically; the global trie ingests each FINISHED request.
    Returns the aggregate verdict row for this (arm, policy, corpus mode)."""
    global_trie = SuffixTrie(depth=policy["global_depth"]) if cross_request else None
    agg = defaultdict(float)
    acc_hist = Counter()
    for req in requests:
        st = replay_request(req["prompt_ids"], req["gen_ids"], global_trie, policy,
                            eagle_committed, k_arm)
        for k in ("rounds", "routed", "routed_committed", "tokens_routed", "n_gen"):
            agg[k] += st[k]
        acc_hist.update(st["acc_hist"])
        if cross_request:
            global_trie.insert_seq(list(req["prompt_ids"]) + list(req["gen_ids"]))
    rounds = max(agg["rounds"], 1.0)
    routed = agg["routed"]
    n = agg["n_gen"]
    hybrid_cpr = (agg["tokens_routed"] + (rounds - routed) * eagle_committed) / rounds
    # optimistic: pure-EAGLE baseline at its flat measured mean
    mult_opt = hybrid_cpr / eagle_committed
    # pessimistic: EAGLE would have FULL-ACCEPTED (K_arm+1 committed) every routed round
    rounds_hybrid = n / hybrid_cpr if hybrid_cpr > 0 else float("inf")
    rounds_base_pess = agg["tokens_routed"] / (k_arm + 1) + (n - agg["tokens_routed"]) / eagle_committed
    mult_pess = rounds_base_pess / rounds_hybrid if rounds_hybrid else 0.0
    # engine K-bucket view: committed on routed rounds if the frame caps acc at K
    bucket = {}
    for K in (6, 8, 16, 32, 64):
        c = sum(cnt * (min(a, K) + 1) for a, cnt in acc_hist.items())
        r = sum(acc_hist.values())
        bucket[K] = round(c / r, 2) if r else None
    return {
        "policy": {k: v for k, v in policy.items()},
        "cross_request": cross_request,
        "n_requests": len(requests),
        "n_gen_total": int(n),
        "rounds": int(rounds),
        "routed_frac": round(routed / rounds, 4),
        "acc_mean_routed": round((agg["routed_committed"] - routed) / routed, 2) if routed else None,
        "hybrid_committed_per_round": round(hybrid_cpr, 3),
        "eagle_committed_per_round": eagle_committed,
        "multiplier_opt": round(mult_opt, 3),
        "multiplier_pess": round(mult_pess, 3),
        "routed_committed_capped_at_K": bucket,
        "acc_hist_routed": {int(k): int(v) for k, v in sorted(acc_hist.items())},
    }


# ---------------------------------------------------------------- CLI

# EAGLE committed/round baselines: g_off_med per arm at B4 (honest g = committed/round,
# docs/receipts/perstream-trees-ab-20260712.json K6 chain reference) + mix-B1.
EAGLE_G = {
    "tools": 5.0, "code": 3.8, "reasoning": 2.31, "qa": 3.3,
    "prose": 3.52, "summarize": 2.91, "mix": 3.13, "mix-B1": 3.8,
}
K_ARM = 6  # the K-tuned reference chain

DEFAULT_POLICY = dict(theta_route=6, spec_max=64, min_count=2, max_pat=32,
                      global_depth=64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", help="JSONL: {'arm':str,'prompt_ids':[int],'gen_ids':[int]} per line, chronological")
    ap.add_argument("--theta", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--spec-max", type=int, default=64)
    ap.add_argument("--min-count", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default=None, help="write the full verdict JSON here")
    args = ap.parse_args()

    arms = defaultdict(list)
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                arms[r["arm"]].append(r)

    verdict = {}
    for arm, reqs in sorted(arms.items()):
        g = EAGLE_G.get(reqs[0].get("eagle_arm") or arm, EAGLE_G["mix"])
        rows = []
        for theta in args.theta:
            for mc in args.min_count:
                pol = dict(DEFAULT_POLICY, theta_route=theta, min_count=mc,
                           spec_max=args.spec_max)
                for cross in (False, True):
                    rows.append(replay_arm(reqs, pol, g, K_ARM, cross_request=cross))
        verdict[arm] = rows
        best = max(rows, key=lambda r: r["multiplier_pess"])
        print(f"[{arm}] n_req={len(reqs)} gen={best['n_gen_total']} | best-pess: "
              f"theta={best['policy']['theta_route']} mc={best['policy']['min_count']} "
              f"cross={best['cross_request']} routed={best['routed_frac']:.0%} "
              f"acc|routed={best['acc_mean_routed']} "
              f"mult=[{best['multiplier_pess']:.2f} pess, {best['multiplier_opt']:.2f} opt]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
