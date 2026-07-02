"""Model-free n-gram / prompt-lookup drafter for the long-context spec-decode path.

Why this exists: over WAN, tok/s = g / (hops*RTT), where g = committed tokens per ring
traversal. A draft MODEL is the usual way to raise g, but the gpt-oss-20b draft OOMs at
100k context (its own KV blows the card), so spec-decode dies exactly where long context
needs it most. An n-gram drafter raises g with ZERO model and ZERO KV: it proposes the
continuation that followed the most recent earlier occurrence of the current suffix in the
text so far. At 100k context that text is huge and (for code / documents / structured data,
the latency-tolerant long-context demand) highly self-similar, so g climbs — for free.

Losslessness: this only PROPOSES. The distributed target verifies every proposed token in
the ring and greedy-commits the accepted prefix + one correction, so the output is bit-for-bit
the same greedy decode no matter how good or bad the proposals are. Proposal quality moves g
(speed), never correctness — so we're free to be heuristic here.

Fixed-shape contract: the fast-verify CUDA graph needs a fixed K+1-token chunk, so propose()
ALWAYS returns exactly k tokens (padded if the match runs short, or no match exists — a pad
token just gets rejected and the round still commits the one free verify token, i.e. it
degrades to plain decode, never worse).

Interface mirrors the async draft-socket the pipelined coordinator already speaks
(request -> fetch, one outstanding), so it drops into coordinate_pipe as `local_draft`:
    d.request(ids, k)   # snapshot the conditioning prefix (cheap)
    ds = d.fetch()      # exactly k proposed token ids
"""


class NgramDrafter:
    """Incrementally-indexed prompt-lookup. `table` maps an ng-token suffix -> the latest
    earlier position whose continuation we copy. The index is grown forward as the committed
    sequence grows (O(appended) per round); on a divergence (the coordinator rewinds a short
    speculative tail) it re-indexes only from the divergence point — stale entries pointing
    past the new end are harmless (a bad proposal is just rejected). The whole search is a
    dict lookup, so it stays ~O(1) per round even at 100k context, fully hidden behind the
    WAN verify."""

    def __init__(self, ng=3, max_ext=64, max_cand=48, margin=256, min_match=1):
        self.ng = ng                 # anchor suffix length to index on (the lookup key)
        self.max_ext = max_ext       # cap how far back we extend a match (bounds per-round cost)
        self.max_cand = max_cand     # only weigh the most-recent N occurrences of an anchor
        # `matched` (the HybridDrafter n-gram-vs-EAGLE router) requires best_len >= min_match: a bare
        # ng-token anchor recurrence with ZERO preceding-context agreement is a coincidence on novel/
        # reasoning text, and claiming it starved EAGLE of exactly the rounds it exists for (garbage
        # n-gram draft -> accept 0 -> wasted traversal). min_match=0 restores the old any-anchor routing.
        # The n-gram PROPOSAL itself is unchanged either way — only the routing signal moves.
        self.min_match = min_match
        # only index positions older than `margin` from the end: the recent tail is the pipeline's
        # SPECULATIVE region (in-flight drafts, rewritten on a divergence), so indexing it would
        # create stale entries and force rebuilds. Everything older is committed + immutable, so the
        # index only ever GROWS (no rebuild) and stays O(1)/round after the one-time prompt scan.
        # margin must exceed the speculative lookahead (~2*depth*K); 256 covers depth<=8,K<=16.
        self.margin = margin
        self.indexed = 0             # committed positions < this are in `table`
        self.table = {}              # ng-token anchor -> LIST of continuation-start positions (chronological)
        self._pending = None         # snapshotted (ids, k) between request() and fetch()
        self.matched = False         # did the LAST propose() find a real longest-match? (HybridDrafter routing)

    # ---- async-draft-socket shim ------------------------------------------------
    def request(self, ids, k):
        self._pending = (list(ids), k)          # snapshot: coordinator may mutate its prefix

    def fetch(self):
        ids, k = self._pending
        return self.propose(ids, k)

    def cancel(self):
        self._pending = None                    # drop a stale request without computing the proposal

    # ---- the drafter ------------------------------------------------------------
    def _sync(self, seq):
        """Index newly-committed anchors only — positions in [indexed, len-margin). The recent
        `margin` tail is the speculative region (rewritten on a divergence) so we never index it;
        everything older is immutable. So the table only grows: O(committed prompt) once on the
        first call, then ~K positions/round. No prefix compare, no rebuild — O(1)/round at any ctx."""
        stable = len(seq) - self.margin
        if stable <= self.indexed:
            return
        ng, tbl = self.ng, self.table
        for p in range(max(self.indexed, ng), stable):       # anchor seq[p-ng:p], continuation start p
            tbl.setdefault(tuple(seq[p - ng:p]), []).append(p)
        self.indexed = stable

    def propose(self, seq, k):
        self._sync(seq)
        n = len(seq)
        self.matched = False                     # default: no real match (HybridDrafter -> EAGLE on a miss)
        if n < self.ng:                          # not enough context yet -> plain decode
            return [seq[-1] if seq else 0] * k
        cands = self.table.get(tuple(seq[n - self.ng:n]))
        if not cands:                            # the suffix never occurred in committed text -> plain decode
            return [seq[-1]] * k
        # LONGEST-MATCH: among all earlier occurrences of the anchor, pick the one whose preceding
        # context matches the current position the longest -> at large context this disambiguates
        # which file/region we're truly copying (a generic 2-gram has many homes; a long match has one).
        ng, me = self.ng, self.max_ext
        best_p, best_len = None, -1
        for p in cands[-self.max_cand:][::-1]:   # most recent first (ties -> most recent wins)
            if p >= n:
                continue
            L = 0
            while L < me and p - ng - 1 - L >= 0 and seq[p - ng - 1 - L] == seq[n - ng - 1 - L]:
                L += 1
            if L > best_len:
                best_len, best_p = L, p
                if L == me:
                    break
        if best_p is None:
            return [seq[-1]] * k
        self.matched = best_len >= self.min_match   # real longest-match -> draftable (see min_match)
        cont = seq[best_p:best_p + k]
        if len(cont) < k:                        # ran off the end -> pad (pads get rejected, harmless)
            cont = cont + [cont[-1] if cont else seq[-1]] * (k - len(cont))
        return cont


def simulate_g(seq_ids, prompt_len, ng=3, k=4):
    """Offline upper-bound on g for a fixed greedy output `seq_ids` (prompt + generation):
    walk the generation region simulating greedy n-gram spec-decode and count committed
    tokens per traversal. Faithful because the verify always lands >=1 true token and the
    accepted prefix is exactly what real greedy spec-decode commits. Returns (g, traversals)."""
    d = NgramDrafter(ng=ng, margin=0)            # offline: no speculative tail, index everything
    i = prompt_len
    end = len(seq_ids)
    traversals = 0
    while i < end:
        ds = d.propose(seq_ids[:i], k)
        acc = 0
        for j in range(min(k, end - i - 1)):
            if ds[j] == seq_ids[i + j]:
                acc += 1
            else:
                break
        i += acc + 1                              # accepted drafts + the free verify token
        traversals += 1
    return (end - prompt_len) / max(traversals, 1), traversals
