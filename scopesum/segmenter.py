"""Turn scope-tracker output (or a fixed chunk size) into summary segments.

Pure Python. A *segment* is a set of raw tokens that gets one ``<sum>`` token.
Raw token indices here are indices into the raw (pre-insertion) token list of
one example, without BOS.

* scope mode (variants C/D/E): one segment per scope from
  :mod:`scopesum.scope_tracker`. A segment's raw tokens are the tokens
  *directly* in that scope (tokens of nested scopes belong to the nested
  segment). Nested segments have ``parent`` set.
* fixed mode (variant B): consecutive chunks of ``K`` raw tokens, no nesting.
  The last chunk may be shorter; it gets a ``<sum>`` too (at the end).
* capped scope mode (variant F): scope mode, but a scope whose own tokens
  span more than ``max_span`` raw tokens is cut (at logical-line starts when
  possible) into *parts*. Each part gets its own ``<sum>`` as soon as it ends
  and is a child of the scope's segment, so the scope's ``<sum>`` still sees
  every part. A part's ``<sum>`` fires at most ``max_span`` raw tokens after
  the part's first token, so with ``max_span <= W`` no raw token has to stay
  in the KV cache longer than in variant B. Header tokens count as the
  scope's first tokens, so a long scope's header goes into its first part
  (otherwise it would stay live until the scope's own ``<sum>``). Original
  segments keep their ``seg_id``; parts are appended after them.

Fairness (variant B): :func:`fit_chunk_size` picks ``K`` so that the total
number of fixed-mode ``<sum>`` tokens over a corpus matches the total number
of scope-mode ``<sum>`` tokens as closely as possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .scope_tracker import UNKNOWN_SCOPE, ScopeInfo

KIND_CHUNK = "chunk"
KIND_PART = "part"


@dataclass
class Segment:
    seg_id: int
    kind: str  # module | def | class | chunk
    name: str | None
    qualname: str | None
    parent: int  # parent seg_id, -1 for none
    depth: int
    insert_before: int  # the <sum> goes right before this raw token index (n = at the end)
    n_tokens: int  # raw tokens directly in this segment
    first_token: int  # first raw token index directly in this segment (-1 if none)
    scope_id: int | None = None


@dataclass
class Segmentation:
    token_segment: list[int]  # per raw token; -1 = not in any segment (unknown)
    token_is_header: list[bool]
    segments: list[Segment]
    order: list[int] = field(default_factory=list)  # seg ids in <sum> emission order

    @property
    def n_sums(self) -> int:
        return len(self.order)


def scope_segments(info: ScopeInfo) -> Segmentation:
    """One segment per non-empty scope, in the tracker's closure order."""
    scopes = info.scopes
    # Keep scopes that have tokens, plus ancestors of kept scopes.
    keep = [sc.n_tokens > 0 for sc in scopes]
    for sc in scopes:
        if sc.n_tokens > 0:
            for a in sc.ancestors:
                keep[a] = True

    scope_to_seg: dict[int, int] = {}
    segments: list[Segment] = []
    order: list[int] = []
    for insert_before, sid in info.closures:
        if not keep[sid]:
            continue
        sc = scopes[sid]
        seg_id = len(segments)
        scope_to_seg[sid] = seg_id
        segments.append(
            Segment(
                seg_id=seg_id,
                kind=sc.kind,
                name=sc.name,
                qualname=sc.qualname,
                parent=-1,  # filled below (parents close after children)
                depth=sc.depth,
                insert_before=insert_before,
                n_tokens=sc.n_tokens,
                first_token=sc.open_token,
                scope_id=sid,
            )
        )
        order.append(seg_id)
    for seg in segments:
        p = scopes[seg.scope_id].parent
        seg.parent = scope_to_seg.get(p, -1) if p >= 0 else -1

    token_segment = [scope_to_seg.get(s, -1) if s != UNKNOWN_SCOPE else -1 for s in info.token_scope]
    return Segmentation(
        token_segment=token_segment,
        token_is_header=list(info.token_is_header),
        segments=segments,
        order=order,
    )


def _part_cuts(own: list[int], end: int, max_span: int, line_starts: set[int]) -> list[int]:
    """Cut points (insert-before raw indices) splitting ``own`` tokens into parts.

    ``own`` are the segment's raw token indices, headers included (sorted), ``end`` the
    segment's own ``<sum>`` insertion point. Every part (including the last,
    which stays the segment itself) satisfies ``cut - first_token <= max_span``.
    """
    cuts = []
    i = 0
    while i < len(own) and end - own[i] > max_span:
        start = own[i]
        limit = start + max_span
        line_cuts = [c for c in range(limit, start, -1) if c in line_starts]
        cut = line_cuts[0] if line_cuts else limit
        cuts.append(cut)
        while i < len(own) and own[i] < cut:
            i += 1
    return cuts


def cap_segments(seg: Segmentation, max_span: int, token_line: list[int] | None = None) -> Segmentation:
    """Split long scope segments into parts (variant F); see the module docstring."""
    if max_span <= 0:
        raise ValueError(f"max_span must be positive, got {max_span}")
    n = len(seg.token_segment)
    line_starts = set()
    if token_line is not None:
        line_starts = {t for t in range(1, n) if token_line[t] >= 0 and token_line[t] != token_line[t - 1]}
    own: dict[int, list[int]] = {}
    for t, s in enumerate(seg.token_segment):
        if s >= 0:
            own.setdefault(s, []).append(t)

    segments = [Segment(**vars(s)) for s in seg.segments]
    token_segment = list(seg.token_segment)
    parts_before: dict[int, list[int]] = {}
    for s in seg.order:
        base = seg.segments[s]
        toks = own.get(s, [])
        cuts = _part_cuts(toks, base.insert_before, max_span, line_starts)
        i = 0
        for k, cut in enumerate(cuts):
            pid = len(segments)
            members = []
            while i < len(toks) and toks[i] < cut:
                members.append(toks[i])
                i += 1
            for t in members:
                token_segment[t] = pid
            segments.append(
                Segment(
                    seg_id=pid,
                    kind=KIND_PART,
                    name=None,
                    qualname=f"{base.qualname}#part{k}" if base.qualname else None,
                    parent=s,
                    depth=base.depth + 1,
                    insert_before=cut,
                    n_tokens=len(members),
                    first_token=members[0] if members else -1,
                    scope_id=base.scope_id,
                )
            )
            parts_before.setdefault(s, []).append(pid)
        segments[s].n_tokens = base.n_tokens - i
    order = []
    for s in seg.order:
        order += parts_before.get(s, [])
        order.append(s)
    return Segmentation(
        token_segment=token_segment,
        token_is_header=list(seg.token_is_header),
        segments=segments,
        order=order,
    )


def capped_scope_segments(info: ScopeInfo, max_span: int) -> Segmentation:
    """Scope segments with long scopes split into parts of span <= ``max_span``."""
    return cap_segments(scope_segments(info), max_span, info.token_line)


def fixed_segments(n_tokens: int, chunk_size: int) -> Segmentation:
    """Chunks of ``chunk_size`` raw tokens; the last one may be shorter."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    segments = []
    token_segment = [-1] * n_tokens
    for c, start in enumerate(range(0, n_tokens, chunk_size)):
        end = min(start + chunk_size, n_tokens)
        segments.append(
            Segment(
                seg_id=c,
                kind=KIND_CHUNK,
                name=None,
                qualname=None,
                parent=-1,
                depth=0,
                insert_before=end,
                n_tokens=end - start,
                first_token=start,
            )
        )
        for t in range(start, end):
            token_segment[t] = c
    return Segmentation(
        token_segment=token_segment,
        token_is_header=[False] * n_tokens,
        segments=segments,
        order=list(range(len(segments))),
    )


def n_fixed_sums(n_tokens: int, chunk_size: int) -> int:
    return math.ceil(n_tokens / chunk_size) if n_tokens > 0 else 0


def fit_chunk_size(raw_lengths: list[int], target_total_sums: int) -> int:
    """Smallest K >= 1 whose total fixed-chunk ``<sum>`` count is <= target.

    ``sum(ceil(n_i / K))`` is non-increasing in K, so binary search works.
    Choosing the smallest such K makes B's count as close to the target as
    possible from below (the difference is reported by the data pipeline).
    """
    lengths = [n for n in raw_lengths if n > 0]
    if not lengths:
        raise ValueError("no non-empty examples")
    min_possible = len(lengths)  # K >= max length gives one sum per example
    if target_total_sums <= min_possible:
        return max(lengths)

    def total(k: int) -> int:
        return sum(n_fixed_sums(n, k) for n in lengths)

    lo, hi = 1, max(lengths)
    while lo < hi:
        mid = (lo + hi) // 2
        if total(mid) <= target_total_sums:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fairness_stats(raw_lengths: list[int], scope_sum_counts: list[int]) -> dict:
    """Stats saved by the data pipeline; ``chunk_size`` is variant B's K."""
    if len(raw_lengths) != len(scope_sum_counts):
        raise ValueError("raw_lengths and scope_sum_counts must align")
    target = sum(scope_sum_counts)
    k = fit_chunk_size(raw_lengths, target)
    fixed_total = sum(n_fixed_sums(n, k) for n in raw_lengths if n > 0)
    n_ex = max(len(raw_lengths), 1)
    total_raw = sum(raw_lengths)
    return {
        "n_examples": len(raw_lengths),
        "raw_tokens_total": total_raw,
        "scope_sums_total": target,
        "fixed_sums_total": fixed_total,
        "chunk_size": k,
        "avg_scope_len": total_raw / max(target, 1),
        "avg_scope_sums_per_example": target / n_ex,
        "avg_fixed_sums_per_example": fixed_total / n_ex,
        "relative_count_gap": (fixed_total - target) / max(target, 1),
    }
