"""Boolean attention masks for every variant (pure NumPy, no torch).

All functions return ``mask[q, k]`` of shape ``[L, L]`` with ``True`` meaning
"query position q may attend to key position k". Every mask is causal
(``k <= q``). Position kinds come from :mod:`scopesum.sum_insertion`.

Rules (local / sliding layers; global layers stay plain causal unless
``variant.restrict_full`` gives them the same mask, see :mod:`scopesum.model`):

=====  =====================================================================
A      model default: causal sliding window over positions (``k > q - W``),
       matching ``transformers.masking_utils.sliding_window_overlay``.
B,C,F,G raw query: BOS + last ``W`` raw tokens (counted in raw tokens, itself
       included) + every earlier ``<sum>``.
D, E   raw query: BOS + raw tokens of its own scope + header tokens of
       enclosing scopes + last ``w_small`` raw tokens + every earlier ``<sum>``.
=====  =====================================================================

``<sum>`` query (B-E): BOS + itself + raw tokens of its own segment + ``<sum>``
of directly nested segments (all nested descendants if
``all_descendants=True``) + for E the ``<sum>`` of callee segments that are
already emitted. BOS sees only itself. PAD sees only itself, and nothing ever
sees a PAD (keeps softmax rows finite without leaking padding).

B, C, F and G share one function: they differ only in where segments fall
(F = C with long scopes split into parts, see :mod:`scopesum.segmenter`;
G = B with ``K`` refitted to F's ``<sum>`` count).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sum_insertion import KIND_BOS, KIND_PAD, KIND_RAW, KIND_SUM, Example

VARIANTS = ("A", "B", "C", "D", "E", "F", "G", "S")

# SepLLM's separators: punctuation and whitespace tokens (Chen et al., 2024)
SEPARATOR_CHARS = (".", ",", "?", "!", ";", ":")


@dataclass
class MaskMeta:
    """The arrays the mask builders need (a view of an :class:`Example`)."""

    kind: np.ndarray  # int [L]
    segment: np.ndarray  # int [L]
    is_header: np.ndarray  # bool [L]
    seg_parent: np.ndarray  # int [S]
    seg_edges: list  # [(caller, callee)]
    is_separator: np.ndarray | None = None  # bool [L], variant S only

    @classmethod
    def from_example(cls, ex: Example) -> "MaskMeta":
        return cls(
            kind=np.asarray(ex.kind, dtype=np.int64),
            segment=np.asarray(ex.segment, dtype=np.int64),
            is_header=np.asarray(ex.is_header, dtype=bool),
            seg_parent=np.asarray(ex.seg_parent, dtype=np.int64),
            seg_edges=list(ex.seg_edges),
        )

    @property
    def length(self) -> int:
        return int(self.kind.shape[0])

    @property
    def n_segments(self) -> int:
        return int(self.seg_parent.shape[0])


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def causal_mask(length: int) -> np.ndarray:
    return np.tril(np.ones((length, length), dtype=bool))


def _raw_rank(kind: np.ndarray) -> np.ndarray:
    """Index of the last raw token at or before each position (-1 if none).

    For a RAW position this is its index among raw tokens.
    """
    return np.cumsum(kind == KIND_RAW) - 1


def _seg_relation(rel: np.ndarray, q_seg: np.ndarray, k_seg: np.ndarray) -> np.ndarray:
    """Look up ``rel[q_seg[q], k_seg[k]]`` with ``-1`` segments mapping to False."""
    s = rel.shape[0]
    padded = np.zeros((s + 1, s + 1), dtype=bool)
    padded[:s, :s] = rel
    qi = np.where(q_seg >= 0, q_seg, s)
    ki = np.where(k_seg >= 0, k_seg, s)
    return padded[qi[:, None], ki[None, :]]


def parent_matrix(seg_parent: np.ndarray) -> np.ndarray:
    """``P[g, h]`` = segment h is a direct child of segment g."""
    s = seg_parent.shape[0]
    rel = np.zeros((s, s), dtype=bool)
    for h, g in enumerate(seg_parent):
        if g >= 0:
            rel[g, h] = True
    return rel


def ancestor_matrix(seg_parent: np.ndarray) -> np.ndarray:
    """``A[g, h]`` = segment h is a strict ancestor of segment g."""
    s = seg_parent.shape[0]
    rel = np.zeros((s, s), dtype=bool)
    for g in range(s):
        h = seg_parent[g]
        seen = 0
        while h >= 0 and seen <= s:  # ``seen`` guards against malformed cycles
            rel[g, h] = True
            h = seg_parent[h]
            seen += 1
    return rel


def descendant_matrix(seg_parent: np.ndarray) -> np.ndarray:
    """``D[g, h]`` = segment h is a strict descendant of segment g."""
    return ancestor_matrix(seg_parent).T.copy()


def edge_matrix(n_segments: int, edges) -> np.ndarray:
    rel = np.zeros((n_segments, n_segments), dtype=bool)
    for a, b in edges:
        if 0 <= a < n_segments and 0 <= b < n_segments and a != b:
            rel[a, b] = True
    return rel


def _finalize(meta: MaskMeta, raw_rows: np.ndarray, sum_rows: np.ndarray | None) -> np.ndarray:
    """Combine per-kind row rules, enforce causality and BOS/PAD rules."""
    L = meta.length
    kind = meta.kind
    eye = np.eye(L, dtype=bool)
    mask = np.zeros((L, L), dtype=bool)
    is_raw = kind == KIND_RAW
    mask[is_raw] = raw_rows[is_raw]
    if sum_rows is not None:
        is_sum = kind == KIND_SUM
        mask[is_sum] = sum_rows[is_sum]
    mask &= causal_mask(L)
    mask &= (kind != KIND_PAD)[None, :]
    mask |= eye  # every position sees itself (BOS, PAD, and safety for others)
    return mask


def _sum_rows(meta: MaskMeta, all_descendants: bool, edges: bool) -> np.ndarray:
    kind, seg = meta.kind, meta.segment
    k_is_raw = (kind == KIND_RAW)[None, :]
    k_is_sum = (kind == KIND_SUM)[None, :]
    bos_col = (kind == KIND_BOS)[None, :]
    own_raw = k_is_raw & (seg[None, :] == seg[:, None]) & (seg[:, None] >= 0)
    nest = descendant_matrix(meta.seg_parent) if all_descendants else parent_matrix(meta.seg_parent)
    nested_sums = k_is_sum & _seg_relation(nest, seg, seg)
    rows = bos_col | own_raw | nested_sums
    if edges and meta.seg_edges:
        callee = edge_matrix(meta.n_segments, meta.seg_edges)
        rows |= k_is_sum & _seg_relation(callee, seg, seg)
    return rows


def _recent_raw(kind: np.ndarray, window: int) -> np.ndarray:
    """Raw keys among the last ``window`` raw tokens (query itself included)."""
    rank = _raw_rank(kind)
    k_is_raw = (kind == KIND_RAW)[None, :]
    diff = rank[:, None] - rank[None, :]
    return k_is_raw & (diff >= 0) & (diff < window)


# ---------------------------------------------------------------------------
# one function per variant rule
# ---------------------------------------------------------------------------


def mask_baseline(meta: MaskMeta, window: int) -> np.ndarray:
    """Variant A: the model's default sliding-window causal mask (by position)."""
    L = meta.length
    q = np.arange(L)[:, None]
    k = np.arange(L)[None, :]
    rows = k > q - window
    mask = rows & causal_mask(L)
    mask &= (meta.kind != KIND_PAD)[None, :]
    mask |= np.eye(L, dtype=bool)
    return mask


def mask_window_plus_sums(meta: MaskMeta, window: int, all_descendants: bool = False, keep_headers: bool = False) -> np.ndarray:
    """Variants B, C, F and G: BOS + last ``window`` raw tokens + all earlier sums.

    ``keep_headers``: raw queries also see every earlier header token (def/class
    lines and decorators), so exact signatures stay reachable past the window.
    """
    kind = meta.kind
    bos_col = (kind == KIND_BOS)[None, :]
    k_is_sum = (kind == KIND_SUM)[None, :]
    raw_rows = bos_col | _recent_raw(kind, window) | k_is_sum
    if keep_headers:
        raw_rows = raw_rows | ((kind == KIND_RAW) & meta.is_header)[None, :]
    return _finalize(meta, raw_rows, _sum_rows(meta, all_descendants, edges=False))


def mask_separators(meta: MaskMeta, window: int) -> np.ndarray:
    """Variant S (SepLLM-style): BOS + last ``window`` raw tokens + every earlier separator token."""
    kind = meta.kind
    if meta.is_separator is None:
        raise ValueError("variant S needs MaskMeta.is_separator")
    bos_col = (kind == KIND_BOS)[None, :]
    seps = ((kind == KIND_RAW) & meta.is_separator)[None, :]
    raw_rows = bos_col | _recent_raw(kind, window) | seps
    return _finalize(meta, raw_rows, None)


def separator_token_ids(tokenizer) -> np.ndarray:
    """Ids of tokens that are only whitespace or one SepLLM punctuation mark."""
    ids = []
    for i, t in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if t is None or (t.startswith("<") and t.endswith(">")):
            continue  # special and byte-fallback tokens
        s = t.replace("\u2581", " ").strip()
        if s == "" or s in SEPARATOR_CHARS:
            ids.append(i)
    return np.asarray(ids, dtype=np.int64)


def mask_scope_window(
    meta: MaskMeta, w_small: int, all_descendants: bool = False, edges: bool = False
) -> np.ndarray:
    """Variant D (and E with ``edges=True``)."""
    kind, seg = meta.kind, meta.segment
    k_is_raw = (kind == KIND_RAW)[None, :]
    k_is_sum = (kind == KIND_SUM)[None, :]
    bos_col = (kind == KIND_BOS)[None, :]
    same_scope = k_is_raw & (seg[None, :] == seg[:, None]) & (seg[:, None] >= 0)
    anc = _seg_relation(ancestor_matrix(meta.seg_parent), seg, seg)
    enclosing_headers = k_is_raw & meta.is_header[None, :] & anc
    raw_rows = bos_col | same_scope | enclosing_headers | _recent_raw(kind, w_small) | k_is_sum
    return _finalize(meta, raw_rows, _sum_rows(meta, all_descendants, edges=edges))


def mask_graph(meta: MaskMeta, w_small: int, all_descendants: bool = False) -> np.ndarray:
    """Variant E: D plus <sum> -> callee <sum> edges."""
    return mask_scope_window(meta, w_small, all_descendants=all_descendants, edges=True)


def full_causal_mask(meta: MaskMeta) -> np.ndarray:
    """Global layers: plain causal mask (padding hidden), for every variant."""
    L = meta.length
    is_pad = meta.kind == KIND_PAD
    mask = causal_mask(L) & ~is_pad[None, :]
    mask[is_pad] = False  # PAD rows see only themselves, as in the sliding masks
    mask |= np.eye(L, dtype=bool)
    return mask


def build_sliding_mask(
    variant: str,
    meta: MaskMeta,
    window: int,
    w_small: int = 64,
    all_descendants: bool = False,
    keep_headers: bool = False,
) -> np.ndarray:
    """Dispatch on the variant id."""
    if variant == "A":
        return mask_baseline(meta, window)
    if variant in ("B", "C", "F", "G"):
        return mask_window_plus_sums(meta, window, all_descendants, keep_headers)
    if variant == "D":
        return mask_scope_window(meta, w_small, all_descendants)
    if variant == "E":
        return mask_graph(meta, w_small, all_descendants)
    if variant == "S":
        return mask_separators(meta, window)
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def kv_live_counts(mask: np.ndarray) -> np.ndarray:
    """KV entries a streaming implementation must hold at each step.

    Key ``j`` must be kept from step ``j`` until the last query that uses it.
    Returns ``live[t]`` = number of keys ``j <= t`` with some query ``q >= t``
    attending to ``j``. Used by the analytic memory evaluation.
    """
    L = mask.shape[0]
    used = mask.any(axis=0)
    # last query index that attends to each key (keys nobody uses: -1)
    last_q = np.where(used, L - 1 - np.argmax(mask[::-1, :], axis=0), -1)
    j = np.arange(L)
    t = np.arange(L)[:, None]
    alive = (j[None, :] <= t) & (last_q[None, :] >= t)
    return alive.sum(axis=1)
