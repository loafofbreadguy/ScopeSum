"""Hand-drawn expected masks for tiny sequences.

Notation in the comments: B = BOS, rN = raw token N, S<seg> = <sum> of a
segment, u = suffix token, P = pad. Rows are queries, columns keys, "1" =
may attend.
"""

import numpy as np
import pytest

from scopesum.masks import (
    MaskMeta,
    build_sliding_mask,
    causal_mask,
    full_causal_mask,
    kv_live_counts,
    mask_baseline,
)
from scopesum.segmenter import Segment, Segmentation, fixed_segments
from scopesum.sum_insertion import insert_sums

BOS, SUM, PAD = 2, 6, 0


def grid(rows: str) -> np.ndarray:
    lines = [ln.split("#")[0].strip() for ln in rows.strip().splitlines()]
    return np.array([[c == "1" for c in ln.replace(" ", "")] for ln in lines if ln], dtype=bool)


def meta_of(ex):
    return MaskMeta.from_example(ex)


# --- sequence 1: flat, fixed chunks K=2, 5 raw tokens ------------------------
# positions: 0 B, 1 r0, 2 r1, 3 S0, 4 r2, 5 r3, 6 S1, 7 r4, 8 S2


def seq1():
    return insert_sums([10, 11, 12, 13, 14], fixed_segments(5, 2), BOS, SUM)


SEQ1_B_W2 = grid(
    """
    1 0 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0 0   # r1
    1 1 1 1 0 0 0 0 0   # S0: own raw r0 r1
    1 0 1 1 1 0 0 0 0   # r2: last 2 raw (r1 r2) + S0
    1 0 0 1 1 1 0 0 0   # r3: r2 r3 + S0
    1 0 0 0 1 1 1 0 0   # S1: own raw r2 r3
    1 0 0 1 0 1 1 1 0   # r4: r3 r4 + S0 S1
    1 0 0 0 0 0 0 1 1   # S2: own raw r4
    """
)


# --- sequence 2: nested scopes ----------------------------------------------
# raw r0 r1 (outer, r0 header), r2 r3 (inner, r2 header), r4 (outer again)
# positions: 0 B, 1 r0, 2 r1, 3 r2, 4 r3, 5 S_inner, 6 r4, 7 S_outer


def seq2():
    segs = [
        Segment(0, "def", "outer", "outer", -1, 0, insert_before=5, n_tokens=3, first_token=0),
        Segment(1, "def", "inner", "outer.inner", 0, 1, insert_before=4, n_tokens=2, first_token=2),
    ]
    seg = Segmentation([0, 0, 1, 1, 0], [True, False, True, False, False], segs, order=[1, 0])
    return insert_sums([10, 11, 12, 13, 14], seg, BOS, SUM)


SEQ2_C_W2 = grid(
    """
    1 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0   # r1
    1 0 1 1 0 0 0 0   # r2: r1 r2
    1 0 0 1 1 0 0 0   # r3: r2 r3
    1 0 0 1 1 1 0 0   # S_inner: own raw r2 r3
    1 0 0 0 1 1 1 0   # r4: r3 r4 + S_inner
    1 1 1 0 0 1 1 1   # S_outer: own raw r0 r1 r4 + child S_inner
    """
)

SEQ2_D_W1 = grid(
    """
    1 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0   # r0 (outer)
    1 1 1 0 0 0 0 0   # r1 (outer): own scope r0 r1
    1 1 0 1 0 0 0 0   # r2 (inner): own scope r2 + enclosing header r0
    1 1 0 1 1 0 0 0   # r3 (inner): r2 r3 + header r0
    1 0 0 1 1 1 0 0   # S_inner
    1 1 1 0 0 1 1 0   # r4 (outer): own scope r0 r1 r4 + S_inner
    1 1 1 0 0 1 1 1   # S_outer
    """
)


# --- sequence 3: module run + two sibling functions -------------------------
# raw r0 (module), r1 r2 (f, r1 header), r3 r4 (g, r3 header)
# positions: 0 B, 1 r0, 2 S_mod, 3 r1, 4 r2, 5 S_f, 6 r3, 7 r4, 8 S_g


def seq3(edges=None):
    segs = [
        Segment(0, "module", None, "<module>", -1, 0, insert_before=1, n_tokens=1, first_token=0),
        Segment(1, "def", "f", "f", -1, 0, insert_before=3, n_tokens=2, first_token=1),
        Segment(2, "def", "g", "g", -1, 0, insert_before=5, n_tokens=2, first_token=3),
    ]
    seg = Segmentation([0, 1, 1, 2, 2], [False, True, False, True, False], segs, order=[0, 1, 2])
    return insert_sums([10, 11, 12, 13, 14], seg, BOS, SUM, seg_edges=edges)


SEQ3_C_W2 = grid(
    """
    1 0 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0 0   # S_mod
    1 1 1 1 0 0 0 0 0   # r1: r0 r1 + S_mod
    1 0 1 1 1 0 0 0 0   # r2: r1 r2 + S_mod
    1 0 0 1 1 1 0 0 0   # S_f
    1 0 1 0 1 1 1 0 0   # r3: r2 r3 + S_mod S_f
    1 0 1 0 0 1 1 1 0   # r4: r3 r4 + S_mod S_f
    1 0 0 0 0 0 1 1 1   # S_g
    """
)

SEQ3_D_W1 = grid(
    """
    1 0 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0 0   # S_mod
    1 0 1 1 0 0 0 0 0   # r1: own scope r1 + S_mod
    1 0 1 1 1 0 0 0 0   # r2: r1 r2 + S_mod
    1 0 0 1 1 1 0 0 0   # S_f
    1 0 1 0 0 1 1 0 0   # r3: r3 + S_mod S_f
    1 0 1 0 0 1 1 1 0   # r4: r3 r4 + S_mod S_f
    1 0 0 0 0 0 1 1 1   # S_g
    """
)


# --- sequence 4: fixed chunks + unscoped suffix token + padding -------------
# raw r0 r1 r2 (K=2), suffix u, one pad
# positions: 0 B, 1 r0, 2 r1, 3 S0, 4 r2, 5 S1, 6 u, 7 P


def seq4():
    return insert_sums([10, 11, 12], fixed_segments(3, 2), BOS, SUM, pad_id=PAD, pad_to=8, suffix_ids=[99])


SEQ4_B_W2 = grid(
    """
    1 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0   # r1
    1 1 1 1 0 0 0 0   # S0
    1 0 1 1 1 0 0 0   # r2: r1 r2 + S0
    1 0 0 0 1 1 0 0   # S1: own raw r2
    1 0 0 1 1 1 1 0   # u: r2 u + S0 S1
    0 0 0 0 0 0 0 1   # P: only itself
    """
)

SEQ4_D_W1 = grid(
    """
    1 0 0 0 0 0 0 0   # B
    1 1 0 0 0 0 0 0   # r0
    1 1 1 0 0 0 0 0   # r1: same chunk r0 r1
    1 1 1 1 0 0 0 0   # S0
    1 0 0 1 1 0 0 0   # r2: own chunk r2 + S0
    1 0 0 0 1 1 0 0   # S1
    1 0 0 1 0 1 1 0   # u (no scope): itself (w_small=1) + S0 S1
    0 0 0 0 0 0 0 1   # P
    """
)


@pytest.mark.parametrize(
    "make, variant, window, w_small, expected",
    [
        (seq1, "B", 2, 1, SEQ1_B_W2),
        (seq1, "C", 2, 1, SEQ1_B_W2),  # same rule as B; only segments differ
        (seq2, "C", 2, 1, SEQ2_C_W2),
        (seq2, "D", 2, 1, SEQ2_D_W1),
        (seq3, "C", 2, 1, SEQ3_C_W2),
        (seq3, "D", 2, 1, SEQ3_D_W1),
        (seq4, "B", 2, 1, SEQ4_B_W2),
        (seq4, "D", 2, 1, SEQ4_D_W1),
    ],
)
def test_hand_drawn_masks(make, variant, window, w_small, expected):
    mask = build_sliding_mask(variant, meta_of(make()), window=window, w_small=w_small)
    assert mask.shape == expected.shape
    diff = np.argwhere(mask != expected)
    assert diff.size == 0, f"mismatch at (q, k): {diff.tolist()}"


def test_variant_e_adds_callee_edges():
    # g calls f: S_g (pos 8) may also attend S_f (pos 5)
    mask = build_sliding_mask("E", meta_of(seq3(edges=[(2, 1)])), window=2, w_small=1)
    expected = SEQ3_D_W1.copy()
    expected[8, 5] = True
    assert (mask == expected).all()
    # an edge pointing to a later <sum> is ignored by causality
    mask2 = build_sliding_mask("E", meta_of(seq3(edges=[(1, 2)])), window=2, w_small=1)
    assert (mask2 == SEQ3_D_W1).all()


def test_baseline_is_position_sliding_window():
    ex = insert_sums([10, 11, 12, 13], None, BOS, SUM)
    mask = mask_baseline(meta_of(ex), window=2)
    expected = grid(
        """
        1 0 0 0 0
        1 1 0 0 0
        0 1 1 0 0
        0 0 1 1 0
        0 0 0 1 1
        """
    )
    assert (mask == expected).all()
    assert (build_sliding_mask("A", meta_of(ex), window=2) == expected).all()


@pytest.mark.parametrize("variant", ["A", "B", "C", "D", "E"])
@pytest.mark.parametrize("make", [seq1, seq2, seq3, seq4])
def test_generic_properties(variant, make):
    meta = meta_of(make())
    mask = build_sliding_mask(variant, meta, window=2, w_small=1)
    L = meta.length
    assert not (mask & ~causal_mask(L)).any(), "never attend to the future"
    assert mask.diagonal().all(), "every position sees itself"
    pads = np.where(meta.kind == 3)[0]
    for p in pads:
        assert mask[:, p].sum() == 1, "nobody else sees padding"


def test_full_mask_is_causal_without_padding():
    meta = meta_of(seq4())
    full = full_causal_mask(meta)
    assert (full[:7, :7] == causal_mask(7)).all()
    assert full[7].tolist() == [False] * 7 + [True]
    assert not full[:7, 7].any()


def test_kv_live_counts():
    ex = insert_sums([10, 11, 12], None, BOS, SUM)  # L = 4
    live = kv_live_counts(mask_baseline(meta_of(ex), window=2))
    assert live.tolist() == [1, 2, 2, 2]
    # dense causal: everything stays alive
    live_full = kv_live_counts(causal_mask(4))
    assert live_full.tolist() == [1, 2, 3, 4]


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_sliding_mask("Z", meta_of(seq1()), window=2)
