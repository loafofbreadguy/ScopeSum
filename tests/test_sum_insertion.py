from scopesum.segmenter import Segment, Segmentation, fixed_segments
from scopesum.sum_insertion import (
    IGNORE_INDEX,
    KIND_BOS,
    KIND_PAD,
    KIND_RAW,
    KIND_SUM,
    Example,
    insert_sums,
)

BOS, SUM, PAD = 2, 6, 0


def nested_segmentation():
    """raw r0 r1 (outer), r2 r3 (inner), r4 (outer). Inner sum before r4, outer at end."""
    segs = [
        Segment(0, "def", "outer", "outer", -1, 0, insert_before=5, n_tokens=3, first_token=0),
        Segment(1, "def", "inner", "outer.inner", 0, 1, insert_before=4, n_tokens=2, first_token=2),
    ]
    return Segmentation(
        token_segment=[0, 0, 1, 1, 0],
        token_is_header=[True, False, True, False, False],
        segments=segs,
        order=[1, 0],
    )


def test_no_segmentation_is_plain_bos_plus_raw():
    ex = insert_sums([10, 11, 12], None, BOS, SUM)
    assert ex.input_ids == [BOS, 10, 11, 12]
    assert ex.labels == [IGNORE_INDEX, 10, 11, 12]
    assert ex.kind == [KIND_BOS, KIND_RAW, KIND_RAW, KIND_RAW]
    assert ex.seg_parent == [] and ex.sum_position == []


def test_fixed_chunks():
    ex = insert_sums([10, 11, 12, 13, 14], fixed_segments(5, 2), BOS, SUM)
    assert ex.input_ids == [BOS, 10, 11, SUM, 12, 13, SUM, 14, SUM]
    assert ex.kind == [0, 1, 1, 2, 1, 1, 2, 1, 2]
    assert ex.segment == [-1, 0, 0, 0, 1, 1, 1, 2, 2]
    assert ex.raw_index == [-1, 0, 1, -1, 2, 3, -1, 4, -1]
    assert ex.sum_position == [3, 6, 8]
    assert ex.labels == [IGNORE_INDEX, 10, 11, IGNORE_INDEX, 12, 13, IGNORE_INDEX, 14, IGNORE_INDEX]


def test_nested_scope_order_and_parents():
    ex = insert_sums([10, 11, 12, 13, 14], nested_segmentation(), BOS, SUM)
    assert ex.input_ids == [BOS, 10, 11, 12, 13, SUM, 14, SUM]
    assert ex.segment == [-1, 0, 0, 1, 1, 1, 0, 0]
    assert ex.is_header == [False, True, False, True, False, False, False, False]
    assert ex.sum_position == [7, 5]
    assert ex.seg_parent == [-1, 0]


def test_suffix_padding_and_truncation():
    ex = insert_sums([10, 11, 12], fixed_segments(3, 2), BOS, SUM, pad_id=PAD, pad_to=8, suffix_ids=[99])
    assert ex.input_ids == [BOS, 10, 11, SUM, 12, SUM, 99, PAD]
    assert ex.kind == [KIND_BOS, 1, 1, KIND_SUM, 1, KIND_SUM, KIND_RAW, KIND_PAD]
    assert ex.segment[6] == -1 and ex.raw_index[6] == 3
    assert ex.labels[6] == 99 and ex.labels[7] == IGNORE_INDEX

    ex2 = insert_sums([10, 11, 12], fixed_segments(3, 2), BOS, SUM, suffix_ids=[99], suffix_has_labels=False)
    assert ex2.labels[-1] == IGNORE_INDEX

    ex3 = insert_sums([10, 11, 12], fixed_segments(3, 2), BOS, SUM, max_len=5)
    assert ex3.input_ids == [BOS, 10, 11, SUM, 12]
    assert ex3.sum_position == [3, -1] and ex3.n_truncated == 1


def test_roundtrip_dict():
    ex = insert_sums([10, 11, 12, 13, 14], nested_segmentation(), BOS, SUM, seg_edges=[(0, 1)])
    ex2 = Example.from_dict(ex.to_dict())
    assert ex2 == ex
