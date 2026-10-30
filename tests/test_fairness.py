"""B and C must insert (as nearly as possible) the same number of <sum> tokens."""

import textwrap

from conftest import toy_tokenize

from scopesum.scope_tracker import track_scopes
from scopesum.segmenter import fairness_stats, fixed_segments, n_fixed_sums, scope_segments
from scopesum.sum_insertion import KIND_SUM, insert_sums

CORPUS = [
    textwrap.dedent(
        """\
        import os

        def a(x):
            return x + 1

        def b(y):
            def c(z):
                return z * 2
            return c(y)

        class K:
            def m(self):
                return 1
        """
    ),
    "x = 1\ny = 2\nz = x + y\nprint(z)\n" * 5,
    textwrap.dedent(
        """\
        def long_function(a, b, c):
            total = 0
            for i in range(a):
                for j in range(b):
                    total += i * j + c
            if total > 100:
                total = 100
            return total
        """
    ),
]


def corpus_counts():
    lengths, scope_counts = [], []
    for text in CORPUS:
        offs = toy_tokenize(text)
        seg = scope_segments(track_scopes(text, offs))
        lengths.append(len(offs))
        scope_counts.append(seg.n_sums)
    return lengths, scope_counts


def test_fixed_chunk_count_matches_scope_count():
    lengths, scope_counts = corpus_counts()
    stats = fairness_stats(lengths, scope_counts)
    k = stats["chunk_size"]
    fixed_total = sum(n_fixed_sums(n, k) for n in lengths)
    target = sum(scope_counts)
    assert fixed_total == stats["fixed_sums_total"]
    assert fixed_total <= target
    # one token smaller K would overshoot: K is the tightest fair choice
    assert sum(n_fixed_sums(n, k - 1) for n in lengths) > target
    # with this corpus the gap is small relative to the number of examples
    assert target - fixed_total <= len(lengths)


def test_inserted_sum_tokens_equal_segment_counts():
    lengths, scope_counts = corpus_counts()
    k = fairness_stats(lengths, scope_counts)["chunk_size"]
    total_b = total_c = 0
    for text in CORPUS:
        offs = toy_tokenize(text)
        raw = list(range(100, 100 + len(offs)))
        ex_c = insert_sums(raw, scope_segments(track_scopes(text, offs)), 2, 6)
        ex_b = insert_sums(raw, fixed_segments(len(offs), k), 2, 6)
        total_c += sum(1 for x in ex_c.kind if x == KIND_SUM)
        total_b += sum(1 for x in ex_b.kind if x == KIND_SUM)
        # same raw tokens in the same order: B and C differ only in boundaries
        assert [t for t, kd in zip(ex_b.input_ids, ex_b.kind) if kd == 1] == raw
        assert [t for t, kd in zip(ex_c.input_ids, ex_c.kind) if kd == 1] == raw
    assert total_c == sum(scope_counts)
    assert total_b == sum(n_fixed_sums(n, k) for n in lengths)
