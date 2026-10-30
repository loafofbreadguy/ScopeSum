import textwrap

from conftest import toy_tokenize

from scopesum.graph_edges import GraphSchema, call_pairs_from_graph, edges_from_name_pairs, regex_call_edges
from scopesum.masks import MaskMeta, build_sliding_mask
from scopesum.scope_tracker import track_scopes
from scopesum.segmenter import scope_segments
from scopesum.sum_insertion import insert_sums

CODE = textwrap.dedent(
    """\
    def helper(x):
        return x * 2

    class Box:
        def get(self):
            return helper(self.v)

        def twice(self):
            return self.get() + self.get()

    def main():
        b = Box()
        print(b.twice())
        return helper(1)
    """
)


def segmented():
    offs = toy_tokenize(CODE)
    seg = scope_segments(track_scopes(CODE, offs))
    return offs, seg


def names(seg, edges):
    q = lambda s: seg.segments[s].qualname  # noqa: E731
    return sorted((q(a), q(b)) for a, b in edges)


def test_regex_call_edges():
    offs, seg = segmented()
    edges = regex_call_edges(CODE, offs, seg)
    assert names(seg, edges) == [
        ("Box.get", "helper"),
        ("Box.twice", "Box.get"),
        ("main", "Box"),
        ("main", "helper"),
    ]
    # "b.twice()" is an attribute call on a non-self object: not resolved
    # "def helper(" is a definition, not a call


def test_graph_pairs_to_edges():
    _, seg = segmented()
    graph = {
        "nodes": [
            {"id": 1, "name": "pkg.mod.main", "file": "pkg/mod.py"},
            {"id": 2, "name": "pkg.mod.Box.twice", "file": "pkg/mod.py"},
            {"id": 3, "name": "other", "file": "pkg/other.py"},
        ],
        "edges": [
            {"source": 1, "target": 2, "type": "calls"},
            {"source": 1, "target": 3, "type": "imports"},
        ],
    }
    pairs = call_pairs_from_graph(graph, GraphSchema(), file_suffix="pkg/mod.py")
    assert pairs == [("pkg.mod.main", "pkg.mod.Box.twice")]
    assert names(seg, edges_from_name_pairs(pairs, seg)) == [("main", "Box.twice")]


def test_edges_change_only_sum_rows_in_mask():
    offs, seg = segmented()
    edges = regex_call_edges(CODE, offs, seg)
    raw = list(range(100, 100 + len(offs)))
    ex_d = insert_sums(raw, seg, 2, 6)
    ex_e = insert_sums(raw, seg, 2, 6, seg_edges=edges)
    d = build_sliding_mask("D", MaskMeta.from_example(ex_d), window=512, w_small=4)
    e = build_sliding_mask("E", MaskMeta.from_example(ex_e), window=512, w_small=4)
    changed_rows = {int(r) for r in (d != e).any(axis=1).nonzero()[0]}
    assert changed_rows, "E must add at least one edge here"
    assert all(ex_e.kind[r] == 2 for r in changed_rows), "only <sum> rows gain edges"
    # e.g. main's <sum> now sees helper's <sum>
    main = next(s.seg_id for s in seg.segments if s.qualname == "main")
    helper = next(s.seg_id for s in seg.segments if s.qualname == "helper")
    assert e[ex_e.sum_position[main], ex_e.sum_position[helper]]
    assert not d[ex_d.sum_position[main], ex_d.sum_position[helper]]
