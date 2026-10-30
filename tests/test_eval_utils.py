import textwrap

from conftest import toy_tokenize

from scopesum.eval_localize import function_table, parse_patch, topk_hit, touched_functions
from scopesum.eval_longrange import identifier_events, long_range_targets
from scopesum.eval_memory import analytic_kv
from scopesum.probe import function_labels
from scopesum.segmenter import fixed_segments
from scopesum.sum_insertion import insert_sums

PATCH = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -3,4 +3,5 @@ def f():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
diff --git a/new.py b/new.py
--- /dev/null
+++ b/new.py
@@ -0,0 +1,1 @@
+x = 1
"""

CODE = textwrap.dedent(
    """\
    import os

    class S:
        @dec
        def m(self, a):
            def inner():
                return 1
            return inner()

    def f(x, *args, **kw):
        yield x
    """
)


def test_parse_patch():
    assert parse_patch(PATCH) == {"pkg/mod.py": {4}}


def test_function_table_and_touched():
    rows = {r["qualname"]: (r["start"], r["end"]) for r in function_table(CODE)}
    assert rows == {"S.m": (4, 8), "S.m.inner": (6, 7), "f": (10, 11)}
    funcs = function_table(CODE)
    assert touched_functions(funcs, {7}) == {"S.m.inner"}
    assert touched_functions(funcs, {8, 11}) == {"S.m", "f"}
    assert touched_functions(funcs, {1}) == set()
    assert topk_hit(["a", "b", "c"], {"c"}, 3) and not topk_hit(["a", "b", "c"], {"c"}, 2)


def test_probe_labels():
    labs = function_labels(CODE)
    assert labs["S.m"] == {"returns_value": 1, "n_args_bucket": 1, "name_reused": 0}
    assert labs["S.m.inner"] == {"returns_value": 1, "n_args_bucket": 0, "name_reused": 1}
    assert labs["f"] == {"returns_value": 1, "n_args_bucket": 3, "name_reused": 0}


LR_CODE = textwrap.dedent(
    """\
    CONST = 1

    def helper():
        return 2

    def a():
        x = 1
        y = 2
        z = 3
        return x + y + z

    def user():
        return helper() + CONST
    """
)


def test_long_range_targets():
    defs, uses = identifier_events(LR_CODE)
    assert ("CONST", 0) in defs
    offs = toy_tokenize(LR_CODE)
    window = {"offsets": [list(o) for o in offs], "text": LR_CODE, "start_char": 0}
    targets = long_range_targets(window, defs, uses, min_distance=10)
    words = {LR_CODE[offs[t][0]:offs[t][1]] for t in targets}
    assert words == {"helper", "CONST"}, words
    assert long_range_targets(window, defs, uses, min_distance=10_000) == []


def test_long_range_targets_respect_window_offset():
    defs, uses = identifier_events(LR_CODE)
    start = LR_CODE.index("def a():")
    text = LR_CODE[start:]
    offs = toy_tokenize(text)
    window = {"offsets": [list(o) for o in offs], "text": text, "start_char": start}
    # helper/CONST are defined before the window starts: not targets
    assert long_range_targets(window, defs, uses, min_distance=1) == []


def test_analytic_kv_baseline_is_bounded_by_window():
    ex = insert_sums(list(range(20)), None, 2, 6, pad_to=24)
    rep = analytic_kv(ex, "A", window=4, w_small=2)
    assert rep["length"] == 21 and rep["live_max"] == 4
    ex_b = insert_sums(list(range(20)), fixed_segments(20, 5), 2, 6)
    rep_b = analytic_kv(ex_b, "B", window=4, w_small=2)
    assert rep_b["live_max"] > 4  # summaries accumulate on top of the window
