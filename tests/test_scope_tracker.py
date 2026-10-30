import textwrap

import pytest
from conftest import line_labels, toy_tokenize

from scopesum.scope_tracker import (
    LINE_BLANK,
    LINE_CODE,
    LINE_DEF,
    UNKNOWN_SCOPE,
    split_logical_lines,
    track_scopes,
)


def src(s):
    return textwrap.dedent(s).lstrip("\n")


# Each case: (source, expected qualified scope per non-blank physical line)
CASES = {
    "function_and_module_code": (
        src(
            """
            import os

            def f(x):
                y = x + 1
                return y

            z = f(1)
            """
        ),
        ["<module>", "f", "f", "f", "<module>"],
    ),
    "nested_functions_and_class": (
        src(
            """
            class A:
                x = 1

                def m(self):
                    def inner():
                        return 2
                    return inner()

                @property
                def p(self):
                    return self.x
            """
        ),
        ["A", "A", "A.m", "A.m.inner", "A.m.inner", "A.m", "A.p", "A.p", "A.p"],
    ),
    "decorators_and_multiline_signature": (
        src(
            """
            @decorator(
                arg=1,
            )
            @other
            def g(
                a,
                b,
            ):
                return a
            """
        ),
        ["g"] * 9,
    ),
    "docstrings_containing_code_and_indentation": (
        src(
            '''
            def h():
                """Doc.

            def not_a_function():
                still docstring
            """
                s = \'\'\'
            x = 1
            \'\'\'
                return s
            '''
        ),
        ["h"] * 9,
    ),
    "blank_lines_and_comments_do_not_close": (
        src(
            """
            def f():
                a = 1

            # a comment at column 0

                b = 2
                return a + b
            """
        ),
        ["f"] * 5,
    ),
    "async_def_with_tabs": (
        "async def run():\n\tawait x()\n\treturn 1\ny = 2\n",
        ["run", "run", "run", "<module>"],
    ),
    "broken_dedent_falls_back_to_unknown": (
        "def f():\n        a = 1\n    b = 2\nc = 3\n",
        ["f", "f", "?", "<module>"],
    ),
    "unexpected_indent_and_unterminated_string": (
        'x = 1\n    y = 2\ns = "abc\ndef g():\n    return 1\n',
        ["<module>", "?", "?", "g", "g"],
    ),
    "one_liner_definitions": (
        "def a(): return 1\ndef b(): return 2\nclass C: pass\nx = a()\n",
        ["a", "b", "C", "<module>"],
    ),
    "def_inside_module_level_if": (
        src(
            """
            import sys
            if sys.version_info > (3,):
                def f():
                    return 3
            else:
                f = None
            print(f)
            """
        ),
        ["<module>", "<module>", "<module>.f", "<module>.f", "<module>", "<module>", "<module>"],
    ),
    "window_starting_mid_function": (
        "        return x\n    y = 3\ndef k():\n    pass\n",
        ["?", "?", "k", "k"],
    ),
    "crlf_and_backslash_continuation": (
        "def f():\r\n    x = 1 + \\\r\n2\r\n    return x\r\n",
        ["f", "f", "f", "f"],
    ),
    "brackets_spanning_lines_at_column_zero": (
        src(
            """
            def f():
                data = [
            1,
            2,
                ]
                return data
            y = f()
            """
        ),
        ["f", "f", "f", "f", "f", "f", "<module>"],
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_line_scopes(name):
    text, expected = CASES[name]
    labels, info, _ = line_labels(text)
    assert labels == expected, f"{name}: {labels}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_closures_are_consistent(name):
    text, _ = CASES[name]
    offs = toy_tokenize(text)
    info = track_scopes(text, offs)
    idxs = [i for i, _ in info.closures]
    assert idxs == sorted(idxs), "insertion points must be non-decreasing"
    assert sorted(s for _, s in info.closures) == list(range(len(info.scopes))), "every scope closes once"
    for t, sid in enumerate(info.token_scope):
        if sid != UNKNOWN_SCOPE:
            # a scope's tokens all come before its <sum>, and its ancestors close later
            assert t < info.scopes[sid].close_token
            for a in info.scopes[sid].ancestors:
                assert info.scopes[a].close_token >= info.scopes[sid].close_token


def test_nested_structure_details():
    text, _ = CASES["nested_functions_and_class"]
    _, info, offs = line_labels(text)
    by_name = {sc.qualname: sc for sc in info.scopes}
    assert set(by_name) == {"A", "A.m", "A.m.inner", "A.p"}
    assert by_name["A"].depth == 0 and by_name["A.m"].depth == 1 and by_name["A.m.inner"].depth == 2
    assert by_name["A.m.inner"].parent == by_name["A.m"].scope_id
    assert by_name["A.p"].ancestors == (by_name["A"].scope_id,)
    # closure order: inner, m, then p and A at end of text (innermost first)
    order = [info.scopes[s].qualname for _, s in info.closures]
    assert order == ["A.m.inner", "A.m", "A.p", "A"]
    n = len(offs)
    assert by_name["A.p"].close_token == n and by_name["A"].close_token == n
    # inner closes right before the token "return" of "return inner()"
    t = by_name["A.m.inner"].close_token
    assert text[offs[t][0]:offs[t][1]] == "return"
    assert text[offs[t][0]:].startswith("return inner()")


def test_header_flags_cover_decorators_and_signature():
    text, _ = CASES["decorators_and_multiline_signature"]
    offs = toy_tokenize(text)
    info = track_scopes(text, offs)
    words = {text[a:b]: info.token_is_header[t] for t, (a, b) in enumerate(offs)}
    assert words["decorator"] and words["arg"] and words["other"] and words["def"] and words["b"]
    assert words["return"] is False
    assert len(info.scopes) == 1 and info.scopes[0].kind == "def" and info.scopes[0].name == "g"


def test_broken_code_records_anomalies_and_never_raises():
    for text in [
        "def f(:\n  )))\n\tx = '''\n",
        "\t  \n  def\n@\n",
        "class\n    pass\n  x\n",
        "",
        "\n\n\n",
        "x = (\n",
        '"""unterminated\n def f():\n',
    ]:
        offs = toy_tokenize(text)
        info = track_scopes(text, offs)
        assert len(info.token_scope) == len(offs)
    _, info, _ = line_labels(CASES["broken_dedent_falls_back_to_unknown"][0])
    assert info.anomalies and "dedent" in info.anomalies[0][1]


def test_zero_length_special_tokens_are_unknown():
    text = "def f():\n    return 1\n"
    offs = [(0, 0)] + toy_tokenize(text)  # e.g. BOS from a tokenizer
    info = track_scopes(text, offs)
    assert info.token_scope[0] == UNKNOWN_SCOPE
    assert all(s == 0 for s in info.token_scope[1:])
    assert info.closures == [(len(offs), 0)]


def test_logical_lines():
    text = "x = [1,\n  2]\n# c\n\ndef f(a,\n      b):\n    pass\n"
    lines, phys_to_logical = split_logical_lines(text)
    kinds = [ll.kind for ll in lines]
    assert kinds == [LINE_CODE, LINE_BLANK, LINE_BLANK, LINE_DEF, LINE_CODE]
    assert phys_to_logical == [0, 0, 1, 2, 3, 3, 4]
    assert lines[3].ends_with_colon and lines[3].name == "f"


def test_module_run_with_only_unknown_tokens_is_dropped_by_segmenter():
    from scopesum.segmenter import scope_segments

    text, _ = CASES["window_starting_mid_function"]
    info = track_scopes(text, toy_tokenize(text))
    seg = scope_segments(info)
    assert [s.qualname for s in seg.segments] == ["k"]
