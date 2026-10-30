"""Indentation-based scope tracking for Python source code.

Pure Python (no torch, no numpy) so it can be unit-tested anywhere and later
ported to JAX/TPU pipelines unchanged.

What counts as a scope
----------------------
* ``def`` / ``async def`` / ``class`` blocks open a scope. Decorator lines belong
  to the scope of the definition that follows them.
* Module-level code (code that is not inside any def/class) is grouped into
  "module runs": one run starts at the first module-level line and ends when
  the next top-level ``def``/``class``/decorator starts (or at end of text).
* ``if``/``for``/``with``/``try`` blocks do NOT open scopes; they stay part of
  the enclosing scope.

How it works (strictly left to right)
-------------------------------------
1. A small hand-written lexer splits the text into *logical lines*: physical
   lines joined by open brackets, backslash continuations and multi-line
   strings. Only the first physical line of a logical line has meaningful
   indentation. Strings/docstrings/comments never affect indentation.
2. Logical lines are processed in order with an indentation stack (like
   CPython's tokenizer) and a stack of open scopes. A def/class scope closes
   when a non-blank logical line arrives whose indentation is <= the header's
   indentation. Blank lines and comment-only lines never close a scope.
3. Every token is mapped (via its character offsets) to the logical line that
   contains its first non-whitespace character (whitespace-only tokens use
   their start character). The token inherits that line's scope.

Causality note: whether a scope ends is only known when the first token of
the next non-blank logical line is seen. The ``<sum>`` for that scope is
inserted immediately *before* that line's tokens. The tracker never looks at
anything beyond the start of that line; attention masks built from it are
strictly causal.

Broken code never raises. Lines that break indentation rules (unexpected
indent, dedent to an unknown level, inconsistent tabs/spaces, unterminated
strings) are recorded in ``anomalies`` and their tokens get
``UNKNOWN_SCOPE`` (unless the line is a def/class/decorator header, which
still opens its scope so the following body is tracked).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

UNKNOWN_SCOPE = -1

KIND_MODULE = "module"
KIND_DEF = "def"
KIND_CLASS = "class"

# Logical-line kinds.
LINE_BLANK = "blank"  # empty or comment-only
LINE_DEF = "def"
LINE_CLASS = "class"
LINE_DECORATOR = "decorator"
LINE_CODE = "code"

_DEF_RE = re.compile(r"(?:async\s+)?def\s+([^\W\d]\w*)")
_CLASS_RE = re.compile(r"class\s+([^\W\d]\w*)")
_NEWLINE_RE = re.compile(r"\r\n|\r|\n")

_OPENERS = "([{"
_CLOSERS = ")]}"


# ---------------------------------------------------------------------------
# Lexing into logical lines
# ---------------------------------------------------------------------------


@dataclass
class LogicalLine:
    index: int
    first_phys: int
    last_phys: int
    start_char: int
    end_char: int  # exclusive, includes the trailing newline
    indent: tuple[int, int]  # (width with tabsize 8, width with tabsize 1)
    kind: str
    name: str | None = None
    ends_with_colon: bool = False
    anomaly: str | None = None


def physical_lines(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char spans of physical lines; end includes the newline."""
    spans = []
    start = 0
    for m in _NEWLINE_RE.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _indent_width(line: str) -> tuple[int, int]:
    """Indentation width with tabsize 8 (CPython rule) and tabsize 1."""
    w8 = w1 = 0
    for ch in line:
        if ch == " ":
            w8 += 1
            w1 += 1
        elif ch == "\t":
            w8 = (w8 // 8 + 1) * 8
            w1 += 1
        elif ch == "\f":
            w8 = w1 = 0
        else:
            break
    return (w8, w1)


def split_logical_lines(text: str) -> tuple[list[LogicalLine], list[int]]:
    """Lex ``text`` into logical lines.

    Returns ``(logical_lines, phys_to_logical)`` where ``phys_to_logical[p]``
    is the logical-line index that physical line ``p`` belongs to.
    Never raises.
    """
    phys = physical_lines(text)
    logical: list[LogicalLine] = []
    phys_to_logical: list[int] = []

    quote: str | None = None  # active string delimiter, carried across lines
    depth = 0
    cont = False  # previous physical line ended with a backslash continuation
    cur: LogicalLine | None = None
    last_sig = ""  # last significant (non-space, non-comment) char of the logical line
    anomaly: str | None = None

    for p, (s, e) in enumerate(phys):
        line = text[s:e].rstrip("\r\n")
        starts_logical = quote is None and depth == 0 and not cont
        if starts_logical:
            stripped = line.lstrip(" \t\f")
            cur = LogicalLine(
                index=len(logical),
                first_phys=p,
                last_phys=p,
                start_char=s,
                end_char=e,
                indent=_indent_width(line),
                kind=LINE_CODE,
            )
            if stripped.startswith("@"):
                cur.kind = LINE_DECORATOR
            else:
                m = _DEF_RE.match(stripped)
                if m:
                    cur.kind, cur.name = LINE_DEF, m.group(1)
                else:
                    m = _CLASS_RE.match(stripped)
                    if m:
                        cur.kind, cur.name = LINE_CLASS, m.group(1)
            logical.append(cur)
            last_sig = ""
            anomaly = None
            has_code = False
        else:
            assert cur is not None
            cur.last_phys = p
            cur.end_char = e
            has_code = True  # continuation lines are never "blank"
        phys_to_logical.append(cur.index)

        # --- scan characters of this physical line ---
        cont = False
        i, n = 0, len(line)
        escaped_newline = False
        while i < n:
            c = line[i]
            if quote is not None:
                if c == "\\":
                    if i == n - 1:
                        escaped_newline = True
                    i += 2
                    continue
                if line.startswith(quote, i):
                    i += len(quote)
                    quote = None
                    continue
                i += 1
                continue
            if c == "#":
                break
            if c == '"' or c == "'":
                quote = c * 3 if line.startswith(c * 3, i) else c
                i += len(quote)
                has_code = True
                last_sig = c
                continue
            if c in _OPENERS:
                depth += 1
            elif c in _CLOSERS:
                depth -= 1
                if depth < 0:
                    depth = 0
                    anomaly = anomaly or "unbalanced closing bracket"
            if c == "\\" and i == n - 1:
                cont = True
            elif not c.isspace():
                has_code = True
                last_sig = c
            i += 1

        if quote is not None and len(quote) == 1 and not escaped_newline:
            # Single-quoted string not closed on its line: recover.
            quote = None
            anomaly = anomaly or "unterminated string"

        if starts_logical and not has_code and quote is None and depth == 0 and not cont:
            cur.kind = LINE_BLANK

        ends_here = quote is None and depth == 0 and not cont
        if ends_here or p == len(phys) - 1:
            cur.ends_with_colon = last_sig == ":"
            if not ends_here:
                anomaly = anomaly or "unterminated string or bracket at end of text"
            cur.anomaly = anomaly

    return logical, phys_to_logical


# ---------------------------------------------------------------------------
# Scope tracking
# ---------------------------------------------------------------------------


@dataclass
class Scope:
    scope_id: int
    kind: str  # module | def | class
    name: str | None
    depth: int
    parent: int  # -1 for none
    header_indent: tuple[int, int]
    open_line: int
    pending: bool = False  # decorator seen, def/class line not yet seen
    close_line: int | None = None  # logical line whose start closes the scope; None = end of text
    open_token: int = -1
    close_token: int = -1  # <sum> is inserted before this raw token index (n_tokens = end)
    n_tokens: int = 0  # tokens directly in this scope (not in nested scopes)
    ancestors: tuple[int, ...] = ()  # root first, excluding self
    qualname: str = ""


@dataclass
class ScopeInfo:
    """Per-token and per-scope results of :func:`track_scopes`."""

    token_scope: list[int]
    token_depth: list[int]
    token_parents: list[tuple[int, ...]]  # ancestors of the token's scope, root first
    token_is_header: list[bool]
    token_line: list[int]  # logical line index, -1 for zero-length tokens
    scopes: list[Scope]
    closures: list[tuple[int, int]]  # (insert_before_token_idx, scope_id) in emission order
    anomalies: list[tuple[int, str]] = field(default_factory=list)  # (logical line, message)
    lines: list[LogicalLine] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.token_scope)


def _cmp_indent(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, bool]:
    """Compare two indents. Returns (sign, consistent).

    ``consistent`` is False when the tabsize-8 and tabsize-1 interpretations
    disagree (CPython raises TabError in that case).
    """
    s8 = (a[0] > b[0]) - (a[0] < b[0])
    s1 = (a[1] > b[1]) - (a[1] < b[1])
    return s8, s8 == s1


def _token_anchor_chars(text: str, token_offsets) -> list[int]:
    anchors = []
    n = len(text)
    for s, e in token_offsets:
        if e <= s or s >= n:
            anchors.append(-1)
            continue
        j = s
        e = min(e, n)
        while j < e and text[j].isspace():
            j += 1
        anchors.append(j if j < e else s)
    return anchors


def track_scopes(text: str, token_offsets) -> ScopeInfo:
    """Assign every token a scope.

    Args:
        text: Python source text.
        token_offsets: sequence of ``(start_char, end_char)`` per token, as
            returned by a fast tokenizer's ``offset_mapping``. Zero-length
            spans (special tokens) get ``UNKNOWN_SCOPE``.
    """
    lines, phys_to_logical = split_logical_lines(text)
    phys = physical_lines(text)
    phys_starts = [s for s, _ in phys]

    scopes: list[Scope] = []
    open_stack: list[int] = []
    indent_stack: list[tuple[int, int]] = [(0, 0)]
    prev_colon = False
    anomalies: list[tuple[int, str]] = []
    line_scope = [UNKNOWN_SCOPE] * len(lines)
    line_header = [False] * len(lines)
    line_closes: list[list[int]] = [[] for _ in lines]

    def open_scope(kind, name, indent, li, pending=False):
        parent = open_stack[-1] if open_stack else -1
        depth = scopes[parent].depth + 1 if parent >= 0 else 0
        ancestors = (scopes[parent].ancestors + (parent,)) if parent >= 0 else ()
        sc = Scope(
            scope_id=len(scopes),
            kind=kind,
            name=name,
            depth=depth,
            parent=parent,
            header_indent=indent,
            open_line=li,
            pending=pending,
            ancestors=ancestors,
        )
        scopes.append(sc)
        open_stack.append(sc.scope_id)
        return sc.scope_id

    def should_close(sc: Scope, ll: LogicalLine) -> bool:
        sign, _ = _cmp_indent(ll.indent, sc.header_indent)
        if sc.kind == KIND_MODULE:
            return ll.kind in (LINE_DEF, LINE_CLASS, LINE_DECORATOR) and sign <= 0
        if sc.pending and ll.kind in (LINE_DEF, LINE_CLASS, LINE_DECORATOR) and sign == 0:
            return False  # decorator continues into its def/class
        return sign <= 0

    for ll in lines:
        li = ll.index
        if ll.kind == LINE_BLANK:
            if not open_stack:
                open_scope(KIND_MODULE, None, (0, 0), li)
            line_scope[li] = open_stack[-1]
            continue

        # --- indentation bookkeeping (CPython-like) ---
        anomaly = ll.anomaly
        sign, consistent = _cmp_indent(ll.indent, indent_stack[-1])
        if not consistent:
            anomaly = anomaly or "inconsistent use of tabs and spaces"
        if prev_colon and sign <= 0:
            anomaly = anomaly or "expected an indented block"
        if sign > 0:
            if not prev_colon:
                anomaly = anomaly or "unexpected indent"
            indent_stack.append(ll.indent)
        elif sign < 0:
            while len(indent_stack) > 1 and _cmp_indent(indent_stack[-1], ll.indent)[0] > 0:
                indent_stack.pop()
            top_sign, _ = _cmp_indent(ll.indent, indent_stack[-1])
            if top_sign != 0:
                anomaly = anomaly or "dedent does not match any outer indentation level"
                if top_sign > 0:
                    indent_stack.append(ll.indent)

        # --- close finished scopes (innermost first) ---
        while open_stack and should_close(scopes[open_stack[-1]], ll):
            sid = open_stack.pop()
            scopes[sid].close_line = li
            if scopes[sid].pending:
                scopes[sid].pending = False
                anomaly = anomaly or "decorator not followed by def/class"
            line_closes[li].append(sid)

        # --- open / continue scopes ---
        if ll.kind in (LINE_DEF, LINE_CLASS):
            kind = KIND_DEF if ll.kind == LINE_DEF else KIND_CLASS
            top = scopes[open_stack[-1]] if open_stack else None
            if top is not None and top.pending and _cmp_indent(ll.indent, top.header_indent)[0] == 0:
                top.kind, top.name, top.pending = kind, ll.name, False
                sid = top.scope_id
            else:
                sid = open_scope(kind, ll.name, ll.indent, li)
            line_scope[li] = sid
            line_header[li] = True
        elif ll.kind == LINE_DECORATOR:
            top = scopes[open_stack[-1]] if open_stack else None
            if top is not None and top.pending and _cmp_indent(ll.indent, top.header_indent)[0] == 0:
                sid = top.scope_id
            else:
                sid = open_scope(KIND_DEF, None, ll.indent, li, pending=True)
            line_scope[li] = sid
            line_header[li] = True
        else:
            if not open_stack:
                open_scope(KIND_MODULE, None, ll.indent, li)
            line_scope[li] = UNKNOWN_SCOPE if anomaly else open_stack[-1]

        if anomaly:
            anomalies.append((li, anomaly))
        prev_colon = ll.ends_with_colon

    # Scopes still open at end of text close innermost first.
    eof_closes = list(reversed(open_stack))
    for sid in eof_closes:
        scopes[sid].pending = False

    # --- map tokens to logical lines ---
    anchors = _token_anchor_chars(text, token_offsets)
    n_tok = len(anchors)
    token_line = []
    for a in anchors:
        if a < 0:
            token_line.append(-1)
        else:
            p = bisect.bisect_right(phys_starts, a) - 1
            token_line.append(phys_to_logical[p] if p >= 0 else -1)

    token_scope = [UNKNOWN_SCOPE] * n_tok
    token_is_header = [False] * n_tok
    for t, li in enumerate(token_line):
        if li >= 0:
            token_scope[t] = line_scope[li]
            token_is_header[t] = line_header[li] and line_scope[li] != UNKNOWN_SCOPE
            sid = token_scope[t]
            if sid != UNKNOWN_SCOPE:
                sc = scopes[sid]
                sc.n_tokens += 1
                if sc.open_token < 0:
                    sc.open_token = t

    # Insertion index for a logical line = first token anchored at or after it.
    # Token lines are non-decreasing for valid tokens; zero-length tokens (-1)
    # are skipped by searching only over valid entries.
    valid = [(li, t) for t, li in enumerate(token_line) if li >= 0]
    valid_lines = [li for li, _ in valid]

    def first_token_at_or_after(li: int) -> int:
        k = bisect.bisect_left(valid_lines, li)
        return valid[k][1] if k < len(valid) else n_tok

    closures: list[tuple[int, int]] = []
    for li, sids in enumerate(line_closes):
        if sids:
            idx = first_token_at_or_after(li)
            for sid in sids:
                scopes[sid].close_token = idx
                closures.append((idx, sid))
    for sid in eof_closes:
        scopes[sid].close_token = n_tok
        closures.append((n_tok, sid))

    for sc in scopes:
        sc.qualname = qualname(scopes, sc.scope_id)

    token_depth = [scopes[s].depth if s >= 0 else -1 for s in token_scope]
    token_parents = [scopes[s].ancestors if s >= 0 else () for s in token_scope]
    return ScopeInfo(
        token_scope=token_scope,
        token_depth=token_depth,
        token_parents=token_parents,
        token_is_header=token_is_header,
        token_line=token_line,
        scopes=scopes,
        closures=closures,
        anomalies=anomalies,
        lines=lines,
    )


def qualname(scopes: list[Scope], sid: int) -> str:
    """Dotted name like ``Outer.method.inner``; module runs are ``<module>``."""
    parts = []
    while sid >= 0:
        sc = scopes[sid]
        if sc.kind == KIND_MODULE:
            parts.append("<module>")
        else:
            parts.append(sc.name or "<decorated>")
        sid = sc.parent
    return ".".join(reversed(parts))
