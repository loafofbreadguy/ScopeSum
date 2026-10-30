"""Shared helpers for the CPU-only tests (no torch, no tokenizer download)."""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scopesum.scope_tracker import physical_lines, track_scopes  # noqa: E402

# Mimics a BPE tokenizer closely enough for scope tests: a newline token
# swallows the next line's indentation (like Gemma's "\n    " pieces).
TOKEN_RE = re.compile(r"(?:\r\n|\r|\n)[ \t]*|[ \t]+|\w+|[^\w\s]")


def toy_tokenize(text):
    """Return (start, end) offsets that cover every character of ``text``."""
    offs = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    covered = sum(e - s for s, e in offs)
    assert covered == len(text), "toy tokenizer must cover all characters"
    return offs


def first_token_on_line(text, offs, line_start, line_end):
    for t, (a, b) in enumerate(offs):
        tok = text[a:b]
        if not tok.strip():
            continue
        anchor = a + (len(tok) - len(tok.lstrip()))
        if line_start <= anchor < line_end:
            return t
    return None


def line_labels(text):
    """Qualified scope name of each non-blank physical line ("?" = unknown)."""
    offs = toy_tokenize(text)
    info = track_scopes(text, offs)
    labels = []
    for s, e in physical_lines(text):
        if not text[s:e].strip():
            continue
        t = first_token_on_line(text, offs, s, e)
        assert t is not None
        sid = info.token_scope[t]
        labels.append("?" if sid < 0 else info.scopes[sid].qualname)
    return labels, info, offs
