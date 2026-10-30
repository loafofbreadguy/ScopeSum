"""Variant E: caller -> callee edges between ``<sum>`` tokens.

Pure Python. Two sources of edges:

1. :func:`regex_call_edges` -- works on any window of code (training data and
   evaluation). Finds ``name(`` and ``self.name(`` / ``cls.name(`` call
   sites, attributes each call to the segment of the token where it occurs
   (the caller) and resolves the name to a def/class segment in the same
   window (the callee). Cheap and robust to partial windows that do not
   parse. Calls inside strings/comments may add a few spurious edges.

2. The competition's ``graphs/`` files (evaluation only). Their schema is not
   known offline, so :func:`discover_graph_schema` prints what is there and
   :func:`edges_from_graph` maps a graph to segment edges using config-driven
   field names (see ``GraphSchema``).

An edge ``(a, b)`` lets ``<sum>`` of segment ``a`` attend to ``<sum>`` of
segment ``b``; the mask builder drops edges whose callee ``<sum>`` comes later
(causality) and self-edges.
"""

from __future__ import annotations

import bisect
import json
import keyword
import os
import re
from dataclasses import dataclass

from .segmenter import Segmentation

_CALL_RE = re.compile(r"(?<![\w.])(?:(self|cls)\s*\.\s*)?([^\W\d]\w*)\s*\(")
_DEF_BEFORE_RE = re.compile(r"(?:def|class)\s+$")


def _token_index_at(starts: list[int], char: int) -> int:
    return max(bisect.bisect_right(starts, char) - 1, 0)


def _enclosing_class(seg: Segmentation, s: int) -> int:
    """Nearest ancestor segment (or itself) that is a class, else -1."""
    seen = 0
    while s >= 0 and seen <= len(seg.segments):
        if seg.segments[s].kind == "class":
            return s
        s = seg.segments[s].parent
        seen += 1
    return -1


def regex_call_edges(text: str, token_offsets, seg: Segmentation) -> list[tuple[int, int]]:
    """Caller->callee segment edges found by regex over ``text``.

    ``token_offsets`` and ``seg.token_segment`` must describe the same tokens.
    """
    if not seg.segments or not token_offsets:
        return []
    starts = [s for s, _ in token_offsets]
    by_name: dict[str, list[int]] = {}
    for s in seg.segments:
        if s.kind in ("def", "class") and s.name:
            by_name.setdefault(s.name, []).append(s.seg_id)

    edges: set[tuple[int, int]] = set()
    for m in _CALL_RE.finditer(text):
        receiver, name = m.group(1), m.group(2)
        if keyword.iskeyword(name) or name not in by_name:
            continue
        if _DEF_BEFORE_RE.search(text[max(0, m.start() - 16) : m.start()]):
            continue  # "def name(" is a definition, not a call
        caller = seg.token_segment[_token_index_at(starts, m.start(2))]
        if caller < 0:
            continue
        candidates = by_name[name]
        if receiver is not None:
            cls = _enclosing_class(seg, caller)
            candidates = [c for c in candidates if seg.segments[c].parent == cls and cls >= 0]
        else:
            # plain call: prefer module-level / sibling definitions over methods
            plain = [c for c in candidates if _enclosing_class(seg, seg.segments[c].parent) < 0 or seg.segments[c].kind == "class"]
            candidates = plain or candidates
        for callee in candidates:
            if callee != caller:
                edges.add((caller, callee))
    return sorted(edges)


# ---------------------------------------------------------------------------
# competition graphs (schema discovered on Kaggle)
# ---------------------------------------------------------------------------


@dataclass
class GraphSchema:
    """Field names inside one competition graph file. VERIFY with discovery."""

    nodes_key: str = "nodes"
    edges_key: str = "edges"
    node_id_key: str = "id"
    node_name_key: str = "name"  # function name or qualified name
    node_file_key: str = "file"
    edge_src_key: str = "source"
    edge_dst_key: str = "target"
    edge_type_key: str | None = "type"
    call_edge_types: tuple[str, ...] = ("calls", "call", "CALLS")


def _load_any(path: str):
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def discover_graph_schema(graphs_dir: str, max_files: int = 3, max_chars: int = 1500) -> list[str]:
    """Print the structure of a few graph files; return the file list."""
    files = []
    for root, _, names in os.walk(graphs_dir):
        for n in sorted(names):
            files.append(os.path.join(root, n))
    print(f"[graphs] {len(files)} files under {graphs_dir}")
    for p in files[:max_files]:
        print(f"--- {p}")
        try:
            obj = _load_any(p)
        except Exception as e:  # unknown format: show raw text
            print(f"  (not JSON: {e}); first bytes:")
            with open(p, errors="replace") as f:
                print(f.read(max_chars))
            continue
        _describe(obj)
        print(json.dumps(obj, default=str)[:max_chars])
    return files


def _describe(obj, indent: str = "  ", depth: int = 0) -> None:
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:20]:
            print(f"{indent}{k}: {type(v).__name__}" + (f" (len {len(v)})" if isinstance(v, (list, dict)) else ""))
            if isinstance(v, (dict, list)):
                _describe(v, indent + "  ", depth + 1)
    elif isinstance(obj, list) and obj:
        print(f"{indent}[0]: {type(obj[0]).__name__}")
        _describe(obj[0], indent + "  ", depth + 1)


def call_pairs_from_graph(graph: dict, schema: GraphSchema, file_suffix: str | None = None) -> list[tuple[str, str]]:
    """(caller name, callee name) pairs from one graph, optionally for one file."""
    nodes = {n[schema.node_id_key]: n for n in graph.get(schema.nodes_key, [])}
    pairs = []
    for e in graph.get(schema.edges_key, []):
        if schema.edge_type_key and schema.edge_type_key in e and e[schema.edge_type_key] not in schema.call_edge_types:
            continue
        a, b = nodes.get(e.get(schema.edge_src_key)), nodes.get(e.get(schema.edge_dst_key))
        if a is None or b is None:
            continue
        if file_suffix is not None:
            fa = str(a.get(schema.node_file_key, ""))
            if not fa.endswith(file_suffix) and not file_suffix.endswith(fa):
                continue
        pairs.append((str(a.get(schema.node_name_key, "")), str(b.get(schema.node_name_key, ""))))
    return pairs


def edges_from_name_pairs(pairs: list[tuple[str, str]], seg: Segmentation) -> list[tuple[int, int]]:
    """Map (caller, callee) names to segment ids by qualname, then by last name part."""
    by_qual: dict[str, list[int]] = {}
    by_last: dict[str, list[int]] = {}
    for s in seg.segments:
        if s.kind not in ("def", "class"):
            continue
        q = (s.qualname or "").replace("<module>.", "")
        by_qual.setdefault(q, []).append(s.seg_id)
        if s.name:
            by_last.setdefault(s.name, []).append(s.seg_id)

    def lookup(name: str) -> list[int]:
        name = name.replace("::", ".").replace(":", ".")
        if name in by_qual:
            return by_qual[name]
        for q, ids in by_qual.items():
            if name.endswith("." + q):
                return ids
        return by_last.get(name.rsplit(".", 1)[-1], [])

    edges = set()
    for a, b in pairs:
        for x in lookup(a):
            for y in lookup(b):
                if x != y:
                    edges.add((x, y))
    return sorted(edges)
