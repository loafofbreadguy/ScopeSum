"""Insert ``<sum>`` tokens into a raw token sequence.

Pure Python. Produces the model input (``input_ids``, ``labels``) plus the
per-position metadata that :mod:`scopesum.masks` needs to rebuild every
variant's attention mask.

Layout of one example::

    [BOS] raw_0 raw_1 ... <sum> raw_k ... <sum> [suffix tokens] [PAD ...]

* Position IDs are simply ``0..L-1``: a ``<sum>`` takes the next position.
* Labels: ``<sum>``, BOS and PAD positions are ``-100`` (never predicted).
  Raw tokens (and suffix tokens unless ``suffix_has_labels=False``) keep their
  id. Labels are in Hugging Face convention (aligned with ``input_ids``; the
  loss shifts by one), so the raw token right after a ``<sum>`` is predicted
  from the ``<sum>`` position.
* Suffix tokens (e.g. an evaluation prompt) are raw tokens that belong to no
  segment (segment -1) and come after every ``<sum>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .segmenter import Segmentation

KIND_BOS = 0
KIND_RAW = 1
KIND_SUM = 2
KIND_PAD = 3

IGNORE_INDEX = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    kind: list[int]  # KIND_* per position
    segment: list[int]  # RAW: owning segment, SUM: summarized segment, else -1
    is_header: list[bool]
    raw_index: list[int]  # RAW: index into raw tokens (suffix continues the count), else -1
    seg_parent: list[int]  # per segment
    sum_position: list[int]  # per segment: position of its <sum>, -1 if truncated away
    seg_edges: list[tuple[int, int]] = field(default_factory=list)  # (caller seg, callee seg), variant E
    n_truncated: int = 0  # tokens dropped because the sequence exceeded max_len

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "kind": self.kind,
            "segment": self.segment,
            "is_header": [int(h) for h in self.is_header],
            "raw_index": self.raw_index,
            "seg_parent": self.seg_parent,
            "sum_position": self.sum_position,
            "seg_edges": [list(e) for e in self.seg_edges],
            "n_truncated": self.n_truncated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(
            input_ids=list(d["input_ids"]),
            labels=list(d["labels"]),
            kind=list(d["kind"]),
            segment=list(d["segment"]),
            is_header=[bool(h) for h in d["is_header"]],
            raw_index=list(d["raw_index"]),
            seg_parent=list(d["seg_parent"]),
            sum_position=list(d["sum_position"]),
            seg_edges=[tuple(e) for e in d.get("seg_edges", [])],
            n_truncated=int(d.get("n_truncated", 0)),
        )


def insert_sums(
    raw_ids: list[int],
    seg: Segmentation | None,
    bos_id: int,
    sum_id: int,
    pad_id: int = 0,
    max_len: int | None = None,
    pad_to: int | None = None,
    suffix_ids: list[int] | None = None,
    suffix_has_labels: bool = True,
    seg_edges: list[tuple[int, int]] | None = None,
) -> Example:
    """Build one example.

    Args:
        raw_ids: raw token ids (no BOS).
        seg: segmentation of ``raw_ids``; ``None`` means no ``<sum>`` tokens
            (variant A). Its ``token_segment`` must have ``len(raw_ids)`` entries.
        max_len: truncate the final sequence to this many positions.
        pad_to: right-pad with ``pad_id`` up to this length.
        suffix_ids: extra raw tokens appended after everything (segment -1).
        seg_edges: optional (caller, callee) segment pairs for variant E.
    """
    n = len(raw_ids)
    if seg is not None and len(seg.token_segment) != n:
        raise ValueError(f"segmentation has {len(seg.token_segment)} tokens, raw_ids has {n}")

    ids = [bos_id]
    kind = [KIND_BOS]
    segment = [-1]
    is_header = [False]
    raw_index = [-1]
    labels = [IGNORE_INDEX]

    n_segs = len(seg.segments) if seg is not None else 0
    sum_position = [-1] * n_segs
    order = sorted(seg.order, key=lambda s: seg.segments[s].insert_before) if seg is not None else []
    # sorted() is stable, so sums that share an insertion point keep emission order.
    ptr = 0

    def emit_sums_before(t: int) -> None:
        nonlocal ptr
        while ptr < len(order) and seg.segments[order[ptr]].insert_before <= t:
            s = order[ptr]
            sum_position[s] = len(ids)
            ids.append(sum_id)
            kind.append(KIND_SUM)
            segment.append(s)
            is_header.append(False)
            raw_index.append(-1)
            labels.append(IGNORE_INDEX)
            ptr += 1

    for t in range(n):
        emit_sums_before(t)
        ids.append(raw_ids[t])
        kind.append(KIND_RAW)
        segment.append(seg.token_segment[t] if seg is not None else -1)
        is_header.append(bool(seg.token_is_header[t]) if seg is not None else False)
        raw_index.append(t)
        labels.append(raw_ids[t])
    emit_sums_before(n)

    for k, tok in enumerate(suffix_ids or []):
        ids.append(tok)
        kind.append(KIND_RAW)
        segment.append(-1)
        is_header.append(False)
        raw_index.append(n + k)
        labels.append(tok if suffix_has_labels else IGNORE_INDEX)

    n_truncated = 0
    if max_len is not None and len(ids) > max_len:
        n_truncated = len(ids) - max_len
        for arr in (ids, kind, segment, is_header, raw_index, labels):
            del arr[max_len:]
        sum_position = [p if p < max_len else -1 for p in sum_position]

    if pad_to is not None and len(ids) < pad_to:
        k = pad_to - len(ids)
        ids += [pad_id] * k
        kind += [KIND_PAD] * k
        segment += [-1] * k
        is_header += [False] * k
        raw_index += [-1] * k
        labels += [IGNORE_INDEX] * k

    seg_parent = [s.parent for s in seg.segments] if seg is not None else []
    return Example(
        input_ids=ids,
        labels=labels,
        kind=kind,
        segment=segment,
        is_header=is_header,
        raw_index=raw_index,
        seg_parent=seg_parent,
        sum_position=sum_position,
        seg_edges=list(seg_edges or []),
        n_truncated=n_truncated,
    )
