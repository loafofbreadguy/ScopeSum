"""Paper figures: the architecture diagram (draw.io + PNG from one spec) and the result charts.

Run from the repo root:  python paper/figures/make_figures.py
Writes paper/figures/{fig_insertion,fig_layers,fig_mask,fig_attention}.{drawio,png}, results.pdf and kv.pdf.
The mask in fig_mask is computed with scopesum's own mask code.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from scopesum.data import build_example  # noqa: E402
from scopesum.masks import MaskMeta, build_sliding_mask  # noqa: E402
from scopesum.sum_insertion import KIND_BOS, KIND_PAD, KIND_RAW, KIND_SUM  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # chart colours (validated palette); diagrams use the greys below
INK, INK2, MUTED, GRID, SURFACE = "#000000", "#4d4d4d", "#8a8a8a", "#e5e5e5", "#ffffff"
BLUE_SOFT, ORANGE_SOFT, GRAY_SOFT, HIDDEN = "#e6e6e6", "#e6e6e6", "#efefef", "#f5f5f5"
RED = "#e34948"


# ---------------------------------------------------------------------------
# a tiny shape spec rendered twice: draw.io XML and matplotlib
# ---------------------------------------------------------------------------


class Canvas:
    def __init__(self, width, height):
        self.w, self.h, self.items = width, height, []

    def rect(self, x, y, w, h, fill, stroke="none", text="", fs=11, color=INK, mono=False, bold=False,
             rounded=True, dashed=False, align="center", lw=1.0):
        self.items.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, text=text, fs=fs, color=color,
                                        mono=mono, bold=bold, rounded=rounded, dashed=dashed, align=align, lw=lw)))

    def text(self, x, y, w, h, text, fs=11, color=INK, mono=False, bold=False, align="left"):
        self.rect(x, y, w, h, "none", "none", text, fs, color, mono, bold, rounded=False, align=align)

    def line(self, pts, color=INK2, lw=1.2, dashed=False, arrow=True):
        self.items.append(("line", dict(pts=pts, color=color, lw=lw, dashed=dashed, arrow=arrow)))

    # -- draw.io ---------------------------------------------------------------
    def to_drawio(self, path):
        cells, n = [], 2
        for kind, a in self.items:
            n += 1
            if kind == "rect":
                font = "Courier New" if a["mono"] else "Helvetica"
                style = (f"rounded={int(a['rounded'])};arcSize=12;whiteSpace=wrap;html=0;fillColor={a['fill']};"
                         f"strokeColor={a['stroke']};fontColor={a['color']};fontSize={a['fs']};fontFamily={font};"
                         f"align={a['align']};verticalAlign=middle;spacingLeft=4;spacingRight=4;"
                         f"strokeWidth={a['lw']};fontStyle={1 if a['bold'] else 0};" + ("dashed=1;" if a["dashed"] else ""))
                cells.append(f'<mxCell id="c{n}" value="{escape(a["text"], {chr(10): "&#xa;", chr(34): "&quot;"})}" '
                             f'style="{style}" vertex="1" parent="1"><mxGeometry x="{a["x"]}" y="{a["y"]}" '
                             f'width="{a["w"]}" height="{a["h"]}" as="geometry"/></mxCell>')
            else:
                (x0, y0), (x1, y1), mids = a["pts"][0], a["pts"][-1], a["pts"][1:-1]
                style = (f"endArrow={'classic' if a['arrow'] else 'none'};endSize=6;html=1;rounded=0;"
                         f"strokeColor={a['color']};strokeWidth={a['lw']};" + ("dashed=1;" if a["dashed"] else ""))
                way = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in mids)
                cells.append(f'<mxCell id="c{n}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                             f'<mxPoint x="{x0}" y="{y0}" as="sourcePoint"/><mxPoint x="{x1}" y="{y1}" as="targetPoint"/>'
                             + (f'<Array as="points">{way}</Array>' if mids else "") + "</mxGeometry></mxCell>")
        xml = (f'<mxfile host="scopesum"><diagram name="architecture" id="arch"><mxGraphModel dx="{self.w}" dy="{self.h}" '
               f'grid="0" gridSize="10" guides="1" page="1" pageWidth="{self.w}" pageHeight="{self.h}" background="#ffffff">'
               '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
        with open(path, "w") as f:
            f.write(xml)

    # -- matplotlib --------------------------------------------------------------
    def to_png(self, path, dpi=220):
        fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        pt = 72 / 100  # draw.io px -> points at 100 px per inch
        for kind, a in self.items:
            if kind == "rect":
                if a["fill"] != "none" or a["stroke"] != "none":
                    box = (FancyBboxPatch((a["x"], a["y"]), a["w"], a["h"], boxstyle=f"round,pad=0,rounding_size={min(a['w'], a['h']) * 0.12}")
                           if a["rounded"] else Rectangle((a["x"], a["y"]), a["w"], a["h"]))
                    box.set_facecolor(a["fill"] if a["fill"] != "none" else "none")
                    box.set_edgecolor(a["stroke"] if a["stroke"] != "none" else "none")
                    box.set_linewidth(a["lw"] * pt)
                    if a["dashed"]:
                        box.set_linestyle((0, (4, 3)))
                    ax.add_patch(box)
                if a["text"]:
                    ha = {"left": "left", "center": "center", "right": "right"}[a["align"]]
                    tx = {"left": a["x"] + 4, "center": a["x"] + a["w"] / 2, "right": a["x"] + a["w"] - 4}[a["align"]]
                    ax.text(tx, a["y"] + a["h"] / 2, a["text"], ha=ha, va="center", fontsize=a["fs"] * pt, color=a["color"],
                            family="monospace" if a["mono"] else "sans-serif", fontweight="bold" if a["bold"] else "normal",
                            linespacing=1.25)
            else:
                xs, ys = zip(*a["pts"])
                ls = (0, (4, 3)) if a["dashed"] else "-"
                ax.plot(xs, ys, color=a["color"], lw=a["lw"] * pt * 1.4, ls=ls, solid_capstyle="round")
                if a["arrow"]:
                    ax.annotate("", xy=a["pts"][-1], xytext=a["pts"][-2],
                                arrowprops=dict(arrowstyle="-|>", color=a["color"], lw=a["lw"] * pt * 1.4, mutation_scale=9,
                                                shrinkA=0, shrinkB=0))
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        from PIL import Image  # plain RGB: an alpha channel renders as a grey box in some PDF viewers

        Image.open(path).convert("RGB").save(path)


# ---------------------------------------------------------------------------
# the example: a class with two methods, then module code
# ---------------------------------------------------------------------------

CODE = "class C:\n  def f(x):\n    return x\n  def g(y):\n    return f(y)\nC().g(1)\n"
TOKEN_RE = re.compile(r"\s*(\w+|[^\w\s])|\s+")
W_DEMO = 6


def demo_example():
    offs = [(m.start(), m.end()) for m in TOKEN_RE.finditer(CODE)]
    assert sum(b - a for a, b in offs) == len(CODE)
    win = {"text": CODE, "offsets": [list(o) for o in offs], "raw_ids": list(range(10, 10 + len(offs)))}
    ex = build_example(win, "scope", 2, 6, 0, len(offs) + 16)
    L = int(sum(1 for k in ex.kind if k != KIND_PAD))
    mask = build_sliding_mask("C", MaskMeta.from_example(ex), window=W_DEMO)[:L, :L]
    labels, r = [], 0
    for k in ex.kind[:L]:
        if k == KIND_BOS:
            labels.append("BOS")
        elif k == KIND_SUM:
            labels.append("<sum>")
        else:
            a, b = offs[r]
            labels.append(CODE[a:b].strip() or "\\n")
            r += 1
    return np.asarray(ex.kind[:L]), mask, labels


# one border style for every box; greys only, no black fills
EDGE, LW = "#555555", 1.0
F_RAW, F_SUM, F_BOS, F_WHITE, F_PANEL = "#ffffff", "#bdbdbd", "#e3e3e3", "#ffffff", "#f2f2f2"
M_RAW, M_SUM, M_BOS, M_HIDDEN = "#c4c4c4", "#6f6f6f", "#9c9c9c", "#f4f4f4"  # mask cells (no borders)


def _save(c, name):
    c.to_drawio(os.path.join(OUT, f"{name}.drawio"))
    c.to_png(os.path.join(OUT, f"{name}.png"))


def fig_insertion():
    """Figure: scopes of a small example and the sequence with <sum> inserted."""
    c = Canvas(770, 390)
    lines = CODE.rstrip("\n").split("\n")
    c.rect(20, 20, 250, 150, F_WHITE, EDGE, rounded=False, lw=LW)
    for i, ln in enumerate(lines):
        c.text(30, 28 + i * 23, 230, 22, ln, fs=13, mono=True)
    for name, l0, l1, x in [("class C", 0, 4, 300), ("f", 1, 2, 430), ("g", 3, 4, 430), ("module", 5, 5, 300)]:
        c.line([(x, 30 + l0 * 23), (x, 48 + l1 * 23)], color=INK2, lw=2, arrow=False)
        c.text(x + 6, 28 + l0 * 23, 120, 22, f"scope {name}", fs=11, color=INK2)
    c.text(20, 200, 480, 24, "sequence fed to the model (raw tokens + <sum>)", fs=12, color=INK2)
    strip = [("BOS", F_BOS, 36), ("class C:", F_RAW, 62), ("def f(x): return x", F_RAW, 124), ("<sum> f", F_SUM, 56),
             ("def g(y): return f(y)", F_RAW, 142), ("<sum> g", F_SUM, 56), ("<sum> C", F_SUM, 56),
             ("C().g(1)", F_RAW, 66), ("<sum> mod", F_SUM, 72)]
    x, y, h, pos = 20, 270, 34, {}
    for text, fill, w in strip:
        is_sum = fill == F_SUM
        c.rect(x, y, w, h, fill, EDGE, text, fs=10, color=INK, mono=not is_sum, bold=is_sum, rounded=False, lw=LW)
        pos[text] = (x, w)
        x += w + 4

    def mid(t):
        return pos[t][0] + pos[t][1] / 2

    for s, src in [("<sum> f", "def f(x): return x"), ("<sum> g", "def g(y): return f(y)")]:
        c.line([(mid(s), y), (mid(s), y - 14), (mid(src), y - 14), (mid(src), y)], color=INK2, lw=1.3)
    c.line([(mid("<sum> C"), y), (mid("<sum> C"), y - 30), (mid("<sum> f"), y - 30), (mid("<sum> f"), y - 1)], color=INK2, lw=1.3)
    c.line([(mid("<sum> C") - 6, y), (mid("<sum> C") - 6, y - 22), (mid("<sum> g"), y - 22), (mid("<sum> g"), y - 1)], color=INK2, lw=1.3)
    q, yb = mid("C().g(1)"), y + h
    c.line([(q, yb), (q, yb + 40), (pos["def g(y): return f(y)"][0] + 90, yb + 40), (pos["def g(y): return f(y)"][0] + 90, yb + 1)], color=INK2, lw=1.3)
    for s, dy in [("<sum> f", 58), ("<sum> g", 50), ("<sum> C", 26)]:
        c.line([(q - 8, yb), (q - 8, yb + dy), (mid(s), yb + dy), (mid(s), yb + 1)], color=INK2, lw=1.3, dashed=True)
    _save(c, "fig_insertion")


def fig_layers():
    """Figure: a compact stack of the five layers of a block; the front layer shows its contents."""
    c = Canvas(560, 400)
    side, off, x0, y0 = 250, 9, 60, 60
    for i in range(4, 0, -1):  # layers 5 (full) .. 2, back to front
        c.rect(x0 + i * off, y0 - i * off, side, side, F_BOS if i == 4 else F_WHITE, EDGE, rounded=False, lw=LW)
    c.rect(x0, y0, side, side, F_WHITE, EDGE, rounded=False, lw=LW)
    c.text(x0 + 6, y0 + 4, side - 12, 22, "layer 1: sliding window", fs=11, bold=True)
    inner = [("RMSNorm", 26), ("self-attention\nQ, K, V, O projections (+ LoRA)\nmask: ours in sliding layers,\nplain causal in full layers", 70),
             ("RMSNorm", 26), ("MLP: gate, up, down (+ LoRA)", 30), ("+ per-layer embedding (Gemma 4)", 26)]
    iy = y0 + 32
    for txt, ih in inner:
        c.rect(x0 + 14, iy, side - 28, ih, F_PANEL if ih > 26 else F_WHITE, EDGE, txt, fs=10, rounded=False, lw=LW)
        iy += ih + 8
    lx = x0 + 4 * off + side + 14
    c.text(lx, y0 - 4 * off - 6, 200, 40, "layer 5: full attention\n(plain causal)", fs=11, bold=True)
    c.text(lx, y0 - 4 * off + 40, 200, 22, "layers 2-4: sliding window", fs=11)
    c.text(lx, y0 + 100, 200, 70, "× 7 blocks = 35 layers\nlocal window W = 512", fs=12, bold=True)
    c.line([(x0 - 30, y0 + side - 10), (x0 - 30, y0 - 4 * off + 6), (x0 + 4 * off - 1, y0 - 4 * off + 6)], color=INK2, lw=2, dashed=True)
    c.text(x0 - 34, y0 + side + 10, 480, 22, "dashed: far raw tokens go straight to the full-attention layer", fs=11, color=INK2)
    c.text(x0 - 34, y0 + side + 34, 480, 44, "That layer sees every token, so distant code is read directly\nand the model does not need <sum> to reach it.", fs=11, color=INK2)
    _save(c, "fig_layers")


def fig_mask():
    """Figure: the real sliding-layer mask our code builds for the example."""
    kind, mask, _ = demo_example()
    M_RAW, M_SUM, M_BOS, M_HIDDEN = BLUE, ORANGE, "#9c9c9c", "#f4f4f4"  # chart colours: raw blue, <sum> orange
    c = Canvas(520, 500)
    cx, cy, cell = 40, 30, 11
    L = len(kind)
    for qi in range(L):
        for ki in range(qi + 1):
            fill = M_HIDDEN if not mask[qi, ki] else M_SUM if kind[ki] == KIND_SUM else M_BOS if kind[ki] == KIND_BOS else M_RAW
            c.rect(cx + 30 + ki * cell, cy + 20 + qi * cell, cell - 1, cell - 1, fill, rounded=False)
    for i in range(L):
        col = M_SUM if kind[i] == KIND_SUM else M_BOS if kind[i] == KIND_BOS else "#c9ddf5"
        c.rect(cx + 30 + i * cell, cy + 8, cell - 1, 8, col, rounded=False)
        c.rect(cx + 18, cy + 20 + i * cell, 8, cell - 1, col, rounded=False)
    c.text(cx + 30, cy - 12, L * cell, 18, "key →", fs=11, color=INK2, align="center")
    c.text(cx - 26, cy + 20 + L * cell / 2 - 10, 44, 20, "query ↓", fs=10, color=INK2)
    ly = cy + 30 + L * cell
    for fill, lab, dx in [(M_RAW, "raw key", 0), (M_SUM, "<sum> key", 100), (M_BOS, "BOS", 210), (M_HIDDEN, "masked", 270)]:
        c.rect(cx + 30 + dx, ly, 14, 14, fill, EDGE, rounded=False, lw=LW)
        c.text(cx + 48 + dx, ly - 4, 100, 22, lab, fs=11, color=INK2)
    _save(c, "fig_mask")


def fig_attention():
    """Figure: measured attention shares of raw queries by key type."""
    attn = json.load(open(os.path.join(ROOT, "results/diag/C.json")))["attention"]
    M_SUM = ORANGE  # <sum> stands out in orange; everything else stays grey
    c = Canvas(600, 400)
    dx0 = 0
    c.text(dx0 + 20, 10, 560, 22, "variant C, W = 512, share of attention by key type, averaged over layers", fs=11, color=INK2)
    sl, fl = attn["sliding_attention"], attn["full_attention"]
    by, bw, bh = 80, 540, 34
    for r, (name, segs) in enumerate([
        ("sliding layers", [("recent raw", F_BOS, 1 - sl["bos"] - sl["sum"]), ("BOS", M_BOS, sl["bos"]), ("<sum>", M_SUM, sl["sum"])]),
        ("full layers", [("recent raw", F_BOS, 1 - fl["bos"] - fl["sum"] - fl["far_raw"]), ("far raw", F_WHITE, fl["far_raw"]),
                         ("BOS", M_BOS, fl["bos"]), ("<sum>", M_SUM, fl["sum"])])]):
        yy = by + r * 100
        c.text(dx0 + 20, yy - 24, 200, 20, name, fs=12, bold=True)
        x = dx0 + 20
        for lab, fill, share in segs:
            w = bw * share
            c.rect(x, yy, max(w, 1.5), bh, fill, EDGE if w > 3 else "none", f"{lab} {100 * share:.0f}%" if w > 70 else (f"{100 * share:.0f}%" if w > 30 else ""),
                   fs=10, color=INK, rounded=False, lw=LW)
            x += w
        c.line([(x - 1, yy + bh), (x - 1, yy + bh + 12), (x - 60, yy + bh + 12)], color=INK2, lw=1, arrow=False)
        c.text(x - 230, yy + bh + 2, 168, 20, f"<sum>: {100 * segs[-1][2]:.1f}% of attention", fs=10, color=INK, align="right")
    for fill, lab, dx in [(F_BOS, "recent raw (last W)", 0), (F_WHITE, "far raw (> W back)", 170), (M_BOS, "BOS", 330), (M_SUM, "<sum>", 400)]:
        c.rect(dx0 + 20 + dx, by + 180, 14, 14, fill, EDGE, rounded=False, lw=LW)
        c.text(dx0 + 38 + dx, by + 176, 150, 22, lab, fs=11, color=INK2)
    c.text(dx0 + 20, by + 220, 590, 80,
           "The <sum> tokens get under 1% of attention, while the full layers spend\n35% on raw tokens more than W back. Hiding every <sum> at test time\nchanges long-range loss by only −0.004: the summaries are not used.", fs=11, color=INK2)
    _save(c, "fig_attention")


def architecture():
    fig_insertion()
    fig_layers()
    fig_mask()
    fig_attention()


# ---------------------------------------------------------------------------
# result charts
# ---------------------------------------------------------------------------


def _style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def results():
    main = _read_csv(os.path.join(ROOT, "results/eval/main_results.csv"))
    probes = _read_csv(os.path.join(ROOT, "results/eval/probe_results.csv"))
    diag = {v: json.load(open(os.path.join(ROOT, f"results/diag/{v}.json"))) for v in "BCDE"}
    diag128 = {v: json.load(open(os.path.join(ROOT, f"results/w128/diag/{v}.json"))) for v in "FG"}

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.titlesize": 9, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    # (a) long-range loss, dot plot
    ax = axs[0, 0]
    names = [r["variant"].replace('"', "").split()[0] for r in main]
    vals = [float(r["loss_long_range"]) for r in main]
    ypos = np.arange(len(names))[::-1]
    ax.axvline(vals[0], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.hlines(ypos, 0.30, vals, color=GRID, lw=1)
    ax.scatter(vals, ypos, s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    for v, yv in zip(vals, ypos):
        ax.text(v + 0.0012, yv, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(ypos, names)
    ax.set_xlim(0.30, 0.365)
    ax.set_xlabel("long-range loss (nats, lower is better)", color=INK2)
    ax.set_title("(a)", color=INK)
    ax.set_ylim(-1.1, len(names) - 0.5)
    ax.text(vals[0] + 0.001, -0.85, "baseline A", fontsize=7, color=MUTED, ha="left")
    _style(ax)

    # (b) delta when <sum> is hidden
    ax = axs[0, 1]
    rows = [(f"{v}  W=512", diag[v]) for v in "BCDE"] + [(f"{v}  W=128", diag128[v]) for v in "FG"]
    d = [r["loss"]["hide_all"]["long_range"]["loss"] - r["loss"]["normal"]["long_range"]["loss"] for _, r in rows]
    yp = np.arange(len(rows))[::-1]
    ax.axvspan(-0.005, 0.005, color=GRAY_SOFT, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.barh(yp, d, height=0.55, color=BLUE, edgecolor="white", linewidth=1)
    for v, yv in zip(d, yp):
        ax.text(v + (0.0004 if v >= 0 else -0.0004), yv, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.set_yticks(yp, [n for n, _ in rows])
    ax.set_xlim(-0.012, 0.012)
    ax.set_xlabel("Δ long-range loss when every <sum> is hidden\n(> 0 means the summaries help)", color=INK2)
    ax.set_ylim(-1.1, len(rows) - 0.5)
    ax.text(0.0055, -0.85, "noise band ±0.005", fontsize=7, color=MUTED, ha="left")
    ax.set_title("(b)", color=INK)
    _style(ax)

    # (c) attention shares
    ax = axs[1, 0]
    groups = [("C  W=512", diag["C"]), ("F  W=128", diag128["F"]), ("G  W=128", diag128["G"])]
    s_sum = [100 * r["attention"]["sliding_attention"]["sum"] for _, r in groups]
    s_far = [100 * r["attention"]["full_attention"]["far_raw"] for _, r in groups]
    yp = np.arange(len(groups))[::-1]
    ax.barh(yp + 0.17, s_far, height=0.32, color=BLUE, edgecolor="white", linewidth=1, label="far raw tokens (full layers)")
    ax.barh(yp - 0.17, s_sum, height=0.32, color=ORANGE, edgecolor="white", linewidth=1, label="<sum> tokens (sliding layers)")
    for v, yv in zip(s_far, yp):
        ax.text(v + 0.8, yv + 0.17, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    for v, yv in zip(s_sum, yp):
        ax.text(v + 0.8, yv - 0.17, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(yp, [n for n, _ in groups])
    ax.set_xlim(0, 70)
    ax.set_xlabel("share of raw-query attention (%)", color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, fontsize=7, frameon=False, labelcolor=INK2)
    ax.set_title("(c)", color=INK)
    _style(ax)

    # (d) probes
    ax = axs[1, 1]
    tasks = [("returns_value", "returns value"), ("n_args_bucket", "# args"), ("name_reused", "name reused")]

    def acc(v, t, feat="sum"):
        return next(float(r["accuracy"]) for r in probes if r["variant"] == v and r["task"] == t
                    and r.get("feature", "") in (feat, "" if feat == "sum" else feat) and r.get("subset", "") in ("all", ""))

    series = [("G <sum> (fixed chunks)", AQUA, lambda t: acc("G", t)), ("F <sum> (scopes)", ORANGE, lambda t: acc("F", t)),
              ("raw-mean of the function (F)", BLUE, lambda t: acc("F", t, "raw_mean"))]
    xp = np.arange(len(tasks))
    for i, (lab, col, f) in enumerate(series):
        vals = [f(t) for t, _ in tasks]
        ax.bar(xp + (i - 1) * 0.26, vals, width=0.24, color=col, edgecolor="white", linewidth=1, label=lab)
        for x0, v in zip(xp + (i - 1) * 0.26, vals):
            ax.text(x0, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
    maj = [next(float(r["majority"]) for r in probes if r["variant"] == "F" and r["task"] == t) for t, _ in tasks]
    for x0, m in zip(xp, maj):
        ax.hlines(m, x0 - 0.42, x0 + 0.42, color=INK, lw=1, ls=(0, (2, 1.5)))
    ax.plot([], [], color=INK, lw=1, ls=(0, (2, 1.5)), label="majority class")
    ax.set_xticks(xp, [n for _, n in tasks])
    ax.set_ylim(0.3, 1.18)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("probe accuracy", color=INK2)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False, labelcolor=INK2, ncol=1, bbox_to_anchor=(1.0, 1.02))
    ax.set_title("(d)", color=INK)
    _style(ax, "y")

    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    fig.savefig(os.path.join(OUT, "results.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # KV memory: mean and peak live entries per variant
    mem = _read_csv(os.path.join(ROOT, "results/eval/memory_results.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
    v = [r["variant"] for r in mem]
    yp = np.arange(len(v))[::-1]
    mean = [float(r["live_mean"]) for r in mem]
    peak = [float(r["live_max"]) for r in mem]
    ax.barh(yp + 0.18, peak, height=0.34, color=AQUA, edgecolor="white", linewidth=1, label="peak")
    ax.barh(yp - 0.18, mean, height=0.34, color=BLUE, edgecolor="white", linewidth=1, label="mean")
    for p_, yv in zip(peak, yp):
        ax.text(p_ + 15, yv + 0.18, f"{p_:.0f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(512, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(512, yp[0] + 0.75, "W = 512", fontsize=6.5, color=MUTED, ha="center")
    ax.set_yticks(yp, v)
    ax.set_xlim(0, 1350)
    ax.set_xlabel("live KV entries per sliding layer", color=INK2)
    ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=INK2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv.pdf"), facecolor="white")
    plt.close(fig)


def training_curves():
    """Training loss per logged step for the main runs (seed 1)."""
    runs = [("A, standard mask", "results/train/variant_A", BLUE),
            ("C, standard mask", "results/train/variant_C", "#eda100"),
            ("F, restrict_full", "results/restrict_full/train/variant_F", ORANGE),
            ("G, restrict_full", "results/restrict_full/train/variant_G", AQUA),
            ("F, restrict_full + signatures", "results/restrict_full_signatures/train/variant_F", "#e87ba4"),
            ("S, SepLLM-style, restrict_full", "results/seeds/Srf1", "#6250d6")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    for lab, d, col in runs:
        rows = _read_csv(os.path.join(ROOT, d, "train_log.csv"))
        ax.plot([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows], color=col, lw=1.6, label=lab)
    ax.set_xlabel("training step", color=INK2)
    ax.set_ylabel("training loss (nats)", color=INK2)
    ax.legend(fontsize=6.5, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "train_curves.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def tradeoff():
    """Long-range loss against projected KV cache at 128k tokens, mean over seeds (bars: range)."""
    def lr(path):
        return json.load(open(os.path.join(ROOT, path)))["loss"]["normal"]["long_range"]["loss"]

    n, w = 131072, 512
    rates = json.load(open(os.path.join(ROOT, "results/long_context/long_context_memory.json")))["sum_rates"]
    sig_share, sep_share = rates["signatures"], rates["separators"]  # measured on all training windows

    def mb(kept):  # 12 sliding layers x 1 KB + 3 full layers x 2 KB per cached token (fp16)
        return kept * (12 * 1024 + 3 * 2048) / 2**20

    pts = [
        ("A, full attention", mb(n) * 0 + 774.0, [lr("results/diag/A.json"), lr("results/seeds/A2/A.json")], BLUE),
        ("F, restrict_full", mb(w + rates["F"] * n), [lr("results/restrict_full/diag/F.json"), lr("results/seeds/Frf2/F.json")], ORANGE),
        ("G, restrict_full", mb(w + rates["G"] * n), [lr("results/restrict_full/diag/G.json"), lr("results/seeds/Grf2/G.json")], AQUA),
        ("F + signatures", mb(w + (rates["F"] + sig_share) * n), [lr("results/restrict_full_signatures/diag/F.json"), lr("results/seeds/Fsig2/F.json")], "#e87ba4"),
        ("S, SepLLM-style", mb(w + sep_share * n), [lr("results/seeds/Srf1/S.json"), lr("results/seeds/Srf2/S.json")], "#6250d6"),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=200)
    for lab, x, ys, col in pts:
        m = float(np.mean(ys))
        ax.errorbar([x], [m], yerr=[[m - min(ys)], [max(ys) - m]], fmt="o", ms=6, color=col, mec="white", mew=1, capsize=3, lw=1.2, label=lab)
        ax.annotate(f"{m:.3f}", (x, m), xytext=(6, -2), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(12, 2000)
    ax.set_xlabel("projected KV cache at 128k tokens (MB, log)", color=INK2)
    ax.set_ylabel("long-range loss (nats)", color=INK2)
    ax.legend(fontsize=6.3, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def kv_length():
    """Projected KV cache per sequence against context length (results/long_context)."""
    rows = json.load(open(os.path.join(ROOT, "results/long_context/long_context_memory.json")))["rows"]
    series = [("A: Gemma default (no <sum>)", "A, full attention", BLUE, (0, (4, 2)), 3),
              ("F: standard mask (full layers see all)", "F, standard mask", AQUA, "-", 2),
              ("S: restrict_full (SepLLM-style separators)", "S, SepLLM-style", "#6250d6", "-", 2),
              ("F + signatures: restrict_full", "F + signatures", "#e87ba4", "-", 2),
              ("F: restrict_full (every layer: last W + <sum>)", "F, restrict_full", ORANGE, "-", 2)]
    fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=200)
    for key, lab, col, ls, z in series:
        pts = [(r["N"], r["kv_mb"]) for r in rows if r["setting"] == key]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=1.6, ls=ls, marker="o", ms=3.5, mec="white", mew=0.6, label=lab, zorder=z)
        if lab != "F, standard mask":  # overlaps A at every length
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6, color=INK2, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2048, 8192, 32768, 131072], ["2k", "8k", "32k", "128k"])
    ax.set_xlim(1500, 260000)
    ax.set_xlabel("context length (raw tokens)", color=INK2)
    ax.set_ylabel("KV cache per sequence (MB)", color=INK2)
    ax.legend(fontsize=6.3, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv_vs_length.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def layer_attention():
    """Per layer: attention of raw queries on the tokens each variant keeps beyond the window."""
    runs = [("C, W = 512: <sum>", "results/diag/C.json", "sum", BLUE, "-"),
            ("F, W = 128: <sum>", "results/w128/diag/F.json", "sum", AQUA, "-"),
            ("F, restrict_full: <sum>", "results/restrict_full/diag/F.json", "sum", ORANGE, "-"),
            ("F + signatures: far signatures", "results/restrict_full_signatures/diag/F.json", "far_raw", "#e87ba4", (0, (3, 1.5))),
            ("S, SepLLM-style: far separators", "results/seeds/Srf1/S.json", "far_raw", "#6250d6", (0, (3, 1.5)))]
    fig, ax = plt.subplots(figsize=(3.6, 2.6), dpi=200)
    types = None
    for lab, path, key, col, ls in runs:
        per = sorted(json.load(open(os.path.join(ROOT, path)))["attention_per_layer"], key=lambda r: r["layer"])
        types = [r["type"] for r in per]
        ax.plot([r["layer"] for r in per], [100 * r[key] for r in per], color=col, lw=1.3, ls=ls, marker="o", ms=2.2, label=lab)
    for i, t in enumerate(types):
        if t == "full_attention":
            ax.axvspan(i - 0.5, i + 0.5, color=GRAY_SOFT, zorder=0, lw=0)
    ax.set_xlabel("layer (shaded: full attention)", color=INK2)
    ax.set_ylabel("attention on kept far tokens (%)", color=INK2)
    ax.set_xlim(-0.5, len(types) - 0.5)
    ax.legend(fontsize=6.0, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "layer_attention.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results()
    training_curves()
    tradeoff()
    kv_length()
    layer_attention()
    print("wrote", sorted(f for f in os.listdir(OUT) if not f.endswith(".py")))
