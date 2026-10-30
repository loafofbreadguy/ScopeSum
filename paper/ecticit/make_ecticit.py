"""Build the ECTI-CIT review manuscript from paper/scopesum.tex.

Run from paper/ecticit:  python make_ecticit.py && pdflatex scopesum_ecticit.tex (twice)

ECTI-CIT format (review template, A4): two columns 0.2" apart, margins 1" top and
bottom and 0.8" left and right, Times 10 pt body with a 0.25" first-line indent,
20 pt bold title, bold abstract, keywords, "1. INTRODUCTION" section headings,
"Fig. 1:" captions below figures and "Table 1:" captions above tables (italic),
numbered references in citation order, and no author names (double-blind).
"""

import re

SRC, OUT = "../scopesum.tex", "scopesum_ecticit.tex"
WIDE_FIGS = {"fig:insert", "fig:results"}
COLUMN_FIGS = {"fig:mask", "fig:layers", "fig:kv", "fig:attn", "fig:kvlen", "fig:train", "fig:tradeoff", "fig:layerattn"}
COLUMN_TABLES = {"tab:setup"}
KEYWORDS = "Code language models, Context compression, Summary tokens, KV cache, Sliding-window attention, Negative result"

PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,top=1in,bottom=1in,left=0.8in,right=0.8in,columnsep=0.2in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{caption}
\captionsetup{labelsep=colon,font={small,it},labelfont={bf,it}}
\captionsetup[table]{position=top}
\renewcommand{\figurename}{Fig.}
\setlength{\parindent}{0.25in}
% let floats sit close to their text instead of drifting to the end
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.08}\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.9}\renewcommand{\dblfloatpagefraction}{0.75}
\setcounter{topnumber}{3}\setcounter{dbltopnumber}{3}\setcounter{totalnumber}{5}

\makeatletter
% ECTI-CIT headings: "1. INTRODUCTION" (bold capitals) and "2.1 Subsection" (bold)
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname\ifx#1section.\fi\quad}
\renewcommand{\section}{\@startsection{section}{1}{\z@}{12pt}{4pt}{\normalfont\normalsize\bfseries\MakeUppercase}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}{12pt}{4pt}{\normalfont\normalsize\bfseries}}
\renewcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}{6pt}{-0.6em}{\normalfont\normalsize\bfseries}}
\makeatother

\newcommand{\sumtok}{\texttt{<sum>}}
"""


def balanced(s: str, start: int) -> int:
    """Index just past the brace group that opens at s[start] == '{'."""
    depth = 0
    for i in range(start, len(s)):
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth == 0:
            return i + 1
    raise ValueError("unbalanced braces")


def body_of(src: str) -> tuple[str, str, str]:
    title = src[src.index("\\title{") + 7: balanced(src, src.index("\\title{") + 6) - 1]
    abstract = src[src.index("\\begin{abstract}") + 16: src.index("\\end{abstract}")].strip()
    body = src[src.index("\\end{abstract}") + 14: src.index("\\end{document}")]
    return title.replace("\\\\", " "), abstract, body


def convert_floats(body: str) -> str:
    def fix(m):
        env, inner = m.group(1), m.group(2)
        label = re.search(r"\\label\{([^}]+)\}", inner).group(1)
        if env == "table":
            # caption above the table
            c0 = inner.index("\\caption{")
            c1 = balanced(inner, c0 + 8)
            cap = inner[c0:c1]
            inner = inner[:c0] + inner[c1:]
            inner = inner.replace("\\centering", "\\centering\n" + cap, 1)
            star = "" if label in COLUMN_TABLES else "*"
            return f"\\begin{{table{star}}}[t]{inner}\\end{{table{star}}}"
        if label in WIDE_FIGS:
            inner = re.sub(r"width=[0-9.]*\\textwidth", r"width=\\textwidth", inner)
            return f"\\begin{{figure*}}[t]{inner}\\end{{figure*}}"
        if label in COLUMN_FIGS:
            inner = re.sub(r"width=[0-9.]*\\textwidth", r"width=\\columnwidth", inner)
            return f"\\begin{{figure}}[!htbp]{inner}\\end{{figure}}"
        raise ValueError(f"unplaced float {label}")

    body = re.sub(r"\\begin\{figure\}\[[a-z]+\](.*?)\\end\{figure\}", lambda m: fix(_m("figure", m.group(1))), body, flags=re.S)
    body = re.sub(r"\\begin\{table\}\[[a-z]+\](.*?)\\end\{table\}", lambda m: fix(_m("table", m.group(1))), body, flags=re.S)
    return body


class _m:
    def __init__(self, env, inner):
        self._g = (None, env, inner)

    def group(self, i):
        return self._g[i]


def order_bibliography(body: str) -> str:
    """Numbered references must follow the order of first citation."""
    b0, b1 = body.index("\\begin{thebibliography}"), body.index("\\end{thebibliography}")
    items = re.split(r"(?=\\bibitem)", body[b0:b1])
    head, items = items[0], items[1:]
    key = {re.match(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", it).group(1): it for it in items}
    cited = []
    for group in re.findall(r"\\cite[tp]?\*?\{([^}]+)\}", body[:b0]):
        for k in group.split(","):
            k = k.strip()
            if k not in cited:
                cited.append(k)
    missing = [k for k in key if k not in cited]
    assert not missing, f"references never cited: {missing}"
    ordered = "".join(key[k].rstrip() + "\n" for k in cited)
    return body[:b0] + head + ordered + body[b1:]


def main():
    src = open(SRC).read()
    title, abstract, body = body_of(src)
    body = convert_floats(body)
    body = order_bibliography(body)
    body = body.replace("\\begin{thebibliography}{99}", "\\section*{REFERENCES}\n\\begingroup\\renewcommand{\\section}[2]{}\\small\n\\begin{thebibliography}{99}")
    body = body.replace("\\end{thebibliography}", "\\end{thebibliography}\n\\endgroup")
    body = body.replace("figures/", "../figures/")
    front = rf"""
\begin{{document}}
\twocolumn[{{%
\begin{{center}}
{{\fontsize{{20}}{{24}}\selectfont\bfseries {title}\par}}
\vspace{{24pt}}
\end{{center}}
\noindent{{\bfseries ABSTRACT}}\par\vspace{{4pt}}
\noindent{{\bfseries\small {abstract}}}\par
\vspace{{12pt}}
\noindent{{\bfseries Keywords:}} {KEYWORDS}\par
\vspace{{12pt}}
}}]
"""
    open(OUT, "w").write(PREAMBLE + front + body + "\\end{document}\n")


if __name__ == "__main__":
    main()
