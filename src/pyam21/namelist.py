"""A small Fortran namelist reader for the SCM input.nml files.

Handles what those files use: groups ``&name ... /``, ``!`` comments, strings in
single or double quotes, logicals (.true., .t., T, ...), integers, reals (with d/e
exponents), comma- or blank-separated lists continued over lines, and repeat counts
``n*value``. Keys are lower-cased. A scalar is returned for one value, a list
otherwise.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"""
    '(?:[^']|'')*'            # single-quoted string
  | "(?:[^"]|"")*"            # double-quoted string
  | [^\s,='"!]+               # bare word / number / logical
  | =                         # assignment
""", re.X)


def _strip_comment(line: str) -> str:
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _value(tok: str):
    if tok[0] in "'\"":
        return tok[1:-1].replace(tok[0] * 2, tok[0])
    low = tok.lower()
    if low in (".true.", ".t.", "t", ".true", "true"):
        return True
    if low in (".false.", ".f.", "f", ".false", "false"):
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return tok


def parse_namelist(text: str) -> dict:
    groups: dict = {}
    body = "\n".join(_strip_comment(l) for l in text.splitlines())
    for m in re.finditer(r"&(\w+)(.*?)(?:^|\s)/", body, re.S | re.M):
        gname = m.group(1).lower()
        toks = _TOKEN.findall(m.group(2))
        entries: dict = {}
        key, vals = None, []
        i = 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i + 1] == "=":
                if key is not None:
                    entries[key] = vals
                key = toks[i].lower()
                vals = []
                i += 2
                continue
            t = toks[i]
            rep = re.fullmatch(r"(\d+)\*(.+)", t)
            if rep:
                vals.extend([_value(rep.group(2))] * int(rep.group(1)))
            else:
                vals.append(_value(t))
            i += 1
        if key is not None:
            entries[key] = vals
        groups[gname] = {k: (v[0] if len(v) == 1 else v) for k, v in entries.items()}
    return groups


def read_namelist(path: str) -> dict:
    with open(path) as f:
        return parse_namelist(f.read())
