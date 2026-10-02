"""
Matching sample names to the BAM files of their sequencing runs.

The BAM files of the runs are given as a listing of an object store (URIs) or
as file paths. A sample matches a BAM when the file name without ".bam" is the
sample name (case ignored), or when it is the sample name followed by a
separator and a suffix, as for a run written in several chunks
(SAMPLE_0.bam, SAMPLE_1.bam). Anything looser, such as the sample name
appearing elsewhere in the path, is not accepted automatically but listed for
review, because short names are often contained in longer ones.

The pieces of one run are written into one folder. A file named like a piece
but lying in another folder than the sample's own file, or pieces spread over
several folders when there is no such file, belong to other runs, such as a
dilution series named SAMPLE_10, SAMPLE_20: they are listed for review rather
than pooled with the sample.
"""

import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

SEPARATORS = "_.-"


def read_names(path: str) -> List[str]:
    names = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = re.sub(r"\.csv$", "", s.split("\t")[0].split(",")[0].strip(), flags=re.IGNORECASE)
            if s:
                names.append(s)
    return list(dict.fromkeys(names))


def read_listing(path: str) -> List[Tuple[str, int]]:
    """Lines of 'size<TAB>uri', 'uri' alone, or the output of an 's3 ls' style listing; returns (uri, size)."""
    out = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            uri = next((p for p in parts if p.lower().endswith(".bam")), None)
            if uri is None:
                continue
            size = next((int(p) for p in parts if p.isdigit()), -1)
            out.append((uri, size))
    return out


def stem(uri: str) -> str:
    return re.sub(r"\.bam$", "", PurePosixPath(uri).name, flags=re.IGNORECASE)


def folder(uri: str) -> str:
    return str(PurePosixPath(uri).parent)


def match_samples(names: List[str], listing: List[Tuple[str, int]]):
    """
    Returns (matched, review, unmatched):
      matched   : {sample: [(uri, size, 'exact' or 'chunk')]}
      review    : {sample: [(uri, size, reason)]}: 'contains' for a name found only inside a longer path, 'piece in
                  another folder' and 'pieces in several folders' for files of other runs (a sample can be both
                  matched and listed here), 'shared' for a BAM claimed by more than one sample
      unmatched : [sample]
    A BAM claimed by more than one sample is moved to review for all of them.
    """
    by_stem: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for uri, size in listing:
        by_stem[stem(uri).upper()].append((uri, size))
    matched: Dict[str, List[Tuple[str, int, str]]] = {}
    review: Dict[str, List[Tuple[str, int, str]]] = {}
    for name in names:
        key = name.upper()
        exact = [(u, s, "exact") for u, s in by_stem.get(key, [])]
        chunks = [(u, s, "chunk") for st, items in by_stem.items()
                  if len(st) > len(key) and st.startswith(key) and st[len(key)] in SEPARATORS
                  for u, s in items]
        if exact or chunks:
            # pieces of one run share its folder; files named like pieces elsewhere are other runs (see above)
            if exact:
                folders = {folder(u) for u, _, _ in exact}
                elsewhere = [(u, s, "piece in another folder") for u, s, _ in chunks if folder(u) not in folders]
                chunks = [c for c in chunks if folder(c[0]) in folders]
            elif len({folder(u) for u, _, _ in chunks}) > 1:
                review[name] = [(u, s, "pieces in several folders") for u, s, _ in chunks]
                continue
            else:
                elsewhere = []
            matched[name] = exact + chunks
            if elsewhere:
                review[name] = elsewhere
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        contains = [(u, s, "contains") for u, s in listing if pattern.search(u)]
        if contains:
            review[name] = contains
    claimed = defaultdict(list)
    for name, items in matched.items():
        for u, _, _ in items:
            claimed[u].append(name)
    shared = {u for u, owners in claimed.items() if len(owners) > 1}
    for name in list(matched):
        if any(u in shared for u, _, _ in matched[name]):
            review[name] = [(u, s, "shared" if u in shared else t) for u, s, t in matched.pop(name)] \
                + review.get(name, [])
    unmatched = [n for n in names if n not in matched and n not in review]
    return matched, review, unmatched
