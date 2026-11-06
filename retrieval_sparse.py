"""Small, snapshot-bound Chinese BM25 inverted index."""
from collections import Counter, defaultdict
import math
import re
import unicodedata

PROFILE = {"tokenizer": "zh-char-bigram-ascii-identifier-v1", "k1": 1.2, "b": 0.75}
TOKEN = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[-_./][a-z0-9]+)*(?:[%℃])?|[¥￥]")


def tokenize(text):
    for match in TOKEN.finditer(unicodedata.normalize("NFKC", text).casefold()):
        term = match.group()
        if "\u3400" <= term[0] <= "\u9fff":
            yield from term
            yield from (term[i:i + 2] for i in range(len(term) - 1))
        else:
            yield term


def build_sparse(snapshot_id, corpus_id, chunks):
    terms = defaultdict(list)
    lengths = []
    for row, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk["text"]))
        lengths.append(sum(counts.values()))
        for term, count in counts.items():
            terms[term].append([row, count])
    return dict(snapshot_id=snapshot_id, corpus_snapshot_id=corpus_id, profile=PROFILE,
                child_ids=[c["child_id"] for c in chunks], lengths=lengths, terms=dict(terms))


def validate_sparse(index, snapshot_id, corpus_id, chunks):
    lengths, terms = index["lengths"], index["terms"]
    if (index["snapshot_id"] != snapshot_id or index["corpus_snapshot_id"] != corpus_id
            or index["profile"] != PROFILE or index["child_ids"] != [c["child_id"] for c in chunks]
            or len(lengths) != len(chunks) or any(type(v) is not int or v < 0 for v in lengths)):
        raise ValueError("BM25 snapshot identity or row map mismatch")
    counts = [0] * len(lengths)
    for term, postings in terms.items():
        previous = -1
        for row, tf in postings:
            if not 0 <= row < len(lengths) or row <= previous or type(tf) is not int or tf <= 0:
                raise ValueError(f"Invalid BM25 postings for {term!r}")
            counts[row] += tf
            previous = row
    if counts != lengths:
        raise ValueError("BM25 document lengths do not match postings")


def bm25(index, query, limit):
    scores = defaultdict(float)
    lengths = index["lengths"]
    size = len(lengths)
    average = sum(lengths) / size if size else 0
    if not average:
        return []
    k1, b = index["profile"]["k1"], index["profile"]["b"]
    for term in sorted(set(tokenize(query))):
        postings = index["terms"].get(term, [])
        idf = math.log1p((size - len(postings) + 0.5) / (len(postings) + 0.5))
        for row, tf in postings:
            scores[row] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[row] / average))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
