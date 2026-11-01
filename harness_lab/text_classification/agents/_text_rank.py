"""Shared TF-IDF ranking over arbitrary texts (same weighting as TfidfCover._vecs)."""

import math
from collections import Counter

from .tfidf_cover import tokens


def rank_texts(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """Return (cosine score, index) pairs for texts, best first."""
    if not texts:
        return []
    docs = [Counter(tokens(t)) for t in texts]
    df = Counter(t for d in docs for t in d)
    n = len(docs) + 1
    idf = {t: math.log(n / (1 + c)) + 1 for t, c in df.items()}

    def vec(c: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(f)) * idf.get(t, 0) for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    q = vec(Counter(tokens(query)))
    scored = [(sum(w * vec(d).get(t, 0) for t, w in q.items()), i) for i, d in enumerate(docs)]
    return sorted(scored, reverse=True)
