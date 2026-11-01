"""Baseline: the k most similar training examples only (DSPy KNNFewShot-style selection).

This is what the midterm ``confusion_contrast`` effectively ran, because offline
training never produced a wrong prediction for it to record.
"""

from ..llm import LLMCallable
from . import tfidf_cover


class KnnTop6(tfidf_cover.TfidfCover):
    def __init__(self, llm: LLMCallable, k: int = 6):
        super().__init__(llm, k=k)

    def _select(self, query: str) -> list[int]:
        return [i for _, i in self._rank(query)[: self.k]][::-1]  # nearest last
