"""confusion_contrast with its error contract made explicit.

Records (true, predicted) pairs and the indices of examples the model got wrong.
Only meaningful in online mode, where learn_from_batch receives real predictions.
"""

import json

from ..llm import LLMCallable
from . import confusion_contrast


class ContrastOnline(confusion_contrast.ConfusionContrast):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.n_errors_seen = 0

    def learn_from_batch(self, batch_results):
        self.n_errors_seen += sum(1 for r in batch_results if r.get("was_correct") is False)
        super().learn_from_batch(batch_results)

    def get_state(self) -> str:
        state = json.loads(super().get_state())
        state.update(n_errors_seen=self.n_errors_seen, n_items=len(self.error_ids))
        return json.dumps(state)

    def set_state(self, state: str) -> None:
        super().set_state(state)
        self.n_errors_seen = json.loads(state).get("n_errors_seen", len(self.error_ids))
