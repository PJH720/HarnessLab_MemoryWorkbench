"""ReasoningBank-style memory: distill each real mistake into one structured, reusable cue.

Adapted from FAILED_SI (reasoning_bank WebArena/prompts/memory_instruction.py):
reflect on why the attempt failed, then write a general lesson, not a copy of the case.
"""

from ..memory_system import extract_json_field
from . import _error_memory

CUE_PROMPT = """You are reviewing a mistake made by a symptom-to-disease classifier.

Patient symptoms: {symptoms}
Classifier's answer: {wrong}
Correct answer: {right}
Classifier's reasoning at the time: {reasoning}

First think about why the classifier failed: which symptom(s) misled it toward "{wrong}", and which symptom(s) actually point to "{right}".
Then write ONE reusable lesson that would prevent this mistake for future, different patients.
Do not copy the patient's sentence; describe general symptom patterns. You may name the two diseases.

Respond in JSON:
{{"title": "[short name of the lesson]", "when": "[when this lesson applies]", "cue": "[one sentence: the symptom cue that separates the two diseases]"}}"""


class CueBank(_error_memory.ErrorMemory):
    lessons_header = "Diagnostic cues learned from past mistakes on similar cases:"

    def _induce(self, symptoms, wrong, right, reasoning):
        response = self.call_llm(CUE_PROMPT.format(symptoms=symptoms, wrong=wrong, right=right, reasoning=reasoning))
        cue = extract_json_field(response, "cue").strip()
        if not cue:
            return None
        return {"title": extract_json_field(response, "title").strip(),
                "when": extract_json_field(response, "when").strip(), "cue": cue}

    def _render(self, item):
        return f"- [{item['title']}] When: {item['when']} -> {item['cue']}"
