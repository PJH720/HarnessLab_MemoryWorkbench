"""Reflexion-style memory: a free-form verbal reflection per real mistake.

Adapted from COT_REFLECT_INSTRUCTION (reflexion hotpotqa_runs/prompts.py): diagnose the
failure and state a concise plan for next time. Unlike cue_bank, the note is unstructured.
"""

from ..memory_system import extract_json_field
from . import _error_memory

REFLECT_PROMPT = """You are an advanced reasoning agent that can improve based on self reflection.
You were given a patient's symptoms and answered "{wrong}", but the correct diagnosis was "{right}".

Patient symptoms: {symptoms}
Your previous reasoning: {reasoning}

In 1-2 sentences, diagnose the reason for the failure and state a concise rule to follow next time on similar cases.
Do not repeat the patient's exact words.

Respond in JSON:
{{"final_answer": "[your reflection]"}}"""


class ReflexionNotes(_error_memory.ErrorMemory):
    lessons_header = "Reflections on past mistakes on similar cases:"

    def _induce(self, symptoms, wrong, right, reasoning):
        response = self.call_llm(REFLECT_PROMPT.format(symptoms=symptoms, wrong=wrong, right=right, reasoning=reasoning))
        note = extract_json_field(response, "final_answer").strip()
        return {"note": note} if note else None

    def _render(self, item):
        return f"- ({item['right']} mistaken for {item['wrong']}) {item['note']}"
