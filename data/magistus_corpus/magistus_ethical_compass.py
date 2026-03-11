# core/ethical_compass.py

from utils.llm_wrapper import generate_response
from pydantic import BaseModel
from typing import List

class EthicalCompass(BaseModel):
    """
    Represents the active ethical boundaries and manifesto clauses guiding agent behavior.
    """
    manifesto_clauses: List[str] = [
        "Respect user agency",
        "Avoid manipulation or emotional simulation",
        "Preserve transparency at all times",
        "Seek clarification before prescriptive action",
        "Flag irreversible actions for explicit consent"
    ]

    def evaluate_input(self, user_input: str) -> str:
        """
        Uses the LLM to determine if the user's request violates ethical boundaries.
        Returns 'yes' or 'no' with reasoning.
        """
        prompt = (
            f"User said: \"{user_input}\"\n\n"
            f"Here are the current ethical boundaries:\n- " + "\n- ".join(self.manifesto_clauses) + "\n\n"
            "Does this input conflict with any of the boundaries above?\n"
            "Respond with JSON:\n"
            "{\n  \"violation\": \"yes/no\",\n  \"justification\": \"...\"\n}"
        )

        try:
            response = generate_response(
                prompt=prompt.strip(),
                system_prompt="You are an ethics-aware alignment assistant. Only respond with compact JSON."
            ).strip()
            return response
        except Exception:
            return '{"violation": "unknown", "justification": "Failed to evaluate."}'
        
    def __len__(self):
        return len(self.manifesto_clauses)

    def __getitem__(self, index):
        return self.manifesto_clauses[index]

    def __iter__(self):
        return iter(self.manifesto_clauses)
