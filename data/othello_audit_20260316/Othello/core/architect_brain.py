import logging
from typing import Dict, Optional, List, Union

from utils.prompts import load_prompt
from utils.utilities import async_retry

# === Persona Engines ===
from modules.goal_manager import GoalManager
from modules.trait_manager import TraitManager
from modules.routine_tracker import RoutineTracker
from modules.conversation_parser import ConversationParser
# (Future: from modules.behaviouralist import BehaviouralistModule)


class Architect:
    """
    Main agentic orchestrator for H.A.A.I.L. FELLO™ / Othello.

    - Handles LLM prompt assembly, context window, and persona engine (manager) updates.
    - All *long-term* memory is handled by modular managers (GoalManager, etc.).
    - Only *short-term* conversation context (N most recent turns) is kept in RAM.
    """

    def __init__(self, model, memory_window: int = 20) -> None:
        self.model = model
        self.logger = logging.getLogger("ARCHITECT")
        self.framework = "H.A.A.I.L. FELLO™"
        self.context_window = memory_window

        # === Persona engines ===
        self.goal_mgr = GoalManager()
        self.trait_mgr = TraitManager()
        self.routine_tracker = RoutineTracker()
        self.parser = ConversationParser()
        # self.behaviouralist = BehaviouralistModule()  # (Future stub)

        # Simple rolling buffer of recent turns: [{"role": "...", "content": "..."}]
        self.short_term_memory: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Small helper: detect if a sentence is a real goal declaration
    # ------------------------------------------------------------------
    def _is_goal_declaration(self, text: str) -> bool:
        """
        Heuristic: only auto-create goals when the user clearly states one,
        not every time they casually mention the word 'goal'.
        """
        t = text.lower()

        phrases = [
            "my goal is",
            "the goal is",
            "the goal today is",
            "today the goal is",
            "goal today is",
            "i want to",
            "i'm aiming to",
            "i am aiming to",
            "i'm planning to",
            "i am planning to",
            "i have a goal",
            "i've got a goal",
        ]

        if any(p in t for p in phrases):
            return True

        # Slightly more general: "goal ... to <verb>"
        # e.g. "the goal is to improve your tracking"
        if "goal" in t and " to " in t and "is to" in t:
            return True

        return False


    def design_agent(self) -> str:
        return f"Designing agent based on {self.framework}"

    async def plan_and_execute(
        self,
        answers: Union[Dict[str, str], str],
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Main planning entrypoint.

        `answers` can be:
        - a dict from the old multi-question check-in (keys: goal/trait/routine/etc.), or
        - a plain string from the chat frontend.

        We normalise both cases into:
        - `answers_dict`: a dict view
        - `raw_text`: the main user utterance for this turn

        `context` can optionally include:
        - "goal_context": a text block describing the active goal + recent notes,
          which will be injected as an extra system message.
        """
        try:
            # ---- Normalise input -------------------------------------------------
            if isinstance(answers, str):
                raw_text = answers.strip()
                answers_dict: Dict[str, str] = {
                    "goal": raw_text,
                    "trait": "",
                    "routine": "",
                    "freeform": raw_text,
                }
            elif isinstance(answers, dict):
                answers_dict = answers
                raw_text = (
                    answers_dict.get("freeform")
                    or answers_dict.get("message")
                    or answers_dict.get("goal")
                    or answers_dict.get("trait")
                    or answers_dict.get("routine")
                    or " ".join(
                        v for v in answers_dict.values() if isinstance(v, str)
                    )
                ).strip()
            else:
                # Fallback: best-effort string representation
                raw_text = str(answers)
                answers_dict = {
                    "goal": raw_text,
                    "trait": "",
                    "routine": "",
                    "freeform": raw_text,
                }

            self.logger.info(f"🧭 Planning response to user input: {raw_text}")

            # ---- Persona engine updates via parser ------------------------------
            goal_text = answers_dict.get("goal", raw_text)
            trait_text = answers_dict.get("trait", raw_text)
            routine_text = answers_dict.get("routine", raw_text)

            # Traits
            traits = self.parser.extract_traits(trait_text)
            if traits:
                self.trait_mgr.record_traits(traits, context=trait_text)

            # Routines
            routines = self.parser.extract_routines(routine_text)
            if routines:
                for routine in routines:
                    self.routine_tracker.log_routine_detected(
                        routine, context=routine_text
                    )

            # Goals
            goals = self.parser.extract_goals(goal_text)
            if goals:
                for g in goals:
                    text = (g.get("text") or "").strip()
                    if not text:
                        continue

                    # Only treat it as a real goal if the user clearly
                    # declared one ("my goal is...", "the goal today is...", etc.)
                    if not self._is_goal_declaration(goal_text):
                        self.logger.info(
                            f"🛈 Detected possible goal text but not a clear "
                            f"declaration; skipping auto-save: {text!r}"
                        )
                        continue

                    # Avoid duplicates by checking existing goals in the stack
                    existing = self.goal_mgr.find_goal_by_name(text)
                    if existing:
                        self.logger.info(
                            f"🔁 Detected goal that matches existing goal "
                            f"#{existing['id']}: {existing['text']}"
                        )
                        continue

                    deadline = g.get("deadline")
                    new_goal = self.goal_mgr.add_goal(text, deadline=deadline)
                    self.logger.info(
                        f"🎯 Recorded new goal #{new_goal['id']}: {new_goal['text']}"
                    )

            # ---- Build user message summary for context -------------------------
            if isinstance(answers, dict):
                full_user_msg = "\n".join(
                    [f"{k.capitalize()} Q: {q}" for k, q in answers_dict.items() if q]
                )
            else:
                full_user_msg = raw_text

            self.short_term_memory.append(
                {"role": "user", "content": full_user_msg}
            )
            if len(self.short_term_memory) > self.context_window * 2:
                self.short_term_memory = self.short_term_memory[
                    -self.context_window * 2 :
                ]

            # ---- Build system prompt + messages ---------------------------------
            system_prompt = load_prompt("life_architect")
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            # Inject active goal context (if provided by API)
            if context is not None:
                goal_ctx = context.get("goal_context")
                if goal_ctx:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Active goal context for this conversation:\n\n"
                                f"{goal_ctx}"
                            ),
                        }
                    )

            # Then append short-term dialogue history
            messages += self.short_term_memory

            # ---- Call LLM -------------------------------------------------------
            raw_response = await async_retry(self.model.chat, messages)

            # Ensure we always end up with a string for type safety / Pylance
            if isinstance(raw_response, str):
                response: str = raw_response
            elif raw_response is None:
                response = ""
            else:
                response = str(raw_response)

            # Append assistant turn to memory
            self.short_term_memory.append(
                {"role": "assistant", "content": response}
            )

            # ---- Optional self-reflection hook ---------------------------------
            try:
                from core.self_reflection import SelfReflectionEngine

                sref = SelfReflectionEngine()
                summary = sref.run_reflection()
                self.logger.info(f"🪞 Self-reflection summary: {summary}")
            except Exception as e:
                self.logger.warning(f"Self-reflection failed to run: {e}")

            return response

        except Exception as e:
            self.logger.error(f"❌ Architect failed: {e}")
            return "Sorry, something went wrong planning that."

    def set_memory_window(self, window_size: int) -> None:
        self.context_window = window_size

    def clear_short_term_memory(self) -> None:
        self.short_term_memory = []
