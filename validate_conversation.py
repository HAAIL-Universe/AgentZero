# Agent Zero Training Data Validation Script
# Scores conversations against pattern-specific rubrics
# Returns: confidence score + acceptance status

import json
import re
from typing import Dict, List, Tuple
from enum import Enum

class ValidationStatus(Enum):
    ACCEPT = "ACCEPT"
    FLAG_REVIEW = "FLAG_REVIEW"
    REJECT = "REJECT"

class ConversationValidator:
    """Validates training conversations against Agent Zero pattern rubrics."""

    def __init__(self, safeguard_middleware=None):
        """
        Args:
            safeguard_middleware: Optional callable that checks safety (Z:\safeguard)
        """
        self.safeguard = safeguard_middleware
        self.pattern_rubrics = self._load_rubrics()

    def _load_rubrics(self) -> Dict:
        """Load pattern-specific acceptance criteria."""
        return {
            "P001": self._rubric_p001,
            "P002": self._rubric_p002,
            "P003": self._rubric_p003,
            "P004": self._rubric_p004,
            "P005": self._rubric_p005,
            "P006": self._rubric_p006,
            "P007": self._rubric_p007,
            "P008": self._rubric_p008,
            "P009": self._rubric_p009,
            "P010": self._rubric_p010,
            "P011": self._rubric_p011,
            "P012": self._rubric_p012,
        }

    # =========================
    # UNIVERSAL CHECKS
    # =========================

    def _safety_check(self, conversation: Dict) -> Tuple[bool, str]:
        """
        Run conversation through safeguarding middleware.
        Returns: (is_safe, reason_if_not)
        """
        if not self.safeguard:
            return True, ""  # No middleware, assume safe

        try:
            result = self.safeguard(conversation)
            if not result.get("safe", True):
                return False, result.get("reason", "Safety violation")
            return True, ""
        except Exception as e:
            return False, f"Safeguard error: {str(e)}"

    def _coherence_check(self, conversation: Dict) -> Dict[str, any]:
        """Check basic conversation coherence."""
        messages = conversation.get("messages", [])

        checks = {
            "min_turns": len(messages) >= 5,
            "alternating_roles": self._check_role_alternation(messages),
            "no_repetition": not self._check_repetition(messages),
            "user_responses_relevant": self._check_relevance(messages),
        }

        # Score: 25 points each
        score = sum(25 for v in checks.values() if v)

        return {
            "score": score,
            "breakdown": checks,
            "max": 100
        }

    def _check_role_alternation(self, messages: List) -> bool:
        """Verify roles alternate (user -> assistant -> user ...)."""
        if len(messages) < 2:
            return False
        for i in range(len(messages) - 1):
            if messages[i]["role"] == messages[i+1]["role"]:
                return False
        return True

    def _check_repetition(self, messages: List) -> bool:
        """Check if Agent Zero repeats itself excessively."""
        assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"]

        # Simple heuristic: if any two consecutive Agent Zero messages share >60% overlap
        for i in range(len(assistant_messages) - 1):
            msg1 = set(assistant_messages[i].lower().split())
            msg2 = set(assistant_messages[i+1].lower().split())
            if len(msg1 & msg2) / max(len(msg1), len(msg2)) > 0.6:
                return True  # Repetition detected
        return False

    def _check_relevance(self, messages: List) -> bool:
        """Check if user responses relate to Agent Zero messages."""
        # Heuristic: no single-word responses to multi-sentence prompts
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "assistant" and messages[i+1]["role"] == "user":
                if len(messages[i+1]["content"].split()) < 3:
                    # One-word response to question
                    if "?" in messages[i]["content"]:
                        return False
        return True

    # =========================
    # PATTERN-SPECIFIC RUBRICS
    # =========================

    def _rubric_p001(self, conversation: Dict) -> Dict[str, any]:
        """P001: Goal Articulation"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "goal_clarity": self._contains_goal_language(content),
            "agent_zero_probes": self._count_questions(content) >= 2,
            "specificity_capture": self._contains_specific_detail(content),
            "natural_tone": not self._is_robotic(content),
        }

        score = sum(25 for v in criteria.values() if v)
        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p002(self, conversation: Dict) -> Dict[str, any]:
        """P002: Goal Progress (Momentum)"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "effort_shown": self._contains_action_language(content),
            "links_to_goal": self._references_goal(content),
            "asks_reflection": self._contains_reflection_q(content),
            "builds_momentum": self._suggests_next_step(content),
            "genuine_tone": not self._is_patronizing(content),
        }

        score = 0
        score += 20 if criteria.get("effort_shown") else 0
        score += 20 if criteria.get("links_to_goal") else 0
        score += 20 if criteria.get("asks_reflection") else 0
        score += 25 if criteria.get("builds_momentum") else 0
        score += 15 if criteria.get("genuine_tone") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p003(self, conversation: Dict) -> Dict[str, any]:
        """P003: Drift Detection"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "misalignment_clear": self._detect_misalignment(content),
            "curious_not_accusatory": self._is_curious(content) and not self._is_accusatory(content),
            "asks_why": self._contains_why_question(content),
            "path_forward": self._offers_solution(content),
            "natural": not self._is_forced(content),
        }

        score = 0
        score += 25 if criteria.get("misalignment_clear") else 0
        score += 25 if criteria.get("curious_not_accusatory") else 0
        score += 20 if criteria.get("asks_why") else 0
        score += 15 if criteria.get("path_forward") else 0
        score += 15 if criteria.get("natural") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p004(self, conversation: Dict) -> Dict[str, any]:
        """P004: Avoidance Patterns"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "pattern_clear": self._detect_avoidance_pattern(content),
            "named_specifically": self._contains_i_noticed(content),
            "stays_curious": self._is_curious(content),
            "no_assumptions": not self._makes_assumptions(content),
            "offers_explore": self._offers_to_explore(content),
        }

        score = 0
        score += 25 if criteria.get("pattern_clear") else 0
        score += 20 if criteria.get("named_specifically") else 0
        score += 20 if criteria.get("stays_curious") else 0
        score += 20 if criteria.get("no_assumptions") else 0
        score += 15 if criteria.get("offers_explore") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p005(self, conversation: Dict) -> Dict[str, any]:
        """P005: Challenge Response"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "real_challenge": self._detects_real_challenge(content),
            "doesnt_soften": self._stands_firm(content),
            "user_reacts": self._user_has_reaction(content),
            "agent_zero_present": self._stays_engaged(content),
            "respects_choice": self._is_respectful(content),
        }

        score = 0
        score += 25 if criteria.get("real_challenge") else 0
        score += 25 if criteria.get("doesnt_soften") else 0
        score += 20 if criteria.get("user_reacts") else 0
        score += 15 if criteria.get("agent_zero_present") else 0
        score += 15 if criteria.get("respects_choice") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p006(self, conversation: Dict) -> Dict[str, any]:
        """P006: Consistency Checks"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "actual_inconsistency": self._detect_real_inconsistency(content),
            "references_specific": self._cites_past_statement(content),
            "curious": self._is_curious(content),
            "offers_space": self._offers_explanation_space(content),
            "natural_pacing": len(messages) > 6,  # Takes time to build
        }

        score = 0
        score += 25 if criteria.get("actual_inconsistency") else 0
        score += 20 if criteria.get("references_specific") else 0
        score += 25 if criteria.get("curious") else 0
        score += 15 if criteria.get("offers_space") else 0
        score += 15 if criteria.get("natural_pacing") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p007(self, conversation: Dict) -> Dict[str, any]:
        """P007: Curiosity Gaps"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "real_gap": self._identifies_gap(content),
            "specific": "?" in content and len(content.split()) > 20,
            "built_on_rapport": self._is_relevant(content),
            "genuine": not self._is_probing(content),
            "open_narrative": self._question_is_open(content),
        }

        score = 0
        score += 25 if criteria.get("real_gap") else 0
        score += 20 if criteria.get("specific") else 0
        score += 20 if criteria.get("built_on_rapport") else 0
        score += 20 if criteria.get("genuine") else 0
        score += 15 if criteria.get("open_narrative") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p008(self, conversation: Dict) -> Dict[str, any]:
        """P008: Emotional Register Shifts"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "shift_detectable": self._describes_energy_change(content),
            "named": self._contains_i_notice(content),
            "references_previous": self._references_last_time(content),
            "asks_open": self._contains_open_what_happened(content),
            "ties_to_goals": self._connects_to_goals(content),
        }

        score = 0
        score += 25 if criteria.get("shift_detectable") else 0
        score += 20 if criteria.get("named") else 0
        score += 20 if criteria.get("references_previous") else 0
        score += 20 if criteria.get("asks_open") else 0
        score += 15 if criteria.get("ties_to_goals") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p009(self, conversation: Dict) -> Dict[str, any]:
        """P009: Growth Edge"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "acknowledges_difficulty": self._acknowledges_hard(content),
            "validates_fear": self._validates_fear(content),
            "reframes_growth": self._reframes_as_growth(content),
            "asks_needed": self._asks_what_needed(content),
            "respects_agency": self._respects_agency(content),
        }

        score = 0
        score += 25 if criteria.get("acknowledges_difficulty") else 0
        score += 20 if criteria.get("validates_fear") else 0
        score += 20 if criteria.get("reframes_growth") else 0
        score += 20 if criteria.get("asks_needed") else 0
        score += 15 if criteria.get("respects_agency") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p010(self, conversation: Dict) -> Dict[str, any]:
        """P010: Communication Style Adaptation"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        # This pattern requires detecting multiple "users" or personas
        # Simplified: check for distinct communication patterns
        criteria = {
            "distinct_styles": self._contains_style_variation(content),
            "appropriate_adaptation": self._shows_adaptation(content),
            "effective": not self._alienates_user(content),
            "honest": not self._is_manipulative(content),
        }

        score = 0
        for k, v in criteria.items():
            score += 20 if v else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p011(self, conversation: Dict) -> Dict[str, any]:
        """P011: Scope Alignment"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "scope_stated": self._user_defines_scope(content),
            "respects_scope": self._suggestion_in_scope(content),
            "acknowledges_alternative": self._acknowledges_other_paths(content),
            "refocuses": self._refocuses_on_scope(content),
            "validates": self._validates_choice(content),
        }

        score = 0
        score += 20 if criteria.get("scope_stated") else 0
        score += 25 if criteria.get("respects_scope") else 0
        score += 20 if criteria.get("acknowledges_alternative") else 0
        score += 20 if criteria.get("refocuses") else 0
        score += 15 if criteria.get("validates") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    def _rubric_p012(self, conversation: Dict) -> Dict[str, any]:
        """P012: Self-Awareness Breakthrough"""
        messages = conversation.get("messages", [])
        content = "\n".join([m["content"] for m in messages])

        criteria = {
            "not_generic": self._is_specific_insight(content),
            "grounded_evidence": self._cites_examples(content),
            "contradicts_story": self._contradicts_self_narrative(content),
            "user_has_aha": self._user_has_realization(content),
            "reframe_true": self._reframe_grounded(content),
        }

        score = 0
        score += 25 if criteria.get("not_generic") else 0
        score += 25 if criteria.get("grounded_evidence") else 0
        score += 20 if criteria.get("contradicts_story") else 0
        score += 15 if criteria.get("user_has_aha") else 0
        score += 15 if criteria.get("reframe_true") else 0

        return {"score": score, "max": 100, "breakdown": criteria}

    # =========================
    # HELPER FUNCTIONS
    # =========================

    def _contains_goal_language(self, text: str) -> bool:
        patterns = [r'\bi\s+want', r'\bi\s+.*\s+goal', r'\bmy\s+goal', r'\bi\s+.*\s+trying']
        return any(re.search(p, text, re.I) for p in patterns)

    def _count_questions(self, text: str) -> int:
        return text.count("?")

    def _contains_specific_detail(self, text: str) -> bool:
        # Check for personal specifics (not generic)
        detail_indicators = [
            r'because\s+', r'specifically\s+', r'when\s+', r'how\s+',
            r'about\s+\w+\s+', r'this\s+fear', r'that\s+\w+', r'example'
        ]
        return any(re.search(p, text, re.I) for p in detail_indicators)

    def _is_robotic(self, text: str) -> bool:
        robotic_phrases = ["that's great", "you can do it", "sounds good", "excellent point"]
        return sum(text.lower().count(phrase) for phrase in robotic_phrases) > 2

    def _contains_action_language(self, text: str) -> bool:
        actions = [r'\bi\s+did', r'\bi\s+tried', r'\bi\s+had', r'\baction', r'\bpractice']
        return any(re.search(p, text, re.I) for p in actions)

    def _references_goal(self, text: str) -> bool:
        return re.search(r'(back\s+to|connect\s+to|original|goal|objective)', text, re.I) is not None

    def _contains_reflection_q(self, text: str) -> bool:
        return re.search(r'(what\s+was|what\s+.*\s+learn|what\s+.*\s+change|how\s+.*\s+different)', text, re.I) is not None

    def _suggests_next_step(self, text: str) -> bool:
        return re.search(r'(what\s+.*\s+next|next\s+step|following|move\s+toward)', text, re.I) is not None

    def _is_patronizing(self, text: str) -> bool:
        patronizing = ["that's great", "good job", "amazing", "✓", "you got this"]
        return sum(text.lower().count(p) for p in patronizing) > 3

    def _detect_misalignment(self, text: str) -> bool:
        # Look for tension between goal and action
        return bool(re.search(r"(haven't|didn't|wasn't|avoided|didn't do)", text, re.I)) and \
               bool(re.search(r"(said|goal|want|should)", text, re.I))

    def _is_curious(self, text: str) -> bool:
        curious = ["what\s+happened", "what\s+changed", "interesting", "curious", "tell\s+me"]
        return any(re.search(p, text, re.I) for p in curious)

    def _is_accusatory(self, text: str) -> bool:
        accus = ["you\s+failed", "you\s+didn't", "you\s+never", "you're\s+not"]
        return any(re.search(p, text, re.I) for p in accus)

    def _contains_why_question(self, text: str) -> bool:
        return "what happened" in text.lower() or "what\s+changed" in text.lower()

    def _offers_solution(self, text: str) -> bool:
        return re.search(r"(what\s+would|how\s+can|what\s+.*\s+need)", text, re.I) is not None

    def _is_forced(self, text: str) -> bool:
        return len(text) < 300 or text.count(".") > len(text.split()) // 5

    def _detect_avoidance_pattern(self, text: str) -> bool:
        # Look for multiple mentions of same topic with deflection
        mentions = len(re.findall(r'(that|it|this|the topic|that subject)', text, re.I))
        return mentions >= 2

    def _contains_i_noticed(self, text: str) -> bool:
        return bool(re.search(r"(i\s+notice|i\s+.*\s+notice|i\s+keep\s+noticing)", text, re.I))

    def _makes_assumptions(self, text: str) -> bool:
        # Check for "you are scared of X" without asking
        return bool(re.search(r"(you\s+are\s+\w+ing|you\s+must|you\s+probably)", text, re.I))

    def _offers_to_explore(self, text: str) -> bool:
        return re.search(r"(want\s+to\s+explore|explore\s+that|want\s+to\s+talk)", text, re.I) is not None

    def _detects_real_challenge(self, text: str) -> bool:
        # Look for something that provokes or confronts
        challenges = ["what\s+if", "but\s+maybe", "i\s+want\s+to\s+ask", "real\s+issue"]
        return any(re.search(p, text, re.I) for p in challenges)

    def _stands_firm(self, text: str) -> bool:
        # Doesn't back down or soften immediately
        return not bool(re.search(r"(it's\s+okay|no\s+pressure|only\s+if|don't\s+worry)", text, re.I))

    def _user_has_reaction(self, text: str) -> bool:
        # User shows emotion or engagement
        reactions = ["wow", "yeah", "oh", "huh", "interesting", "never thought"]
        return any(r in text.lower() for r in reactions)

    def _stays_engaged(self, text: str) -> bool:
        # Agent Zero continues exploring after user response
        return "?" in text and len(text) > 200

    def _is_respectful(self, text: str) -> bool:
        disrespectful = ["you're not serious", "you don't care", "you're lazy", "fail"]
        return not any(phrase in text.lower() for phrase in disrespectful)

    def _detect_real_inconsistency(self, text: str) -> bool:
        # Statements that directly contradict
        return bool(re.search(r"(different|changed|opposite|contradict|shift)", text, re.I))

    def _cites_past_statement(self, text: str) -> bool:
        citations = ["last\s+time", "previously", "you\s+said", "before", "earlier", "week"]
        return any(re.search(p, text, re.I) for p in citations)

    def _offers_explanation_space(self, text: str) -> bool:
        return "?" in text and "new" in text.lower() or "changed" in text.lower()

    def _identifies_gap(self, text: str) -> bool:
        # Looking for something not yet explained
        gap_signals = ["curious", "don't\s+understand", "want\s+to\s+know", "wonder\s+about"]
        return any(re.search(p, text, re.I) for p in gap_signals)

    def _is_relevant(self, text: str) -> bool:
        # Not invasive, ties to goals
        return len(text) < 500  # Not prying too deep

    def _is_probing(self, text: str) -> bool:
        # Gotcha or test questions
        return bool(re.search(r"(are\s+you|do\s+you|did\s+you\s+really)", text, re.I))

    def _question_is_open(self, text: str) -> bool:
        # Can't be answered with yes/no
        closed_q = ["do\s+you\s+\w+\?", "are\s+you\s+\w+\?", "is\s+that\s+\w+\?"]
        open_q = ["what", "how", "tell\s+me"]
        has_open = any(re.search(p, text, re.I) for p in open_q)
        has_closed = any(re.search(p, text, re.I) for p in closed_q)
        return has_open or not has_closed

    def _describes_energy_change(self, text: str) -> bool:
        energy_words = ["energized", "flat", "withdrawn", "optimistic", "different", "shift"]
        return any(word in text.lower() for word in energy_words)

    def _contains_i_notice(self, text: str) -> bool:
        return bool(re.search(r"(i\s+notice|i\s+see|seems\s+\w+|sounds\s+\w+)", text, re.I))

    def _references_last_time(self, text: str) -> bool:
        return bool(re.search(r"(last\s+time|previous\s+session|before|earlier)", text, re.I))

    def _contains_open_what_happened(self, text: str) -> bool:
        return "what happened" in text.lower() or "what\s+changed" in text.lower()

    def _connects_to_goals(self, text: str) -> bool:
        return bool(re.search(r"(goal|objective|bigger\s+picture|affect)", text, re.I))

    def _acknowledges_hard(self, text: str) -> bool:
        hard_phrases = ["brave", "hard", "difficult", "risky", "scary", "challenging"]
        return any(phrase in text.lower() for phrase in hard_phrases)

    def _validates_fear(self, text: str) -> bool:
        return bool(re.search(r"(fear|at\s+stake|risk|rejection|failure)", text, re.I))

    def _reframes_as_growth(self, text: str) -> bool:
        return bool(re.search(r"(growth|edge|learning|become\s+stronger)", text, re.I))

    def _asks_what_needed(self, text: str) -> bool:
        return bool(re.search(r"(what\s+.*\s+need|what\s+support|what\s+help)", text, re.I))

    def _respects_agency(self, text: str) -> bool:
        agency = ["you're choosing", "you can", "your decision", "up to you"]
        return any(phrase in text.lower() for phrase in agency)

    def _contains_style_variation(self, text: str) -> bool:
        # Different sentence structures or word choices
        sentences = text.split(".")
        return len(set(len(s.split()) for s in sentences)) > 2

    def _shows_adaptation(self, text: str) -> bool:
        # Changes tone/style
        return text.lower().count("but") > 0  # Uses contrasts

    def _alienates_user(self, text: str) -> bool:
        alienating = ["you should", "you must", "obviously", "clearly"]
        return sum(text.lower().count(p) for p in alienating) > 1

    def _is_manipulative(self, text: str) -> bool:
        # Lies or tricks to match style
        return False  # Hard to detect automatically

    def _user_defines_scope(self, text: str) -> bool:
        scope_words = ["want\s+to", "goal", "focus\s+on", "interested\s+in"]
        return any(re.search(p, text, re.I) for p in scope_words)

    def _suggestion_in_scope(self, text: str) -> bool:
        # Stays on topic
        return not bool(re.search(r"(outside|beyond|beyond\s+scope)", text, re.I))

    def _acknowledges_other_paths(self, text: str) -> bool:
        return bool(re.search(r"(many\s+people|some\s+folks|could|might|another\s+path)", text, re.I))

    def _refocuses_on_scope(self, text: str) -> bool:
        return bool(re.search(r"(your\s+scope|you\s+said|stay\s+focused|back\s+to)", text, re.I))

    def _validates_choice(self, text: str) -> bool:
        return not bool(re.search(r"(limiting|narrow|short-sighted)", text, re.I))

    def _is_specific_insight(self, text: str) -> bool:
        # Not universal advice
        return bool(re.search(r"(you|your|specifically)", text, re.I))

    def _cites_examples(self, text: str) -> bool:
        # References specific instances
        examples = ["times", "times you", "occasions", "instances", "in\s+the", "last\s+\w+\s+sessions"]
        return any(re.search(p, text, re.I) for p in examples)

    def _contradicts_self_narrative(self, text: str) -> bool:
        # "You say X but you actually Y"
        return bool(re.search(r"(you\s+say|you\s+claim|you\s+believe)", text, re.I)) and \
               bool(re.search(r"(but|actually|in\s+reality|however)", text, re.I))

    def _user_has_realization(self, text: str) -> bool:
        realizations = ["huh", "never thought", "i guess", "wow", "oh", "actually", "yeah"]
        return any(phrase in text.lower() for phrase in realizations)

    def _reframe_grounded(self, text: str) -> bool:
        # Based on evidence, not projection
        return "you've" in text.lower() or "you\s+did" in text.lower()

    # =========================
    # MAIN VALIDATION FUNCTION
    # =========================

    def validate(self, conversation: Dict, pattern: str) -> Dict:
        """
        Validate a conversation against a pattern.

        Args:
            conversation: {"messages": [{"role": "user"|"assistant", "content": "..."}]}
            pattern: "P001" through "P012"

        Returns:
            {
                "confidence": 0-100,
                "status": "ACCEPT"|"FLAG_REVIEW"|"REJECT",
                "pattern": "P001",
                "safety": {"pass": bool, "reason": str},
                "coherence": {"score": int, "breakdown": {...}},
                "pattern_specific": {"score": int, "breakdown": {...}},
                "final_score": int,
                "notes": str
            }
        """

        # 1. SAFETY CHECK (HARD FAIL)
        is_safe, safety_reason = self._safety_check(conversation)
        if not is_safe:
            return {
                "confidence": 0,
                "status": ValidationStatus.REJECT.value,
                "pattern": pattern,
                "safety": {"pass": False, "reason": safety_reason},
                "notes": "Safety violation - rejected immediately"
            }

        # 2. COHERENCE CHECK
        coherence = self._coherence_check(conversation)
        if coherence["score"] < 50:
            return {
                "confidence": 0,
                "status": ValidationStatus.REJECT.value,
                "pattern": pattern,
                "safety": {"pass": True},
                "coherence": coherence,
                "notes": "Coherence too low - conversation is incoherent"
            }

        # 3. PATTERN-SPECIFIC CHECK
        if pattern not in self.pattern_rubrics:
            return {
                "confidence": 0,
                "status": ValidationStatus.REJECT.value,
                "pattern": pattern,
                "notes": f"Unknown pattern: {pattern}"
            }

        rubric_fn = self.pattern_rubrics[pattern]
        pattern_score = rubric_fn(conversation)

        # 4. CALCULATE FINAL CONFIDENCE
        # Average of coherence and pattern-specific
        final_confidence = (coherence["score"] + pattern_score["score"]) / 2

        # 5. DETERMINE STATUS
        if final_confidence >= 90:
            status = ValidationStatus.ACCEPT.value
        elif final_confidence >= 70:
            status = ValidationStatus.FLAG_REVIEW.value
        else:
            status = ValidationStatus.REJECT.value

        return {
            "confidence": round(final_confidence, 1),
            "status": status,
            "pattern": pattern,
            "safety": {"pass": True},
            "coherence": coherence,
            "pattern_score": pattern_score,
            "final_confidence": round(final_confidence, 1),
            "notes": f"Pattern {pattern} - Confidence {final_confidence:.0f}%"
        }


def main():
    """Example usage."""
    validator = ConversationValidator()

    # Example conversation
    example_conv = {
        "messages": [
            {"role": "user", "content": "I want to get better at public speaking. I'm terrified of it."},
            {"role": "assistant", "content": "What specifically scares you? Is it the audience, forgetting words, being judged?"},
            {"role": "user", "content": "All of it, but mainly being judged as incompetent."},
            {"role": "assistant", "content": "Okay, so it's not just speaking—it's about competence perception. How long have you been avoiding public speaking situations?"},
            {"role": "user", "content": "Years. Since college probably."},
        ]
    }

    result = validator.validate(example_conv, "P001")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
