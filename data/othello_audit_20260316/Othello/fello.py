from utils.prompts import generate_daily_prompt
from modules.agents.reflective_agent import ReflectiveAgent
from modules.agentic_agents.architect_agent import ArchitectAgent
from modules.agentic_agents.shadow_agent import ShadowAgent
from modules.agentic_agents.goal_management_agent import GoalManagementAgent
from modules.agents.behavioral_agent import BehavioralAgent
from modules.agentic_agents.psyche_agent import PsycheAgent
from modules.agents.routine_tracker_agent import RoutineTrackerAgent
from modules.agents.impatience_detection_agent import ImpatienceDetectionAgent
from modules.agents.trait_agent import TraitAgent
from modules.agentic_agents.decision_vault_agent import DecisionVaultAgent
from modules.memory_handler import MemoryHandler
from core.llm_wrapper import LLMWrapper  

import os
import json

class Fello:
    """
    Fello: The primary agentic orchestrator.
    Coordinates reflection, goal management, shadow snapshotting, and persistence.
    Integrates all agents cleanly with consent-tier snapshot logic.
    Manages agent autonomy levels: passive, suggestive, active.
    """

    def __init__(self, central_hub=None, agentic_hub=None, model=None):
        model = model or LLMWrapper()
        self.reflective_agent = ReflectiveAgent()
        self.architect = ArchitectAgent(model=model, central_hub=central_hub, agentic_hub=agentic_hub)
        self.shadow_agent = ShadowAgent(central_hub=central_hub, agentic_hub=agentic_hub)
        self.goal_management_agent = GoalManagementAgent(central_hub=central_hub, agentic_hub=agentic_hub)
        self.behavioral_agent = BehavioralAgent()
        self.psyche_agent = PsycheAgent(central_hub=central_hub, agentic_hub=agentic_hub)
        self.routine_tracker_agent = RoutineTrackerAgent()
        self.impatience_detection_agent = ImpatienceDetectionAgent()
        self.trait_agent = TraitAgent()
        self.decision_vault_agent = DecisionVaultAgent(central_hub=central_hub, agentic_hub=agentic_hub)
        self.generate_daily_prompt = generate_daily_prompt
        self.data_handler = MemoryHandler()

        self.autonomy_levels = {
            'passive': 0,
            'suggestive': 1,
            'active': 2
        }

    def set_autonomy_level(self, level, consent_tier=1):
        if level in self.autonomy_levels:
            autonomy_level = self.autonomy_levels[level]
            print(f"Autonomy set to {level} with consent level {consent_tier}.")
            return autonomy_level
        else:
            raise ValueError("Invalid autonomy level. Use 'passive', 'suggestive', or 'active'.")

    async def run_daily_check_in(self, consent_tier=1, autonomy_level='passive', user_data=None):
        if user_data is None:
            user_data = {"text": "", "mood": "", "context": ""}

        autonomy = self.set_autonomy_level(autonomy_level, consent_tier)

        shadow_data = self.shadow_agent.get_refined_shadow()

        reflection_result = self.reflective_agent.run_full_reflection(
            shadow_data=shadow_data,
            user_data=user_data
        )

        self.decision_vault_agent.log_decision(
            decision_type="reflection",
            details={"reflection": reflection_result.get("summary_text", "N/A")},
            outcome="success",
            context={"mood": reflection_result.get("summary_text", "N/A")}
        )

        behavioral_insights = self.behavioral_agent.analyze_behavior(
            shadow_data, reflection_result.get("summary_text", "neutral")
        )

        psyche_insights = await self.psyche_agent.analyze_psyche(
            shadow_data=shadow_data,
            behavioral_data=behavioral_insights,
            user_data=user_data
        )

        routine_snapshot = self.routine_tracker_agent.build_snapshot()

        impatience_result = self.impatience_detection_agent.detect_impatience(
            reflection_result.get("summary_text", ""), behavioral_insights["emotional_state"]
        )

        self.trait_agent.analyze_text(reflection_result.get("summary_text", ""))

        decision_analysis = self.decision_vault_agent.analyze_decisions()

        return {
            "prompt": generate_daily_prompt(
                reflection_result.get("mood", 5),
                reflection_result.get("reflection", ""),
                reflection_result.get("goal_update", "")
            ),
            "mood": reflection_result.get("summary_text", "N/A"),
            "reflection": reflection_result.get("summary_text", "N/A"),
            "goal_update": "N/A",
            "delta": reflection_result.get("delta"),
            "anomalies": reflection_result.get("anomalies"),
            "behavioral_insights": behavioral_insights,
            "psyche_insights": psyche_insights,
            "routine_snapshot": routine_snapshot,
            "impatience_result": impatience_result,
            "decision_analysis": decision_analysis
        }

    def deliberate(self):
        print("[Fello] Running deliberation cycle...")
        return self.run_daily_check_in(consent_tier=1, autonomy_level='suggestive')

    def view_goals(self):
        return self.architect.get_goals()

    def add_goal(self, goal):
        self.architect.add_goal(goal)
        self._save_goals()

    def edit_goal(self, goal_index, new_goal):
        self.architect.edit_goal(goal_index, new_goal)
        self._save_goals()

    def _save_goals(self, file_path="data/goals.json"):
        goals = self.architect.get_goals()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(goals, f, indent=2)
        print(f"[Fello] Goals saved to {file_path}.")
