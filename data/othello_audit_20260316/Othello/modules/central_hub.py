import yaml
from typing import Optional
import logging

from modules.agentic_agents.shadow_agent import ShadowAgent
from modules.agents.reflective_agent import ReflectiveAgent
from modules.agentic_agents.prism_agent import PrismAgent
from modules.agents.trait_agent import TraitAgent
from modules.agents.behavioral_agent import BehavioralAgent
from modules.agentic_agents.goal_management_agent import GoalManagementAgent
from modules.agents.aspirational_coach_agent import AspirationalCoachAgent
from modules.agents.conversation_agent import ConversationAgent
from modules.agents.impatience_detection_agent import ImpatienceDetectionAgent
from modules.agents.routine_tracker_agent import RoutineTrackerAgent
from modules.agentic_agents.psyche_agent import PsycheAgent
from modules.agentic_agents.decision_vault_agent import DecisionVaultAgent

logging.basicConfig(level=logging.DEBUG)

class ConsentStub:
    def __init__(self, level="active"):
        self.autonomy_level = level

    def get(self, key, default=None):
        if key == "autonomy_level":
            return self.autonomy_level
        return default

class CentralHub:
        
    def __init__(self, consent_file="config/consent.yaml", agentic_hub=None):
        self.consent = ConsentStub(level="active")
        self.agentic_hub = agentic_hub
        default_consent = {
            "autonomy": "passive",
            "level": 1,
            "permissions": {
                "passive": ["shadow_update", "reflection"],
                "suggestive": [],
                "active": []
            }
        }

        self.shadow_agent = ShadowAgent(central_hub=self, agentic_hub=agentic_hub)
        self.reflective_agent = ReflectiveAgent(hub=self)
        self.prism_agent = PrismAgent(central_hub=self, agentic_hub=agentic_hub)
        self.trait_agent = TraitAgent(hub=self)
        self.behavioral_agent = BehavioralAgent(hub=self)
        self.goal_agent = GoalManagementAgent(central_hub=self, agentic_hub=agentic_hub)
        self.decision_vault_agent = DecisionVaultAgent(central_hub=self, agentic_hub=agentic_hub)
        self.psyche_agent = PsycheAgent(central_hub=self, agentic_hub=agentic_hub)
        self.coach_agent = AspirationalCoachAgent(hub=self)
        self.conversation_agent = ConversationAgent(hub=self)
        self.impatience_agent = ImpatienceDetectionAgent(hub=self)
        self.routine_agent = RoutineTrackerAgent(hub=self)
    
    def check_autonomy(self, action_type):
        """
        Returns True if the current autonomy level allows the action.
        For now: 
            - 'passive' allows 'shadow_update' and 'reflection'
            - 'suggestive' allows those plus anything you want to add
            - 'active' allows everything
        """
        autonomy_level = self.consent.get("autonomy_level", "passive")

        if autonomy_level == "active":
            return True
        elif autonomy_level == "suggestive":
            return action_type in ["shadow_update", "reflection", "coach_action"]  # Add more as needed
        else:  # passive
            return action_type in ["shadow_update", "reflection"]

    def receive_user_input(self, text):
        logging.debug(f"CentralHub: Received user input: {text}")
        parsed = self.conversation_agent.parse(text)
        logging.debug(f"CentralHub: Data received from ConversationAgent: {parsed}")

        # Behaviors (existing)
        if parsed.get("behaviors"):
            logging.debug("CentralHub: Passing behaviors to BehavioralAgent")
            for behavior in parsed["behaviors"]:
                if hasattr(self.behavioral_agent, "track_behavior"):
                    self.behavioral_agent.track_behavior(behavior)

        # Habits and Events (NEW)
        if hasattr(self.conversation_agent.parser, "extract_habits_and_events"):
            logging.debug("CentralHub: Extracting habits and events")
            habits_events = self.conversation_agent.parser.extract_habits_and_events(text)
            for habit in habits_events.get("habits", []):
                if hasattr(self.behavioral_agent, "track_habit"):
                    self.behavioral_agent.track_habit(habit)
            for event in habits_events.get("events", []):
                if hasattr(self.behavioral_agent, "track_event"):
                    self.behavioral_agent.track_event(event)

        # Goals (existing)
        if parsed.get("goals"):
            logging.debug("CentralHub: Passing goals to GoalManagementAgent")
            for goal in parsed["goals"]:
                if hasattr(self.goal_agent, "add_or_edit_goal"):
                    if isinstance(goal, dict) and "description" in goal:
                        self.goal_agent.add_or_edit_goal(goal)
                    else:
                        self.goal_agent.add_or_edit_goal({"description": goal})

        # Long-Term Goals (NEW, future-proof)
        if hasattr(self.conversation_agent.parser, "extract_long_term_goals"):
            logging.debug("CentralHub: Extracting long-term goals")
            for ltg in self.conversation_agent.parser.extract_long_term_goals(text):
                if hasattr(self.goal_agent, "add_or_edit_goal"):
                    self.goal_agent.add_or_edit_goal(ltg)

        # Traits (existing)
        if parsed.get("traits"):
            logging.debug("CentralHub: Passing traits to TraitAgent")
            if hasattr(self.trait_agent, "set_traits"):
                self.trait_agent.set_traits(parsed["traits"])

        # Routines (existing)
        if parsed.get("routines"):
            logging.debug("CentralHub: Passing routines to RoutineTrackerAgent")
            if hasattr(self.routine_agent, "set_routines"):
                self.routine_agent.set_routines(parsed["routines"])

        # Impatience Data (NEW)
        if hasattr(self.conversation_agent.parser, "extract_impatience_data"):
            logging.debug("CentralHub: Extracting impatience data")
            impatience = self.conversation_agent.parser.extract_impatience_data(text)
            if hasattr(self.impatience_agent, "detect_impatience"):
                self.impatience_agent.detect_impatience(
                    impatience.get("user_input", ""),
                    impatience.get("emotional_state", "neutral"),
                )

        # Psychological/Affect Data (NEW)
        if hasattr(self.conversation_agent.parser, "extract_psychological_data"):
            logging.debug("CentralHub: Extracting psychological data")
            psych_data = self.conversation_agent.parser.extract_psychological_data(text)
            if hasattr(self.prism_agent, "log_affect"):
                self.prism_agent.log_affect(psych_data)
            if hasattr(self.psyche_agent, "log_affect"):
                self.psyche_agent.log_affect(psych_data)

        # Decisions (NEW)
        if hasattr(self.conversation_agent.parser, "extract_decisions"):
            logging.debug("CentralHub: Extracting decisions")
            for decision in self.conversation_agent.parser.extract_decisions(text):
                if hasattr(self.decision_vault_agent, "log_decision"):
                    d_type = decision.get("type", "unspecified")
                    d_details = decision.get("details", {})
                    self.decision_vault_agent.log_decision(d_type, d_details)

        # Suggestions (Optional, if you want to log suggestions)
        if hasattr(self.conversation_agent.parser, "generate_suggestions"):
            suggestions = self.conversation_agent.parser.generate_suggestions(text)
            if suggestions:
                logging.debug(f"CentralHub: Suggestions: {suggestions}")

    def update_shadow(self, data):
        if self.check_autonomy("shadow_update"):
            self.shadow_agent.update_shadow(data)
        else:
            logging.debug("[CentralHub] Shadow update blocked by consent tier.")

    def run_reflection(self):
        if self.check_autonomy("reflection"):
            shadow = self.shadow_agent.get_refined_shadow()
            return self.reflective_agent.run_full_reflection(shadow)
        else:
            logging.debug("[CentralHub] Reflection blocked by consent tier.")
            return None

    def coach_action(self, energy_summary):
        if self.check_autonomy("coach_action"):
            return self.coach_agent.coach_action(energy_summary)
        else:
            logging.debug("[CentralHub] Coach action blocked by consent tier.")
            return None

    def detect_impatience(self, user_input, emotional_state):
        return self.impatience_agent.detect_impatience(user_input, emotional_state)

    async def run_psyche_analysis(self, user_data: Optional[dict] = None) -> dict:
        if self.psyche_agent and self.agentic_hub and hasattr(self.agentic_hub, "get_shadow_snapshot"):
            shadow_data = await self.agentic_hub.get_shadow_snapshot()
            return await self.psyche_agent.analyze_psyche(shadow_data, user_data or {})
        return {}

    # === Curated data for AgenticHub ===

    def get_shadow_snapshot(self):
        return self.shadow_agent.get_refined_shadow()

    def get_prism_data(self, params):
        return self.prism_agent.retrieve_user_data(params)

    def get_curated_goals(self):
        return self.goal_agent.goals

    def get_curated_traits(self):
        return getattr(self.trait_agent, 'traits', [])

    def get_curated_behavior_summary(self):
        return self.behavioral_agent.get_behavioral_summary()

    def get_curated_routines(self):
        return self.routine_agent.get_routines()

    def get_curated_consent(self):
        return self.consent

    def get_decision_vault_data(self, _=None):
        if self.decision_vault_agent:
            return self.decision_vault_agent.list_all()
        return []
