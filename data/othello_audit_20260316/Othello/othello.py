from modules.agentic_agents.shadow_agent import ShadowAgent

class Othello:
    """
    Othello: The ultimate safeguarding agent.
    Filters and validates all actions from FELLO to ensure they are safe, ethical,
    and comprehensible to humans. It is the final gatekeeper before any action
    reaches the user.
    """

    def __init__(self, central_hub=None, agentic_hub=None):
        self.shadow_agent = ShadowAgent(central_hub=central_hub, agentic_hub=agentic_hub)

    def validate_action(self, action_data):
        """
        Validates the action decided by FELLO before it reaches the user.
        - Ensures action is ethical and safe.
        - Simplifies complex data for human comprehension.
        """
        if not self.is_ethical(action_data):
            return {"error": "Action rejected due to ethical concerns."}

        if not self.is_safe(action_data):
            return {"error": "Action rejected due to safety concerns."}

        return self.simplify_for_user(action_data)

    def is_ethical(self, action_data):
        """
        Check whether the action is ethical.
        Example rule: no content that implies harm.
        """
        suggestion = action_data.get("suggestion", "").lower()
        return "harm" not in suggestion

    def is_safe(self, action_data):
        """
        Check whether the action respects user safety and privacy.
        Example rule: high-risk actions are automatically blocked.
        """
        return action_data.get("risk_level", "low") != "high"

    def simplify_for_user(self, action_data):
        """
        Simplifies complex action data into a user-friendly format.
        """
        return {
            "action": action_data.get("suggestion", "No suggestion available"),
            "priority": action_data.get("priority", "Normal"),
            "reasoning": action_data.get("reasoning", "No reasoning available"),
            "recommendation": action_data.get("recommendation", "No recommendation available")
        }

    def update_shadow(self, action_data):
        """
        Log the validated action to the shadow for auditing and learning.
        """
        self.shadow_agent.update_shadow({"actions": action_data})
        self.shadow_agent.audit_shadow_state()
