import time
import logging

class Safeguard:
    def __init__(self):
        self.agent_core_prompt = (
            "You are Fellow, a conscientious AI assistant..."
        )
        self.current_mode = 'ACP'
        self.last_ground_time = time.time()
        self.ground_interval_seconds = 3600  # e.g., 1 hour
        self.logger = logging.getLogger("FELLO_SAFEGUARD")

    def should_ground(self) -> bool:
        """
        Returns True if it's time for soft grounding (based on elapsed time).
        """
        return (time.time() - self.last_ground_time) > self.ground_interval_seconds

    def soft_ground(self):
        """
        Perform soft grounding: step out to ACP briefly to re-anchor.
        """
        self.logger.info("Performing soft grounding to ACP.")
        self.current_mode = 'ACP'
        self.last_ground_time = time.time()
        # Return grounding prompt to prepend
        return self.agent_core_prompt

    def hard_reset(self):
        """
        User-triggered hard reset:
        1) Step fully into ACP mode
        2) Run 'self-profile' phase (simulate)
        3) Reset grounding timer
        """
        self.logger.info("Hard reset triggered by user.")
        self.current_mode = 'ACP'
        self.last_ground_time = time.time()
        
        # Simulated self-profile prompt - can be expanded later
        self_profile_prompt = (
            "I am Fellow, an AI assistant designed to support and respect boundaries. "
            "I maintain a clear distinction between my identity and the user's."
        )
        
        # Return a sequence of prompts or messages for self-profile phase
        return [self.agent_core_prompt, self_profile_prompt]

    def switch_mode(self, mode):
        if mode not in ['ACP', 'UP']:
            self.logger.warning(f"Invalid mode switch attempt: {mode}")
            return
        self.current_mode = mode
        self.logger.info(f"Switched to mode: {mode}")

    def get_prompt(self, user_persona_context=None):
        """
        Return the appropriate prompt based on mode and grounding timer.
        """
        if self.should_ground():
            return self.soft_ground()
        
        if self.current_mode == 'ACP':
            return self.agent_core_prompt
        else:
            return user_persona_context or "[USER PERSONA CONTEXT PLACEHOLDER]"
