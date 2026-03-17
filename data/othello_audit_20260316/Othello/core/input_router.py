from modules.impatience_detector import ImpatienceDetector
from core.logger import FelloLogger

class InputRouter:
    """
    Routes incoming user input through FELLO’s processing pipeline.
    Detects impatience, mood, and routes to appropriate engine/module.
    Fully modular for future detectors and hooks.
    """
    
    def __init__(self, enable_impatience=True):
        """
        Initializes the InputRouter.
        :param enable_impatience: Bool flag to enable/disable impatience detection.
        """
        self.enable_impatience = enable_impatience
        self.impatience_detector = ImpatienceDetector() if enable_impatience else None
        self.logger = FelloLogger()
    
    def route_input(self, input_text, context=None):
        """
        Main entry point. Routes input through detection pipeline and returns tags + clean input.
        :param input_text: The raw user input string.
        :param context: Optional dict with additional routing context.
        :return: dict with processed input, tags, and routing target.
        """
        tags = {}
        
        # Impatience detection
        if self.enable_impatience and self.impatience_detector:
            impatience_level = self.impatience_detector.analyze(input_text)
            tags['impatience_level'] = impatience_level
        
        # Future: mood detection, energy tagging, etc.
        # tags['mood'] = self.detect_mood(input_text)
        
        self.log_input(input_text, tags)
        
        # Determine routing target (stubbed for now)
        routing_target = 'default_engine'
        
        return {
            'input': input_text,
            'tags': tags,
            'routing_target': routing_target
        }
    
    def log_input(self, input_text, tags):
        """
        Logs the input and tags for traceability.
        :param input_text: The raw input.
        :param tags: Dict of tags applied during processing.
        """
        self.logger.log_event('input_received', {
            'input': input_text,
            'tags': tags
        })

    # TODO: Add detect_mood(), pre_processors() as needed for future expansion.
