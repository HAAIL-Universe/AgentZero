# interface/suggestion_display.py

from datetime import datetime

class SuggestionDisplay:
    def __init__(self):
        # In a future version, suggestions could be fetched from PRISM state or habit tracker
        self.suggestions = self._get_default_suggestions()

    def _get_default_suggestions(self):
        """Return simple time-aware default suggestions."""
        hour = datetime.now().hour
        if hour < 12:
            return ["Take a deep breath — you’ve got time to shape the day.",
                    "Consider setting 1 small goal before noon."]
        elif hour < 18:
            return ["How’s your focus this afternoon?",
                    "A quick stretch or water break might help you reset."]
        else:
            return ["Start winding down with something relaxing.",
                    "Reflect on one small win from today."]

    def set_suggestions(self, suggestions: list):
        self.suggestions = suggestions

    def display(self):
        if not self.suggestions:
            print("\n✨ No suggestions at the moment.\n")
            return

        print("\n📌 Key Nudges & Pointers:")
        for s in self.suggestions:
            print(f" - {s}")
        print("")
