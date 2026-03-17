class TrainingModeController:
    def __init__(self):
        self.active = False

    def toggle(self):
        self.active = not self.active
        state = "ON" if self.active else "OFF"
        print(f"🎯 Training Mode is now {state}.")

    def is_active(self):
        return self.active
