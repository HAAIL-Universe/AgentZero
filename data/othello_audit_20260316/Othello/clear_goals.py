# clear_goals.py
from modules.goal_manager import GoalManager

if __name__ == "__main__":
    gm = GoalManager()
    gm.clear_all(delete_logs=True, reset_ids=True)