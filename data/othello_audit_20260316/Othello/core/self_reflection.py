"""
core/self_reflection.py

Self-Reflection & Metacognition Engine
Handles intervention logging, outcome tracking, meta reflection summaries.

Author: Julius
"""

import json
import os
import uuid
from datetime import datetime
from core.meta_analysis import MetaAnalysis
from core.shadow_linker import ShadowLinker

INTERVENTION_LOG_PATH = "data/intervention_log.json"
META_LOG_PATH = "data/self_reflection_log.json"

class SelfReflectionEngine:
    def __init__(self):
        self.meta = MetaAnalysis()
        self.shadow_linker = ShadowLinker()

    def _load_json(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            try:
                return json.load(f)
            except:
                return []

    def _save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_intervention(self, nudge_type, reason, context):
        interventions = self._load_json(INTERVENTION_LOG_PATH)
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "nudge_type": nudge_type,
            "reason": reason,
            "context": context,
            "outcome": None,
            "user_feedback": None
        }
        interventions.append(entry)
        self._save_json(INTERVENTION_LOG_PATH, interventions)

        self.shadow_linker.safe_update({"decisions": {"last_intervention": entry}}, consent_level="auto")

        return entry["id"]

    def log_outcome(self, nudge_id, outcome):
        interventions = self._load_json(INTERVENTION_LOG_PATH)
        found = False
        for item in interventions:
            if item["id"] == nudge_id:
                item["outcome"] = outcome
                found = True
                break
        if found:
            self._save_json(INTERVENTION_LOG_PATH, interventions)
            self.shadow_linker.safe_update({"decisions": {"intervention_outcome": {"id": nudge_id, "outcome": outcome}}}, consent_level="auto")

        return found

    def run_reflection(self):
        interventions = self._load_json(INTERVENTION_LOG_PATH)
        summary = self.meta.summarize_reflection(interventions)
        meta_log = self._load_json(META_LOG_PATH)
        meta_log.append(summary)
        self._save_json(META_LOG_PATH, meta_log)

        self.shadow_linker.safe_update({"micro_patterns": {"reflection_summary": summary}}, consent_level="auto")

        return summary

    def explain_intervention(self, nudge_id):
        interventions = self._load_json(INTERVENTION_LOG_PATH)
        for item in interventions:
            if item["id"] == nudge_id:
                return {
                    "reason": item["reason"],
                    "context": item["context"]
                }
        return None

    def wipe_meta_logs(self):
        self._save_json(META_LOG_PATH, [])
        self.shadow_linker.safe_update({"micro_patterns": {"reflection_summary": "logs_wiped"}}, consent_level="auto")
        return True

if __name__ == "__main__":
    s = SelfReflectionEngine()
    id1 = s.log_intervention("walk_reminder", "routine_disruption", {"time_of_day": "afternoon", "mood": "neutral"})
    print(f"Logged intervention: {id1}")
    s.log_outcome(id1, "acted")
    print("Outcome logged.")
    summary = s.run_reflection()
    print("Reflection summary:")
    print(json.dumps(summary, indent=2))
