import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

TRAIT_LOG_FILE_SHADOW = "data/shadow_traits.json"
TRAIT_LOG_FILE_PERSONA = "data/persona_traits.json"
TRAIT_STATE_FILE = "data/trait_state.json"

CONFLICTING_TRAITS = {
    "confident": "insecure",
    "impulsive": "disciplined",
    "patient": "impatient",
    "anxious": "calm"
}

class TraitAgent:
    """
    TraitAgent: Tracks user traits over time with multi-dimensional profiles.
    Supports time-weighted decay, ML/RL hooks, reward scaling, and syncs Shadow vs Persona traits.
    """

    def __init__(self, hub=None):
        self.hub = hub
        self.shadow_traits = self._load_trait_log(TRAIT_LOG_FILE_SHADOW)
        self.persona_traits = self._load_trait_log(TRAIT_LOG_FILE_PERSONA)
        self.reward_system = {"core_trait_growth": 10, "linked_growth": 15}
        self.traits = {}  # for RL/ML delta tracking
        self.history = []

    def _load_trait_log(self, filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r") as f:
            return json.load(f)

    def _save_trait_log(self, filepath, data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def save_traits(self):
        """Save current trait delta state to persistent file (for ML/RL history tracking)"""
        os.makedirs(os.path.dirname(TRAIT_STATE_FILE), exist_ok=True)
        with open(TRAIT_STATE_FILE, "w") as f:
            json.dump(self.traits, f, indent=4)

    def log_trait(self, trait_name: str, intensity: float, context: str = "general",
                  source: str = "persona", trait_type: Optional[str] = None,
                  goal_link: Optional[str] = None, trait_link: Optional[str] = None):

        filepath = TRAIT_LOG_FILE_PERSONA if source == "persona" else TRAIT_LOG_FILE_SHADOW
        trait_log = self.persona_traits if source == "persona" else self.shadow_traits

        entry = {
            "trait": trait_name,
            "intensity": intensity,
            "context": context,
            "trait_type": trait_type,
            "linked_goals": [goal_link] if goal_link else [],
            "linked_traits": [trait_link] if trait_link else [],
            "timestamp": datetime.now().isoformat(),
            "stability_index": 0.5  # Starting midpoint
        }

        trait_log.append(entry)
        self._save_trait_log(filepath, trait_log)

        if self.hub:
            self.hub.update_shadow({f"trait_{source}": entry})
        else:
            print(f"[TraitAgent] No hub assigned — shadow update skipped.")

    def update_trait(self, trait_name: str, delta: float, context: str = "", source: str = "system"):
        now = datetime.now().isoformat()
        if trait_name not in self.traits:
            self.traits[trait_name] = {
                "score": delta,
                "history": [],
                "last_updated": now,
                "sources": [source]
            }
        else:
            self.traits[trait_name]["score"] += delta
            self.traits[trait_name]["last_updated"] = now
            self.traits[trait_name]["sources"].append(source)

        self.traits[trait_name]["history"].append({
            "timestamp": now,
            "delta": delta,
            "context": context,
            "source": source
        })

        self.history.append({
            "trait": trait_name,
            "timestamp": now,
            "delta": delta,
            "context": context,
            "source": source
        })

        self.save_traits()

        if self.hub:
            self.hub.update_shadow({"traits": {trait_name: self.traits[trait_name]}})
        else:
            print(f"[TraitAgent] No hub assigned — shadow update skipped.")

        print(f"[TraitAgent] Updated trait: {trait_name} | Δ{delta} | Source: {source}")

    def decay_inactive_traits(self):
        cutoff = datetime.now() - timedelta(days=5)
        for trait_list in [self.shadow_traits, self.persona_traits]:
            for trait in trait_list:
                last_seen = datetime.fromisoformat(trait["timestamp"])
                if last_seen < cutoff:
                    trait["stability_index"] = max(0.0, trait["stability_index"] - 0.05)

        self._save_trait_log(TRAIT_LOG_FILE_SHADOW, self.shadow_traits)
        self._save_trait_log(TRAIT_LOG_FILE_PERSONA, self.persona_traits)

    def reward_user_for_trait_improvement(self, trait_name: str, weight: float = 1.0):
        reward_points = self.reward_system["core_trait_growth"] * weight
        if self.hub:
            self.hub.update_shadow({"reward_points": reward_points})
        print(f"[TraitAgent] User rewarded with {reward_points} points for trait growth: {trait_name}.")

    def detect_conflicts(self):
        traits = set(t["trait"] for t in self.persona_traits + self.shadow_traits)
        conflicts = []
        for t1, t2 in CONFLICTING_TRAITS.items():
            if t1 in traits and t2 in traits:
                conflicts.append((t1, t2))
        return conflicts

    def predict_trait_evolution(self):
        # Stub for future ML model
        return {"confidence": "+8%", "impulsivity": "-2%"}

    def log_trait_growth(self, trait_name: str, old_value: float, new_value: float):
        growth_log = {
            "trait": trait_name,
            "from": old_value,
            "to": new_value,
            "timestamp": datetime.now().isoformat()
        }
        os.makedirs("data", exist_ok=True)
        path = "data/trait_growth_log.json"
        try:
            with open(path, "r") as f:
                logs = json.load(f)
        except FileNotFoundError:
            logs = []
        except Exception as e:
            print(f"[TraitAgent] Error loading growth log: {e}")
            logs = []
        logs.append(growth_log)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[TraitAgent] Growth logged: {growth_log}")

    def get_trait_summary(self, include_shadow=False):
        summary = {"persona": self.persona_traits}
        if include_shadow:
            summary["shadow"] = self.shadow_traits
        return summary
    
    def get_all_traits(self) -> Dict[str, Dict]:
        """
        Return the full internal trait state used by RL/ML hooks.
        """
        return self.traits

    def get_trait_context(self) -> Dict[str, Dict]:
        """
        Return a summarized structure of all tracked traits,
        including score, last update, and source metadata.
        """
        context_summary = {}
        for trait, data in self.traits.items():
            context_summary[trait] = {
                "score": data["score"],
                "last_updated": data["last_updated"],
                "sources": list(set(data["sources"]))  # Remove duplicates
            }
        return context_summary

    def build_snapshot(self) -> Dict:
        """
        Build a full snapshot of the trait state including shadow/persona logs,
        RL trait tracking, and historical deltas for deep insight.
        """
        return {
            "traits": self.traits,
            "shadow_traits": self.shadow_traits,
            "persona_traits": self.persona_traits,
            "history": self.history
        }
    
    def set_traits(self, traits: Dict[str, float], context: str = "parsed_batch", source: str = "persona"):
        """
        Bulk set traits with associated delta values.
        Integrates into existing update_trait and log_trait pipeline.
        
        Args:
            traits (Dict[str, float]): Trait name → score delta
            context (str): Context in which traits were parsed
            source (str): 'persona' or 'shadow'
        """
        if not isinstance(traits, dict):
            print("[TraitAgent] set_traits received invalid input. Must be dict of trait_name: delta.")
            return
        for trait_name, delta in traits.items():
            self.update_trait(trait_name, delta, context=context, source=source)
            self.log_trait(trait_name, intensity=delta, context=context, source=source)
    
    def analyze_text(self, text: str):
        """
        Analyze a chunk of text for trait-related signals.
        For now, this is a stub; future: NLP/embedding, trait detection, RL reward hooks.
        """
        # Example: rudimentary trait scoring (future = real NLP)
        trait_keywords = {
            "resilient": ["bounce back", "persistent", "handled stress", "resilient"],
            "curious": ["curious", "explored", "questioned"],
            "impatient": ["impatient", "couldn't wait", "frustrated"],
            "calm": ["calm", "relaxed", "handled well"]
        }
        summary = {}
        text_lower = text.lower()
        for trait, keywords in trait_keywords.items():
            if any(kw in text_lower for kw in keywords):
                self.update_trait(trait, 1.0, context="auto_analyze_text", source="reflection")
                summary[trait] = summary.get(trait, 0) + 1

        print(f"[TraitAgent] analyze_text completed. Found: {list(summary.keys())}")
        return summary