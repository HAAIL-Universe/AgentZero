"""
core/privacy.py

Privacy and consent logic for FELLO.
Centralizes checks for user consent, privacy tier management, data erasure, and audit logging.
All behavioral/analytic features must call this before logging/tracking sensitive data.

Author: Julius
"""

import yaml
import os

CONSENT_PATH = "config/consent.yaml"
AUDIT_LOG = "data/privacy_audit_log.json"

def load_consent():
    if not os.path.exists(CONSENT_PATH):
        return {"tier": 1, "feature_toggles": {}, "audit": []}
    with open(CONSENT_PATH, "r") as f:
        return yaml.safe_load(f)

def save_audit(event):
    import json
    if not os.path.exists(AUDIT_LOG):
        audit = []
    else:
        with open(AUDIT_LOG, "r") as f:
            try:
                audit = json.load(f)
            except Exception:
                audit = []
    audit.append(event)
    with open(AUDIT_LOG, "w") as f:
        json.dump(audit, f, indent=2)

def check_consent(feature):
    """
    Checks if the given feature is allowed under current consent tier/toggles.
    :param feature: str
    :return: bool
    """
    consent = load_consent()
    tier = consent.get("tier", 1)
    toggles = consent.get("feature_toggles", {})
    # Example: feature toggles override tier
    if toggles.get(feature) is not None:
        return toggles[feature]
    # Default: restrict by tier (could make a dict of features: tier_min)
    TIER_FEATURES = {
        1: ["basic_nudges", "surface_analysis"],
        2: ["advanced_nudges", "pattern_analysis"],
        3: ["deep_psychology", "manipulative_nudges"]
    }
    allowed = []
    for t in range(1, tier+1):
        allowed.extend(TIER_FEATURES.get(t, []))
    return feature in allowed

def erase_all_data():
    """
    Nukes all local data for privacy reset.
    """
    import shutil
    if os.path.exists("data/"):
        shutil.rmtree("data/")
    os.makedirs("data/", exist_ok=True)
    save_audit({"event": "all_data_erased", "timestamp": "reset"})

def log_privacy_event(event_type, detail):
    """
    Logs a privacy-related event for audit trail.
    """
    from datetime import datetime
    event = {
        "event_type": event_type,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
    save_audit(event)

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    print("Consent for 'deep_psychology':", check_consent("deep_psychology"))
    log_privacy_event("check_consent", {"feature": "deep_psychology"})
