import os
import json
from datetime import datetime

def log_audit(source: str, action: str, data: dict):
    """Log audit data to a JSONL file with timestamp."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "action": action,
        "data": data
    }

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/audit_{datetime.now().strftime('%Y-%m-%d')}.log"

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[AUDIT] {source}.{action}: {log_entry}")
