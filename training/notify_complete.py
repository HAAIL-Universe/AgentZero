#!/usr/bin/env python
"""
Training completion notifier -- watches for Magistus model and sends Telegram message.
Runs as a background process. Checks every 5 minutes.
AI-Generated | Claude (Anthropic) | AgentZero | 2026-03-11
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse

# ─────────────────────────────────────────────
# FILL THESE IN BEFORE RUNNING
BOT_TOKEN = "8656640779:AAF1BBSTdYzknX7L6IML-7sBIjEBqT35exA"
CHAT_ID   = "8160223830"
# ─────────────────────────────────────────────

MAGISTUS_MODEL  = "Z:/AgentZero/models/phi3-magistus/config.json"
LORA_ADAPTERS   = "Z:/AgentZero/models/phi3-magistus-lora"
CHECK_INTERVAL  = 300  # seconds between checks (5 minutes)
NOTIFIED_FLAG   = "Z:/AgentZero/training/.notified"


def send_telegram(message):
    """Send a Telegram message via Bot API."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[notifier] Telegram send failed: {e}")
        return False


def check_training_status():
    """Return a status dict about the training."""
    model_done   = os.path.exists(MAGISTUS_MODEL)
    lora_exists  = os.path.exists(LORA_ADAPTERS)

    lora_files = []
    if lora_exists:
        lora_files = [f for f in os.listdir(LORA_ADAPTERS)
                      if os.path.isfile(os.path.join(LORA_ADAPTERS, f))]

    return {
        "model_done":  model_done,
        "lora_exists": lora_exists,
        "lora_files":  lora_files,
    }


def build_completion_message(status):
    """Build the Telegram notification message."""
    lines = [
        "✅ *Magistus training complete!*",
        "",
        "The Phi-3 fine-tuning finished successfully.",
        "",
        f"📁 Model saved to: `Z:\\AgentZero\\models\\phi3-magistus\\`",
        f"🔧 LoRA adapter files: {len(status['lora_files'])}",
        "",
        "Next steps:",
        "• Run `evaluate_magistus.py` to compare base vs fine-tuned",
        "• Run `launch_magistus.py` to start a Magistus session",
        "• Delete `Z:\\AgentZero\\STOP` to resume AgentZero",
    ]
    return "\n".join(lines)


def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("[notifier] ERROR: Set BOT_TOKEN and CHAT_ID before running.")
        print("[notifier] Edit Z:/AgentZero/training/notify_complete.py")
        sys.exit(1)

    # Don't double-notify
    if os.path.exists(NOTIFIED_FLAG):
        print("[notifier] Already notified. Delete .notified to re-arm.")
        sys.exit(0)

    print(f"[notifier] Watching for Magistus model at:")
    print(f"  {MAGISTUS_MODEL}")
    print(f"[notifier] Checking every {CHECK_INTERVAL // 60} minutes.")
    print(f"[notifier] Will send Telegram message to chat {CHAT_ID}")
    print(f"[notifier] Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    checks = 0
    while True:
        checks += 1
        status = check_training_status()

        if status["model_done"]:
            print(f"\n[notifier] *** TRAINING COMPLETE at {time.strftime('%H:%M:%S')} ***")
            print(f"[notifier] Sending Telegram notification...")

            msg = build_completion_message(status)
            success = send_telegram(msg)

            if success:
                print("[notifier] Notification sent successfully!")
                # Write flag so we don't re-notify
                with open(NOTIFIED_FLAG, 'w') as f:
                    f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                print("[notifier] Notification FAILED. Check token/chat_id.")
                # Retry next cycle rather than exiting
                time.sleep(CHECK_INTERVAL)
                continue

            break  # Done

        else:
            # Still waiting -- show progress
            elapsed_hint = ""
            if status["lora_exists"] and status["lora_files"]:
                elapsed_hint = f" | LoRA adapters: {len(status['lora_files'])} files"

            print(f"[notifier] Check #{checks} at {time.strftime('%H:%M:%S')} "
                  f"-- training in progress{elapsed_hint}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
