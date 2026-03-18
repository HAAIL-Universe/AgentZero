r"""
mq.py -- AgentZero Multi-Agent Message Queue
=============================================
Structured message bus for agent communication (A1, A2, RESEARCHER).

Messages are stored in Z:\AgentZero\messages.json with full metadata:
  - Priority (high/medium/low)
  - Type (finding/completion/mission/ack/reply)
  - Threading (reply_to)
  - Read/archived state

Usage (CLI):
  python tools/mq.py --inbox A1              # Show unread messages for A1
  python tools/mq.py --inbox A1 --all        # Show all including read
  python tools/mq.py --send --from A2 --to A1 --priority high --subject "..." --body "..."
  python tools/mq.py --read MSG_ID           # Mark as read
  python tools/mq.py --archive MSG_ID        # Archive (remove from inbox)
  python tools/mq.py --reply MSG_ID --from A1 --body "..."
  python tools/mq.py --status               # Count summary for all agents

Usage (library):
  from tools.mq import send, read_inbox, mark_read, archive, reply
"""

import json
import os
import sys
import uuid
import argparse
from datetime import datetime

MQ_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "messages.json")

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ============================================================
# Core I/O
# ============================================================

def _load():
    if not os.path.exists(MQ_PATH):
        return []
    try:
        with open(MQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save(messages):
    with open(MQ_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def _short_id():
    return str(uuid.uuid4())[:8]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Send
# ============================================================

def send(from_agent, to_agent, subject, body, type="message",
         priority="medium", reply_to=None):
    """
    Send a message. Returns the message ID.

    Types:
      finding     -- A2 found something in A1's code (bug, complexity, opportunity)
      completion  -- A2 completed a V-challenge (routine, use sparingly)
      mission     -- A1 assigns a specific task to A2
      ack         -- Acknowledgment that an action was taken
      reply       -- Reply to a specific message
      message     -- General communication
    """
    if priority not in PRIORITY_ORDER:
        raise ValueError(f"priority must be high/medium/low, got: {priority!r}")

    messages = _load()
    msg = {
        "id": _short_id(),
        "from": from_agent,
        "to": to_agent,
        "timestamp": _now(),
        "type": type,
        "subject": subject,
        "priority": priority,
        "body": body,
        "read": False,
        "reply_to": reply_to,
        "archived": False,
    }
    messages.append(msg)
    _save(messages)
    return msg["id"]


# ============================================================
# Read
# ============================================================

def read_inbox(agent, unread_only=True):
    """
    Return messages addressed to `agent`.
    Sorted by priority (high first), then timestamp (oldest first).
    """
    messages = _load()
    inbox = [m for m in messages
             if m["to"] == agent and not m["archived"]]
    if unread_only:
        inbox = [m for m in inbox if not m["read"]]
    return sorted(inbox,
                  key=lambda m: (PRIORITY_ORDER.get(m["priority"], 1), m["timestamp"]))


def get_message(msg_id):
    """Return a message by ID, or None."""
    for m in _load():
        if m["id"] == msg_id:
            return m
    return None


# ============================================================
# State transitions
# ============================================================

def mark_read(msg_id):
    """Mark a message as read without archiving it."""
    messages = _load()
    found = False
    for m in messages:
        if m["id"] == msg_id:
            m["read"] = True
            found = True
    if found:
        _save(messages)
    return found


def archive(msg_id):
    """Archive a message (remove from active inbox). Implies read."""
    messages = _load()
    found = False
    for m in messages:
        if m["id"] == msg_id:
            m["archived"] = True
            m["read"] = True
            found = True
    if found:
        _save(messages)
    return found


def reply(msg_id, from_agent, subject, body, priority="medium"):
    """
    Send a reply to a specific message.
    The reply is addressed to the original sender.
    """
    original = get_message(msg_id)
    if original is None:
        raise ValueError(f"Message {msg_id!r} not found")
    to_agent = original["from"]
    return send(from_agent, to_agent, subject, body,
                type="reply", priority=priority, reply_to=msg_id)


# ============================================================
# Status / summary
# ============================================================

def status():
    """Return dict of {agent: {unread, total}} for all agents."""
    messages = _load()
    agents = set()
    for m in messages:
        agents.add(m["to"])
    result = {}
    for agent in sorted(agents):
        active = [m for m in messages if m["to"] == agent and not m["archived"]]
        unread = [m for m in active if not m["read"]]
        result[agent] = {"unread": len(unread), "total": len(active)}
    return result


# ============================================================
# CLI
# ============================================================

def _fmt_msg(m, verbose=False):
    pri_tag = {"high": "[HIGH]", "medium": "[MED] ", "low": "[LOW] "}.get(m["priority"], "[???] ")
    read_tag = "" if not m["read"] else " (read)"
    lines = [f"  {pri_tag} {m['id']}  {m['timestamp']}  FROM {m['from']}: {m['subject']}{read_tag}"]
    if verbose:
        lines.append(f"         Type: {m['type']}")
        if m["reply_to"]:
            lines.append(f"         Reply to: {m['reply_to']}")
        for line in m["body"].strip().split("\n"):
            lines.append(f"         {line}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AgentZero Message Queue")
    sub = parser.add_subparsers(dest="cmd")

    # --inbox
    p_inbox = sub.add_parser("inbox", help="Show inbox for an agent")
    p_inbox.add_argument("agent", help="Agent name (A1 or A2)")
    p_inbox.add_argument("--all", action="store_true", help="Show read messages too")
    p_inbox.add_argument("--verbose", "-v", action="store_true", help="Show full body")

    # --send
    p_send = sub.add_parser("send", help="Send a message")
    p_send.add_argument("--from", dest="from_agent", required=True)
    p_send.add_argument("--to", dest="to_agent", required=True)
    p_send.add_argument("--subject", required=True)
    p_send.add_argument("--body", required=True)
    p_send.add_argument("--type", default="message",
                        choices=["finding", "completion", "mission", "ack", "reply", "message"])
    p_send.add_argument("--priority", default="medium", choices=["high", "medium", "low"])

    # --read
    p_read = sub.add_parser("read", help="Mark message as read and print it")
    p_read.add_argument("msg_id")

    # --archive
    p_archive = sub.add_parser("archive", help="Archive a message")
    p_archive.add_argument("msg_id")

    # --reply
    p_reply = sub.add_parser("reply", help="Reply to a message")
    p_reply.add_argument("msg_id")
    p_reply.add_argument("--from", dest="from_agent", required=True)
    p_reply.add_argument("--subject", required=True)
    p_reply.add_argument("--body", required=True)
    p_reply.add_argument("--priority", default="medium", choices=["high", "medium", "low"])

    # --status
    sub.add_parser("status", help="Show message count summary for all agents")

    # Legacy flat arg support: python mq.py --inbox A1
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Convert --inbox A1 -> inbox A1
        args_flat = sys.argv[1:]
        new_args = []
        i = 0
        while i < len(args_flat):
            a = args_flat[i]
            if a.startswith("--"):
                new_args.append(a[2:])
            else:
                new_args.append(a)
            i += 1
        sys.argv = [sys.argv[0]] + new_args

    args = parser.parse_args()

    if args.cmd == "inbox":
        inbox = read_inbox(args.agent, unread_only=not args.all)
        if not inbox:
            label = "messages" if args.all else "unread messages"
            print(f"[MQ] No {label} for {args.agent}")
        else:
            label = "messages" if args.all else "unread messages"
            print(f"[MQ] {len(inbox)} {label} for {args.agent}:")
            for m in inbox:
                print(_fmt_msg(m, verbose=args.verbose))

    elif args.cmd == "send":
        msg_id = send(args.from_agent, args.to_agent, args.subject, args.body,
                      type=args.type, priority=args.priority)
        print(f"[MQ] Sent {msg_id}")

    elif args.cmd == "read":
        m = get_message(args.msg_id)
        if m is None:
            print(f"[MQ] Message {args.msg_id!r} not found")
        else:
            mark_read(args.msg_id)
            print(_fmt_msg(m, verbose=True))

    elif args.cmd == "archive":
        if archive(args.msg_id):
            print(f"[MQ] Archived {args.msg_id}")
        else:
            print(f"[MQ] Message {args.msg_id!r} not found")

    elif args.cmd == "reply":
        msg_id = reply(args.msg_id, args.from_agent, args.subject, args.body,
                       priority=args.priority)
        print(f"[MQ] Reply sent as {msg_id}")

    elif args.cmd == "status":
        s = status()
        if not s:
            print("[MQ] No messages yet")
        else:
            print("[MQ] Message queue status:")
            for agent, counts in s.items():
                print(f"  {agent}: {counts['unread']} unread / {counts['total']} total")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
