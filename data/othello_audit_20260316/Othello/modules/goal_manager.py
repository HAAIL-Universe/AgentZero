# modules/goal_manager.py

from typing import List, Optional, TypedDict, Literal
from datetime import datetime
from pathlib import Path
import json


class GoalEntry(TypedDict):
    """
    Typed representation of a single goal in Othello.
    """
    id: int
    text: str
    deadline: Optional[str]
    created_at: str


class GoalNote(TypedDict):
    """
    A single note attached to a goal's conversation log.
    """
    timestamp: str
    role: Literal["user", "othello", "system"]
    content: str


class GoalManager:
    """
    GoalManager for Othello.

    - Stores goals in memory AND on disk (data/goals.json).
    - Maintains an 'active_goal_id' so the API can route messages.
    - For each goal, keeps a JSONL log in data/goal_logs/goal_<id>.jsonl
      containing the conversation notes/updates for that goal.
    """

    def __init__(self) -> None:
        self.goals: List[GoalEntry] = []
        self._next_id: int = 1
        self.active_goal_id: Optional[int] = None

        root = Path(__file__).resolve().parent.parent
        self._store_path = root / "data" / "goals.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        self._logs_dir = root / "data" / "goal_logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._load_from_disk()

    # ------------------------------------------------------------------
    # Persistence: goals list
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> None:
        if not self._store_path.exists():
            return

        try:
            data = json.load(self._store_path.open("r", encoding="utf-8"))
            if isinstance(data, list):
                self.goals = []
                max_id = 0
                for raw in data:
                    if not isinstance(raw, dict) or "text" not in raw:
                        continue

                    entry: GoalEntry = {
                        "id": int(raw.get("id", 0)) or 0,
                        "text": str(raw["text"]),
                        "deadline": raw.get("deadline"),
                        "created_at": str(
                            raw.get("created_at", datetime.utcnow().isoformat() + "Z")
                        ),
                    }
                    self.goals.append(entry)
                    if entry["id"] > max_id:
                        max_id = entry["id"]

                self._next_id = max_id + 1 if max_id > 0 else 1
                print(f"[GoalManager] Loaded {len(self.goals)} goals from disk.")
        except Exception as e:
            print(f"[GoalManager] Failed to load goals from disk: {e}")

    def _save_to_disk(self) -> None:
        try:
            with self._store_path.open("w", encoding="utf-8") as f:
                json.dump(self.goals, f, indent=2)
        except Exception as e:
            print(f"[GoalManager] Failed to save goals to disk: {e}")

    # ------------------------------------------------------------------
    # Helpers for per-goal log "folders"
    # ------------------------------------------------------------------
    def _log_path(self, goal_id: int) -> Path:
        return self._logs_dir / f"goal_{goal_id}.jsonl"

    def add_note_to_goal(self, goal_id: int, role: str, content: str) -> None:
        """
        Append a note to this goal's log file.
        """
        if role not in ("user", "othello", "system"):
            role = "system"

        if self.get_goal(goal_id) is None:
            # Ignore notes for unknown goals.
            return

        note: GoalNote = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,  # type: ignore[assignment]
            "content": content,
        }
        path = self._log_path(goal_id)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(note) + "\n")
        except Exception as e:
            print(f"[GoalManager] Failed to write note for goal {goal_id}: {e}")

    def get_recent_notes(self, goal_id: int, max_notes: int = 10) -> List[GoalNote]:
        """
        Load up to `max_notes` most recent notes from the goal's log.
        """
        path = self._log_path(goal_id)
        if not path.exists():
            return []

        notes: List[GoalNote] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        note = json.loads(line)
                        if isinstance(note, dict) and "content" in note:
                            notes.append(note)  # type: ignore[list-item]
                    except Exception:
                        continue
        except Exception as e:
            print(f"[GoalManager] Failed to read notes for goal {goal_id}: {e}")
            return []

        return notes[-max_notes:]

    def build_goal_context(self, goal_id: int, max_notes: int = 10) -> str:
        """
        Construct a text block summarising the goal + recent notes
        for use as LLM context.
        """
        goal = self.get_goal(goal_id)
        if goal is None:
            return ""

        lines = [
            f"Active Goal #{goal['id']}: {goal['text']}",
        ]
        if goal.get("deadline"):
            lines.append(f"Deadline: {goal['deadline']}")
        lines.append("")

        notes = self.get_recent_notes(goal_id, max_notes=max_notes)
        if notes:
            lines.append("Recent updates and conversation about this goal:")
            for n in notes:
                ts = n.get("timestamp", "")
                role = n.get("role", "system")
                content = n.get("content", "")
                lines.append(f"- [{role} @ {ts}]: {content}")
        else:
            lines.append("No prior notes for this goal yet.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _allocate_id(self) -> int:
        goal_id = self._next_id
        self._next_id += 1
        return goal_id

    # ------------------------------------------------------------------
    # Public API: goals
    # ------------------------------------------------------------------
    def add_goal(self, text: str, deadline: Optional[str] = None) -> GoalEntry:
        """
        Add a new goal and return the stored entry.
        """
        entry: GoalEntry = {
            "id": self._allocate_id(),
            "text": text,
            "deadline": deadline,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self.goals.append(entry)
        self._save_to_disk()
        print(f"[GoalManager] Added goal: {entry}")
        return entry

    def list_goals(self) -> List[GoalEntry]:
        """Return all goals recorded so far."""
        return list(self.goals)

    def get_goal(self, goal_id: int) -> Optional[GoalEntry]:
        """Return a single goal by ID, or None if not found."""
        for g in self.goals:
            if g["id"] == goal_id:
                return g
        return None

    def update_goal(
        self,
        goal_id: int,
        *,
        text: Optional[str] = None,
        deadline: Optional[str] = None,
    ) -> Optional[GoalEntry]:
        """
        Update an existing goal in-place. Returns the updated goal, or None
        if the ID doesn't exist.
        """
        goal = self.get_goal(goal_id)
        if goal is None:
            return None

        if text is not None:
            goal["text"] = text
        if deadline is not None:
            goal["deadline"] = deadline

        self._save_to_disk()
        print(f"[GoalManager] Updated goal {goal_id}: {goal}")
        return goal

    def delete_goal(self, goal_id: int) -> bool:
        """
        Remove a goal by ID. Returns True if deleted, False if not found.
        """
        for idx, g in enumerate(self.goals):
            if g["id"] == goal_id:
                removed = self.goals.pop(idx)
                self._save_to_disk()
                print(f"[GoalManager] Deleted goal {goal_id}: {removed}")
                return True
        return False

    # ------------------------------------------------------------------
    # Active goal handling
    # ------------------------------------------------------------------
    def set_active_goal(self, goal_id: int) -> bool:
        """
        Mark a goal as active for the current session.
        """
        if self.get_goal(goal_id) is None:
            return False
        self.active_goal_id = goal_id
        print(f"[GoalManager] Active goal set to #{goal_id}")
        return True

    def get_active_goal(self) -> Optional[GoalEntry]:
        if self.active_goal_id is None:
            return None
        return self.get_goal(self.active_goal_id)

    # ------------------------------------------------------------------
    # Lookup by name (for "talk about <goal name>")
    # ------------------------------------------------------------------
    def find_goal_by_name(self, query: str) -> Optional[GoalEntry]:
        """
        Find a goal whose text matches the query (case-insensitive substring).
        This lets the user refer to a goal by its title instead of ID.
        """
        q = query.strip().lower()
        if not q:
            return None

        # Simple heuristic: prefer goals where the title starts with the query,
        # then fall back to any substring match.
        starts_with = []
        contains = []

        for g in self.goals:
            title = g["text"].strip().lower()
            if title.startswith(q):
                starts_with.append(g)
            elif q in title:
                contains.append(g)

        if starts_with:
            return starts_with[-1]  # latest that starts with query
        if contains:
            return contains[-1]     # latest that contains query
        return None

    def find_goal_by_id_or_name(self, identifier: str | int) -> Optional[GoalEntry]:
        """
        Convenience helper: try interpreting the identifier as an ID first,
        then as a name query.
        """
        # Try numeric ID first
        try:
            if isinstance(identifier, str):
                num = int(identifier)
            else:
                num = int(identifier)
            goal = self.get_goal(num)
            if goal is not None:
                return goal
        except (ValueError, TypeError):
            pass

        # Fallback: use as name query
        if isinstance(identifier, str):
            return self.find_goal_by_name(identifier)
        return None

    # ------------------------------------------------------------------
    # Hard reset (clear stack + logs)
    # ------------------------------------------------------------------
    def clear_all(self, delete_logs: bool = True, reset_ids: bool = True) -> None:
        """
        Clear all goals and (optionally) their logs.

        - Empties the in-memory goal list.
        - Resets the active goal.
        - Deletes data/goals.json.
        - Deletes all data/goal_logs/goal_*.jsonl if delete_logs is True.
        """
        self.goals = []
        self.active_goal_id = None

        # Remove goals.json
        try:
            if self._store_path.exists():
                self._store_path.unlink()
        except Exception as e:
            print(f"[GoalManager] Failed to delete goals.json: {e}")

        # Remove per-goal logs
        if delete_logs:
            try:
                for path in self._logs_dir.glob("goal_*.jsonl"):
                    try:
                        path.unlink()
                    except Exception as e:
                        print(f"[GoalManager] Failed to delete log {path}: {e}")
            except Exception as e:
                print(f"[GoalManager] Failed to iterate logs dir: {e}")

        if reset_ids:
            self._next_id = 1

        print("[GoalManager] All goals and logs cleared.")
