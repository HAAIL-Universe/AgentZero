"""
Persistent Key-Value Store

A file-backed key-value store that survives process restarts.
Challenge C001 solution.

Usage:
    python challenges/C001_kvstore/kvstore.py set key value
    python challenges/C001_kvstore/kvstore.py get key
    python challenges/C001_kvstore/kvstore.py delete key
    python challenges/C001_kvstore/kvstore.py list
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_STORE = Path(__file__).parent / "store.json"


class KVStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_STORE
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
        return {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self, key):
        if key in self._data:
            return self._data[key]
        return None

    def set(self, key, value):
        self._data[key] = value
        self._save()

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def list(self):
        return dict(self._data)

    def __len__(self):
        return len(self._data)


def main():
    parser = argparse.ArgumentParser(description="Persistent Key-Value Store")
    parser.add_argument("command", choices=["get", "set", "delete", "list"])
    parser.add_argument("key", nargs="?")
    parser.add_argument("value", nargs="?")
    parser.add_argument("--store", type=str, help="Path to store file")
    args = parser.parse_args()

    store = KVStore(args.store)

    if args.command == "set":
        if not args.key or args.value is None:
            print("Usage: kvstore.py set <key> <value>")
            sys.exit(1)
        store.set(args.key, args.value)
        print(f"  Set: {args.key} = {args.value}")

    elif args.command == "get":
        if not args.key:
            print("Usage: kvstore.py get <key>")
            sys.exit(1)
        val = store.get(args.key)
        if val is not None:
            print(f"  {args.key} = {val}")
        else:
            print(f"  Key '{args.key}' not found.")
            sys.exit(1)

    elif args.command == "delete":
        if not args.key:
            print("Usage: kvstore.py delete <key>")
            sys.exit(1)
        if store.delete(args.key):
            print(f"  Deleted: {args.key}")
        else:
            print(f"  Key '{args.key}' not found.")
            sys.exit(1)

    elif args.command == "list":
        data = store.list()
        if data:
            print(f"  Store ({len(data)} entries):")
            for k, v in sorted(data.items()):
                print(f"    {k} = {v}")
        else:
            print("  Store is empty.")


if __name__ == "__main__":
    main()
