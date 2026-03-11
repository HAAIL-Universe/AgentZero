# ethical_compass.py

import os
from typing import Optional
from pathlib import Path
from datetime import datetime

MANIFESTO_PATH = Path("MagistusManifesto.md")
CACHE = None


def get_ethics(as_markdown: bool = False) -> str:
    """
    Returns the contents of the immutable Magistus Manifesto.
    Can return as plain string or raw markdown.
    """
    global CACHE

    if CACHE:
        return CACHE if as_markdown else _strip_markdown(CACHE)

    if not MANIFESTO_PATH.exists():
        raise FileNotFoundError(f"Manifesto not found at {MANIFESTO_PATH}")

    with open(MANIFESTO_PATH, "r", encoding="utf-8") as f:
        CACHE = f.read()

    return CACHE if as_markdown else _strip_markdown(CACHE)


def _strip_markdown(md: str) -> str:
    """
    Removes most markdown formatting for cleaner agent input use.
    """
    import re
    text = re.sub(r"#+ ", "", md)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"---+", "", text)
    return text.strip()


def get_manifesto_metadata() -> dict:
    return {
        "loaded": datetime.utcnow().isoformat() + "Z",
        "source": str(MANIFESTO_PATH.resolve()),
        "immutable": True,
        "character": "Magistus Ethical Constitution v1.0"
    }
