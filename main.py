"""Entry point for running the CLI without installing the package.

Prefer `pip install -e .` and the `rag` command; this file exists so a fresh
clone runs with `python main.py ask "..."` before anything is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stacklens.cli import app

if __name__ == "__main__":
    app()
