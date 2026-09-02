"""Shared Holodex helpers: gitignored `.env` loading and API key lookup.

Moved out of ``__main__.py`` so both the CLI and library modules
(``roster``) reuse one implementation. The key lives in the gitignored
``.env`` as ``HOLODEX_API_KEY`` and is never printed or committed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from the repo-root ``.env`` (idempotent:
    existing environment variables win via setdefault)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip())


def holodex_key() -> str:
    """Return the Holodex API key, or exit(2) with a clear message."""
    load_dotenv()
    key = os.environ.get("HOLODEX_API_KEY")
    if not key:
        print("HOLODEX_API_KEY is not set", file=sys.stderr)
        sys.exit(2)
    return key
