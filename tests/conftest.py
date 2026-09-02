"""Bootstrap the src/ layout so tests run without installing the package.

`pytest tests/` from the project root must work with no editable install;
this makes `import vvc` resolve to src/vvc.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
