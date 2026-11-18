"""Track D integration layer; existing Track A modules remain unmodified."""
from pathlib import Path
import sys

# Track A uses sibling imports ("from features import ..."). Keep compatibility.
SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
