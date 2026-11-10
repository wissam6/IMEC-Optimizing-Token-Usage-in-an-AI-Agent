import sys
from pathlib import Path

# The modules in src/ import each other by plain name (e.g. `from config import CONFIGS`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
