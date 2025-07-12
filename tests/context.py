"""
Context to resolve imports of local modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

foo = "  "
