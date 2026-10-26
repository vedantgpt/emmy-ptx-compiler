"""Print PTX for tensor-core + TMA nodes the open set never exercises.

    python examples/extra_isa.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_extra_isa import extra_kernel

if __name__ == "__main__":
    sys.stdout.write(extra_kernel())
