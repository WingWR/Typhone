from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from train.train_pinn import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
