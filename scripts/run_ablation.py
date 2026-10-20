"""Run the synthetic-data ablation: python scripts/run_ablation.py [--seeds 0 1 2]"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from seizure_gnn.ablation import main  # noqa: E402

if __name__ == "__main__":
    main()
