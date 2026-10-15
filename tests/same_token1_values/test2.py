"""Generate token-1 plots and exact source values for trial 066."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_DIR = Path(
    "/home/albertestop/data/processed_data/tba/"
    "2025-07-04_04_ESPM154_vol_000_subset_1024_target_random_order"
)
OUTPUT_DIR = Path(__file__).parent
TRIAL_ID = 66
NEURON_IDS = (467, 487)
TOKEN_INDEX = 1


trial_map = json.loads((DATASET_DIR / "trial_dataset_map.json").read_text())
start, end = map(int, trial_map[str(TRIAL_ID)]["dataset_rows"].split(","))
targets = np.load(DATASET_DIR / "data" / "targets.npy", allow_pickle=True)
result1 = targets[6204, targets[6204, :, 0] == 467, 1]
result2 = targets[6204, targets[6204, :, 0] == 487, 1]
print(result1, result2)