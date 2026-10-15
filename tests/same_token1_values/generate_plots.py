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


def main() -> None:
    trial_map = json.loads((DATASET_DIR / "trial_dataset_map.json").read_text())
    start, end = map(int, trial_map[str(TRIAL_ID)]["dataset_rows"].split(","))
    targets = np.load(DATASET_DIR / "data" / "targets.npy", allow_pickle=True)

    values = {neuron_id: [] for neuron_id in NEURON_IDS}
    for tokens in targets[start:end]:
        for neuron_id in NEURON_IDS:
            row = np.flatnonzero(tokens[:, 0].astype(int) == neuron_id)
            if len(row) != 1:
                raise ValueError(f"Expected one token for neuron {neuron_id}, found {len(row)}")
            values[neuron_id].append(float(tokens[row[0], TOKEN_INDEX]))

    cycles = np.arange(end - start)
    for neuron_id in NEURON_IDS:
        plt.figure(figsize=(10, 4))
        plt.plot(cycles, values[neuron_id], color="black", linewidth=2)
        plt.xlabel("Cycle within trial")
        plt.ylabel("Token 1 value (time)")
        plt.title(f"Trial {TRIAL_ID}, neuron {neuron_id}: target token 1")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"trial_{TRIAL_ID:03d}_neuron_{neuron_id}_token_1.png", dpi=150)
        plt.close()

    lines = ["cycle\tneuron_467_token_1\tneuron_487_token_1"]
    lines.extend(f"{cycle}\t{values[467][cycle]:.9g}\t{values[487][cycle]:.9g}" for cycle in cycles)
    (OUTPUT_DIR / "trial_066_token_1_values.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
