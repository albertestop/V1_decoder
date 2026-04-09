import argparse
import json
from pathlib import Path
import os
import pickle
import numpy as np
import sys

from synthetic_data.sc.plot import *

def parse_dataset_rows(value):
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid dataset_rows string: {value}")
        start, end = int(parts[0]), int(parts[1])
        return start, end

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Invalid dataset_rows list: {value}")
        return int(value[0]), int(value[1])

    raise ValueError(f"Unsupported dataset_rows format: {value}")


def load_trial_entries(trial_map_path):
    with open(trial_map_path, "r", encoding="utf-8") as f:
        trial_map = json.load(f)

    trial_entries = []
    for _, info in sorted(trial_map.items(), key=lambda x: int(x[0])):
        rows = info.get("dataset_rows", [])
        if rows == []:
            continue

        start, end = parse_dataset_rows(rows)
        if end <= start:
            continue

        trial_entries.append(
            {
                "trial_index": int(info["trial_index"]),
                "trial_id": str(info["trial_id"]),
                "start": start,
                "end": end,
            }
        )

    if not trial_entries:
        raise ValueError("No valid trial entries found in trial map.")

    return trial_entries[:-1]


def post(output_dir, trial_len, n_trials):
    postpr_path = f'{output_dir}/data/responses.npy'
    postpr_data = np.load(postpr_path, allow_pickle=True).astype(np.float64)

    plot_trials = np.random.randint(0, n_trials, 3)

    print(f"Saving sanity check plots in {output_dir}")
    for i, trial in enumerate(plot_trials):
        trial_data = postpr_data[trial * trial_len:(trial * trial_len) + trial_len]
        plot_random_cell_activity(trial_data, trial, i, output_dir)

    raster_trial = np.random.choice(plot_trials)
    trial_data = postpr_data[raster_trial * trial_len:(raster_trial * trial_len) + trial_len]
    plot_trial_raster(trial_data, raster_trial, output_dir)