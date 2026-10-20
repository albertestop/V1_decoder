from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

dataset_map = {
  "0": {
    "trial_id": "00438/",
    "trial_index": 0,
    "dataset_rows": "0,5"
  },
  "1": {
    "trial_id": "00414/",
    "trial_index": 1,
    "dataset_rows": "5,10"
  },
  "2": {
    "trial_id": "00414/",
    "trial_index": 2,
    "dataset_rows": "10,15"
  }
}

val_rows = [0,1,2,3,4,10,11,12,13,14]
val_set = set(val_rows)
val_mask = np.array([
    set(map(int, info["dataset_rows"].split(",")[0])).issubset(val_set)
    for info in dataset_map.values()
])
train_mask = ~val_mask
print(train_mask, val_mask)