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

from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder import (
    build_dataloaders,
    build_model,
    build_model_from_target,
    infer_batch_shape,
    save_reconstruction_plots,
)




print(np.random.randint(0, 1, 1))