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





if __name__ == '__main__':

    DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "neural_ae_experiment.toml"
    LOGGER = logging.getLogger(__name__)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Train/evaluate a configurable neural autoencoder skeleton.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/neural_ae_experiment.toml).",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    LOGGER.info("Using config: %s", config_path)
    config = parse_neural_ae_experiment_config(config_path)

    LOGGER.info("Preparing dataset")
    train_loader, val_loader, dataset, dataset_map, val_map_idx = build_dataloaders(config.data)
    LOGGER.info("Dataset loaded | samples=%d | shape=%s", len(dataset), getattr(dataset, "shape", None))
    train_example = next(iter(train_loader))
    num_tokens, token_dim = infer_batch_shape(train_example)
    LOGGER.info("Inferred model input from training batch | token_dim=%d | padded_num_tokens=%d", token_dim, num_tokens)
    dataset_num_tokens = int(getattr(dataset, "shape", (0, num_tokens, token_dim))[1])
    LOGGER.info("Loading %s model", config.model["architecture"])
    if str(config.model["architecture"]).lower() == "custom":
        model_target = str(config.model["target"])
        model_kwargs = dict(config.model.get("kwargs", {}))
        model_kwargs.setdefault("token_dim", token_dim)
        config.model["kwargs"] = model_kwargs
        model = build_model_from_target(model_target, kwargs=model_kwargs)
        model_name = model_target
    else:
        model_config = dict(config.model)
        model_config["token_dim"] = token_dim
        model_config["num_tokens"] = dataset_num_tokens
        config.model = model_config
        model = build_model(model_config)
        model_name = str(config.model["architecture"])
    LOGGER.info("Model initialized: %s", model_name)
    model.load_state_dict(torch.load("/home/albertestop/visual_cortex_study/transformer_arch/outputs/neural_autoencoder/default_run/model.pt"))
    output_dir = config.output_dir
    save_reconstruction_plots(
        model=model,
        output_dir=output_dir,
        dataset=dataset,
        dataset_map=dataset_map,
        val_map_idx=val_map_idx,
        config=config.data,
        device=config.train.device
    )