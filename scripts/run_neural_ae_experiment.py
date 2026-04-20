from __future__ import annotations

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder import (
    build_dataloaders,
    build_model,
    build_model_from_target,
    evaluate_autoencoder,
    infer_batch_shape,
    save_checkpoint,

    train_autoencoder,
)
from v1tovideo.neural_autoencoder.trainer_sc import (
    save_reconstruction_plots,
    save_reconstruction_artifacts,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "neural_ae_experiment.toml"
LOGGER = logging.getLogger(__name__)


def main() -> None:
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
        model_kwargs.setdefault("num_tokens", num_tokens)
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
    latent_dim = int(config.model["latent_dim"]) if "latent_dim" in config.model else None

    LOGGER.info("Training started | epochs=%d | device=%s", config.train.epochs, config.train.device)
    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
    )

    eval_metrics = evaluate_autoencoder(model=model, dataloader=val_loader, device=config.train.device)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "model.pt"
    save_checkpoint(model, checkpoint_path)

    summary: dict[str, Any] = {
        "dataset_shape": dataset.shape,
        "model_name": model_name,
        "model_config": config.model,
        "latent_dim": latent_dim,
        "train_loss": history[-1]["train_loss"] if history else float("nan"),
        "val_loss": history[-1]["val_loss"] if history else float("nan"),
        "val_mse": eval_metrics["mse"],
        "val_mae": eval_metrics["mae"],
    }
    if latent_dim is not None:
        summary["compression_ratio"] = float((token_dim * dataset_num_tokens) / latent_dim)
    else:
        summary["compression_ratio"] = None

    with (output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    shutil.copy(DEFAULT_CONFIG_PATH, output_dir / "config.toml")

    save_reconstruction_artifacts(
        model=model,
        sample_batch=next(iter(val_loader)),
        output_dir=output_dir,
        device=config.train.device,
        prefix="val_sample",
    )
    save_reconstruction_plots(
        model=model,
        output_dir=output_dir,
        dataset=dataset,
        dataset_map=dataset_map,
        val_map_idx=val_map_idx,
        config=config.data,
        device=config.train.device
    )
    LOGGER.info("Saved reconstruction plots")

    LOGGER.info("Run finished | output_dir=%s", output_dir)
    print(summary)


if __name__ == "__main__":
    main()
