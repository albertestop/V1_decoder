from typing import Any
from pathlib import Path
from dataclasses import asdict, is_dataclass

from pytorch_lightning.loggers import WandbLogger





def wandb_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return wandb_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): wandb_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wandb_json_safe(v) for v in value]
    return value


def build_wandb_logger(config: Any, config_path: Path) -> WandbLogger:
    wandb_config = {
        "config_path": str(config_path),
        "data": wandb_json_safe(config.data),
        "model": wandb_json_safe(config.model),
        "train": wandb_json_safe(config.train),
        "output_dir": str(config.output_dir),
    }
    return WandbLogger(
        project="visual-cortex-neural-ae",
        name=config.output_dir.name,
        save_dir=str(config.output_dir),
        config=wandb_config,
        log_model=False,
    )


def save_wandb_outputs(wandb_logger: WandbLogger, output_dir: Path) -> None:
    for pattern in ("history.json", "summary.json", "model.pt", "val_sample.*.pt", "*.png"):
        for path in output_dir.glob(pattern):
            wandb_logger.experiment.save(str(path), policy="now")

