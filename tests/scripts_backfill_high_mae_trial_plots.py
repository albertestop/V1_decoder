#!/home/albertestop/miniconda3/envs/sci_albert/bin/python
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.neural_autoencoder.config_parser import parse_neural_ae_experiment_config
from v1tovideo.neural_autoencoder.data import build_dataset
from v1tovideo.neural_autoencoder.models import build_model_from_target
from v1tovideo.neural_autoencoder.trainer_sc import save_high_mae_trial_plots

"""
Execute with 
/home/albertestop/miniconda3/envs/sci_albert/bin/python   tests/scripts_backfill_high_mae_trial_plots.py   --root outputs/neural_autoencoder   --start 161 --end 202   --data-root /home/albertestop/data/processed_data/tba   --device cuda:1   2>&1 | tee outputs/neural_autoencoder/backfill_161_202.log
"""

LOGGER = logging.getLogger(__name__)


def run_index(path: Path) -> int | None:
    if not path.name.startswith("run_"):
        return None
    suffix = path.name.removeprefix("run_")
    return int(suffix) if suffix.isdigit() else None


def load_model_config(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
        model_config = summary.get("model_config")
        if isinstance(model_config, dict):
            return model_config
    config = parse_neural_ae_experiment_config(run_dir / "config.toml")
    return config.model


def backfill_run(
    run_dir: Path,
    device: str | None,
    mae_threshold: float,
    data_root: Path | None,
) -> None:
    config = parse_neural_ae_experiment_config(run_dir / "config.toml")
    if data_root is not None:
        if config.data.path is None:
            raise ValueError(f"Run {run_dir.name} does not define a dataset path")
        config.data.path = data_root / config.data.path.name
        LOGGER.info("Using local dataset path %s", config.data.path)
    dataset = build_dataset(config.data)

    with (run_dir / "val_indices.json").open("r", encoding="utf-8") as fp:
        val_indices = [int(i) for i in json.load(fp)]
    val_index_set = set(val_indices)
    train_indices = [i for i in range(len(dataset)) if i not in val_index_set]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    model_config = load_model_config(run_dir)
    model_target = str(model_config["target"])
    model_kwargs = dict(model_config.get("kwargs", {}))
    model_kwargs.setdefault("num_tokens", int(getattr(dataset, "max_tokens")))
    model_kwargs.setdefault("token_dim", int(getattr(dataset, "token_dim")))

    model = build_model_from_target(model_target, kwargs=model_kwargs)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    plot_dir = run_dir / "high_mae_trial_plots"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
        LOGGER.info("Removed existing plots from %s", plot_dir)

    result = save_high_mae_trial_plots(
        model=model,
        val_dataset=val_dataset,
        train_dataset=train_dataset,
        dataset_dir=config.data.path,
        output_dir=run_dir,
        device=device or config.train.device,
        mae_threshold=mae_threshold,
    )
    LOGGER.info(
        "%s wrote %s plots to %s",
        run_dir.name,
        sum(int(v) for v in result["plot_counts"].values()),
        result["output_dir"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Backfill high-MAE trial plots into saved neural AE runs.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "neural_autoencoder")
    parser.add_argument("--start", type=int, default=162)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=Path, default=None, help="Local root containing saved dataset directories")
    parser.add_argument("--mae-threshold", type=float, default=0.3)
    args = parser.parse_args()

    run_dirs = []
    for path in args.root.iterdir():
        idx = run_index(path)
        if idx is None or idx < args.start or (args.end is not None and idx > args.end):
            continue
        run_dirs.append((idx, path))
    run_dirs.sort(reverse=True)

    for idx, run_dir in run_dirs:
        required = ["config.toml", "summary.json", "model.pt", "val_indices.json"]
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            LOGGER.warning("Skipping %s; missing %s", run_dir.name, ", ".join(missing))
            continue
        LOGGER.info("Backfilling %s", run_dir.name)
        backfill_run(run_dir, args.device, args.mae_threshold, args.data_root)


if __name__ == "__main__":
    main()
