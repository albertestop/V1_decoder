from __future__ import annotations

import json
import logging
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
NEURAL_AE_DIR = SRC_DIR / "v1tovideo" / "neural_autoencoder"
if str(NEURAL_AE_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_AE_DIR))

from synthetic import (
    SyntheticFactorDatasetConfig,
    save_factor_dataset,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "synthetic_neural_dataset.toml"
LOGGER = logging.getLogger(__name__)


@dataclass
class GeneratorRunConfig:
    output_path: Path
    synthetic: SyntheticFactorDatasetConfig


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid TOML structure in {config_path}")
    return data


def _parse_config(config_path: Path) -> GeneratorRunConfig:
    data = _load_toml(config_path)

    run_cfg = data.get("run")
    synth_cfg = data.get("synthetic")

    if not isinstance(run_cfg, dict):
        raise ValueError("Config must define [run]")
    if not isinstance(synth_cfg, dict):
        raise ValueError("Config must define [synthetic]")

    output_path = _resolve_repo_path(run_cfg.get("output_path", "data/neural/synthetic_factors.npy"))

    synthetic = SyntheticFactorDatasetConfig(
        n_samples=int(synth_cfg.get("n_samples", 336)),
        sequence_length=int(synth_cfg.get("sequence_length", 300)),
        n_neurons=int(synth_cfg.get("n_neurons", 5443)),
        n_factors=int(synth_cfg.get("n_factors", 16)),
        factor_scale=float(synth_cfg.get("factor_scale", 1.0)),
        noise_std=float(synth_cfg.get("noise_std", 0.05)),
        baseline_std=float(synth_cfg.get("baseline_std", 0.0)),
        seed=int(synth_cfg.get("seed", 0)),
    )

    return GeneratorRunConfig(output_path=output_path, synthetic=synthetic)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Generate synthetic neural dataset from K factor traces.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/synthetic_neural_dataset.toml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    LOGGER.info("Using config: %s", config_path)
    config = _parse_config(config_path)

    out_path = save_factor_dataset(config.synthetic, config.output_path)
    summary = {
        "output_path": str(out_path),
        "shape": [config.synthetic.n_samples, config.synthetic.sequence_length, config.synthetic.n_neurons],
        "n_factors": config.synthetic.n_factors,
        "seed": config.synthetic.seed,
    }

    summary_path = out_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    LOGGER.info("Synthetic dataset generated: %s", out_path)
    print(summary)


if __name__ == "__main__":
    main()
