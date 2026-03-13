from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.config_utils import load_toml, resolve_repo_path
from v1tovideo.image_autoencoder import evaluate_random_frames

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "image_vae_batch.toml"
LOGGER = logging.getLogger(__name__)


@dataclass
class BatchRunConfig:
    frames_root: Path
    num_samples: int
    output_dir: Path
    seed: int
    height: int
    width: int


def _parse_config(config_path: Path) -> BatchRunConfig:
    data = load_toml(config_path)
    run = data.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Config must define a [run] table: {config_path}")

    try:
        frames_root = resolve_repo_path(run["frames_root"])
    except KeyError as exc:
        raise ValueError("Missing required config key: run.frames_root") from exc

    num_samples = int(run.get("num_samples", 100))
    output_dir = resolve_repo_path(run.get("output_dir", "outputs/image_compression/batch"))
    seed = int(run.get("seed", 0))
    height = int(run.get("height", 144))
    width = int(run.get("width", 256))

    if num_samples <= 0:
        raise ValueError("run.num_samples must be a positive integer")
    if height <= 0 or width <= 0:
        raise ValueError("run.height and run.width must be positive integers")

    return BatchRunConfig(
        frames_root=frames_root,
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
        height=height,
        width=width,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Evaluate SD3 VAE reconstruction on random video frames.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/image_vae_batch.toml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    LOGGER.info("Using config: %s", config_path)

    config = _parse_config(config_path)
    LOGGER.info(
        "Running batch VAE evaluation | frames_root=%s | samples=%d | size=%dx%d",
        config.frames_root,
        config.num_samples,
        config.height,
        config.width,
    )

    summary = evaluate_random_frames(
        frames_root=config.frames_root,
        num_samples=config.num_samples,
        output_dir=config.output_dir,
        seed=config.seed,
        target_height=config.height,
        target_width=config.width,
    )
    LOGGER.info("Completed batch VAE evaluation")
    print(summary)


if __name__ == "__main__":
    main()
