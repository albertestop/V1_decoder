from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.image_autoencoder import evaluate_random_frames
from v1tovideo.image_autoencoder.config_parser import parse_image_vae_batch_config

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "image_vae_batch.toml"
LOGGER = logging.getLogger(__name__)


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

    config = parse_image_vae_batch_config(config_path)
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
