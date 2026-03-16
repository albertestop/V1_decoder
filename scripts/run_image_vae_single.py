from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.image_autoencoder import encode_decode_image, load_sd3_vae
from v1tovideo.image_autoencoder.config_parser import parse_image_vae_single_config

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "image_vae_single.toml"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Run SD3 VAE compression on a single image.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/image_vae_single.toml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    LOGGER.info("Using config: %s", config_path)

    config = parse_image_vae_single_config(config_path)
    LOGGER.info(
        "Running single-image VAE | image=%s | size=%dx%d",
        config.image_path,
        config.height,
        config.width,
    )

    LOGGER.info("Loading SD3 VAE model")
    vae = load_sd3_vae()
    result = encode_decode_image(
        image_path=config.image_path,
        output_dir=config.output_dir,
        vae=vae,
        target_height=config.height,
        target_width=config.width,
        save_prefix=config.prefix,
    )

    summary = {k: v for k, v in result.items() if isinstance(v, float)}
    LOGGER.info("Completed single-image VAE run")
    print(summary)


if __name__ == "__main__":
    main()
