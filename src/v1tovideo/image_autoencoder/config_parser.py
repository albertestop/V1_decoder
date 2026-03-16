from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v1tovideo.config_utils import load_toml, resolve_repo_path


@dataclass
class SingleRunConfig:
    image_path: Path
    output_dir: Path
    height: int
    width: int
    prefix: str


@dataclass
class BatchRunConfig:
    frames_root: Path
    num_samples: int
    output_dir: Path
    seed: int
    height: int
    width: int


def parse_image_vae_single_config(config_path: Path) -> SingleRunConfig:
    data = load_toml(config_path)
    run = data.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Config must define a [run] table: {config_path}")

    try:
        image_path = resolve_repo_path(run["image_path"])
    except KeyError as exc:
        raise ValueError("Missing required config key: run.image_path") from exc

    output_dir = resolve_repo_path(run.get("output_dir", "outputs/image_compression/single"))
    height = int(run.get("height", 144))
    width = int(run.get("width", 256))
    prefix = str(run.get("prefix", "sample"))

    if height <= 0 or width <= 0:
        raise ValueError("run.height and run.width must be positive integers")

    return SingleRunConfig(
        image_path=image_path,
        output_dir=output_dir,
        height=height,
        width=width,
        prefix=prefix,
    )


def parse_image_vae_batch_config(config_path: Path) -> BatchRunConfig:
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
