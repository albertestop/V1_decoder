#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF = Path(__file__).resolve().parent / "bsc_neural_ae.conf"


def load_conf(path: Path) -> dict[str, str]:
    conf: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        conf[key.strip()] = value.strip()
    return conf


def run_cmd(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, check=True, capture_output=capture)


def reserve_next_local_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    while True:
        max_idx = -1
        for child in base_dir.iterdir():
            match = re.fullmatch(r"run_(\d+)", child.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        candidate = base_dir / f"run_{max_idx + 1}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue


def main() -> None:
    conf_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONF
    conf = load_conf(conf_path)

    ssh_transfer = conf["SSH_TRANSFER"]
    remote_root = Path(conf["REMOTE_ROOT"])

    local_config = (REPO_ROOT / conf["LOCAL_EXPERIMENT_CONFIG"]).resolve()

    with local_config.open("rb") as fh:
        experiment_cfg = tomllib.load(fh)

    dataset_id = str(experiment_cfg["data"]["dataset_id"])
    local_data_root = Path(str(experiment_cfg["data"]["data_root_path"]))
    local_dataset_dir = local_data_root / dataset_id

    if not local_dataset_dir.exists():
        raise FileNotFoundError(f"Local dataset not found: {local_dataset_dir}")

    local_output_base = REPO_ROOT / "outputs" / "neural_autoencoder"
    local_run_dir = reserve_next_local_run_dir(local_output_base)
    remote_run_dir = remote_root / "outputs" / "neural_autoencoder" / local_run_dir.name

    print(f"Local run directory: {local_run_dir}")
    print(f"Remote run directory: {remote_run_dir}")

    run_cmd(["scp", "-r", f"{ssh_transfer}:{remote_run_dir}/.", str(local_run_dir)])
    run_cmd(["ssh", ssh_transfer, f"rm -rf '{remote_run_dir}'"])

    print(f"Results downloaded to: {local_run_dir}")
    print(f"Remote output removed: {remote_run_dir}")



if __name__ == "__main__":
    main()
