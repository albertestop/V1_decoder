#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    bsc_dir = Path(__file__).resolve().parent
    repo_root = bsc_dir.parents[1]
    config_path = bsc_dir / "generated_remote_config.toml"
    entrypoint = bsc_dir / "run_neural_ae_experiment.py"

    cmd = [sys.executable, str(entrypoint), "--config", str(config_path)]
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
