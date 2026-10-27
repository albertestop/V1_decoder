#!/usr/bin/env python3
"""Delete detected BSC neural-autoencoder result directories without downloading.

Usage:
    python scripts/BSC/bsc_neural_ae_delete_results.py [path/to/bsc_neural_ae.conf]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    return subprocess.run(cmd, text=True, check=True, capture_output=capture)


def remote_dir_exists(ssh_transfer: str, remote_dir: Path) -> bool:
    cmd = (
        f"bash -lc \""
        f"if [ -d '{remote_dir}' ]; then "
        f"echo YES; "
        f"else "
        f"echo NO; "
        f"fi\""
    )
    result = run_cmd(["ssh", ssh_transfer, cmd], capture=True)
    return "YES" in result.stdout


def main() -> None:
    conf_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONF
    conf = load_conf(conf_path)
    ssh_transfer = conf["SSH_TRANSFER"]

    found = 0
    for i in range(50):
        remote_root = Path(
            "/gpfs/projects/uab108/uab020077/transformer_arch/"
            f"transformer_arch_{i}/transformer_arch"
        )
        remote_run_dir = remote_root / "outputs" / "neural_autoencoder" / "default_run"
        if not remote_dir_exists(ssh_transfer, remote_run_dir):
            continue

        found += 1
        print(f"Removing remote: transformer_arch_{i}/default_run")
        run_cmd(["ssh", ssh_transfer, f"rm -rf '{remote_run_dir}'"])
        print(f"Removed remote: transformer_arch_{i}/default_run")

    if found == 0:
        print("No remote default_run directories found.")


if __name__ == "__main__":
    main()
