#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


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

    local_output_base = REPO_ROOT / "outputs" / "neural_autoencoder"
    local_output_base.mkdir(parents=True, exist_ok=True)

    found = 0
    for i in range(50):
        remote_root = Path("/gpfs/projects/uab103/uab020077/transformer_arch/transformer_arch_" + str(i) + "/transformer_arch")
        remote_output_base = remote_root / "outputs" / "neural_autoencoder"
        remote_run_dir = remote_output_base / "default_run"
        if not remote_dir_exists(ssh_transfer, remote_run_dir):
            continue

        found += 1
        local_run_dir = reserve_next_local_run_dir(local_output_base)

        print(f"Downloading transformer_arch_{i}/default_run -> {local_run_dir}")
        run_cmd(["scp", "-r", f"{ssh_transfer}:{remote_run_dir}/.", str(local_run_dir)])
        run_cmd(["ssh", ssh_transfer, f"rm -rf '{remote_run_dir}'"])
        print(f"Downloaded and removed remote: transformer_arch_{i}/default_run")
    if found == 0:
        print("No remote default_run directories found.")



if __name__ == "__main__":
    main()
