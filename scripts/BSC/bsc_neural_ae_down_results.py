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
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, check=True, capture_output=capture)


def list_remote_run_dirs(ssh_transfer: str, remote_output_base: Path) -> list[tuple[int, str]]:
    cmd = (
        f"bash -lc \""
        f"if [ -d '{remote_output_base}' ]; then "
        f"find '{remote_output_base}' -maxdepth 1 -mindepth 1 -type d -name 'run_*' -printf '%f\\n'; "
        f"fi\""
    )
    result = run_cmd(["ssh", ssh_transfer, cmd], capture=True)
    runs: list[tuple[int, str]] = []
    for raw in result.stdout.splitlines():
        name = raw.strip()
        match = re.fullmatch(r"run_(\d+)", name)
        if match:
            runs.append((int(match.group(1)), name))
    runs.sort(key=lambda x: x[0])
    return runs


def main() -> None:
    conf_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONF
    conf = load_conf(conf_path)

    ssh_transfer = conf["SSH_TRANSFER"]
    for i in range(10):
        remote_root = Path("/gpfs/projects/uab103/uab020077/transformer_arch/transformer_arch_" + str(i) + "/transformer_arch")
        min_run = int(conf.get("DOWNLOAD_MIN_RUN", "0"))

        local_output_base = REPO_ROOT / "outputs" / "neural_autoencoder"
        remote_output_base = remote_root / "outputs" / "neural_autoencoder"
        local_output_base.mkdir(parents=True, exist_ok=True)

        remote_runs = list_remote_run_dirs(ssh_transfer, remote_output_base)
        selected_runs = [(idx, name) for idx, name in remote_runs if idx >= min_run]

        if not selected_runs:
            print(f"No remote runs found with index >= {min_run} in {remote_output_base}")
            return

        print(f"Found {len(selected_runs)} remote run(s) with index >= {min_run}")

        for _idx, run_name in selected_runs:
            remote_run_dir = remote_output_base / run_name
            local_run_dir = local_output_base / run_name

            print(f"Downloading {run_name} -> {local_run_dir}")
            local_run_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(["scp", "-r", f"{ssh_transfer}:{remote_run_dir}/.", str(local_run_dir)])
            run_cmd(["ssh", ssh_transfer, f"rm -rf '{remote_run_dir}'"])
            print(f"Downloaded and removed remote: {run_name}")



if __name__ == "__main__":
    main()
