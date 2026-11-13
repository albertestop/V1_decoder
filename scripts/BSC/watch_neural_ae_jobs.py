#!/usr/bin/env python3
"""Locally watch BSC neural-AE jobs and recover time-wall interruptions.

Run this script from tmux.  It does not touch active/pending Slurm jobs.  An
output is complete only once summary.json exists; a stopped run is resumable
when config.toml, model.pt, and val_indices.json are present.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF = Path(__file__).resolve().parent / "bsc_neural_ae.conf"
RUN_NAME = "default_run"


def load_conf(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def command(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(cmd, text=True, check=True, capture_output=capture)


def ssh(target: str, remote: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return command(["ssh", target, remote], capture=capture)


def active_jobs(ssh_login: str, user: str) -> str:
    # scontrol includes WorkDir, which identifies each transformer_arch_N job.
    remote = f"for id in $(squeue -h -u {shlex.quote(user)} -o '%i'); do scontrol show job -o \"$id\"; done"
    return ssh(ssh_login, remote, capture=True).stdout


def remote_files(ssh_transfer: str, run_dir: Path) -> set[str]:
    names = "config.toml model.pt val_indices.json summary.json"
    remote = f"for x in {names}; do test -f {shlex.quote(str(run_dir))}/$x && echo $x; done; true"
    return set(ssh(ssh_transfer, remote, capture=True).stdout.split())


def original_epochs(ssh_transfer: str, run_dir: Path) -> int:
    # TOML is read remotely so each workspace uses its own original epoch count.
    remote = "python -c " + shlex.quote(
        "import sys\ntry:\n import tomllib\nexcept ModuleNotFoundError:\n import tomli as tomllib\nprint(tomllib.load(open(sys.argv[1], 'rb'))['train']['epochs'])"
    ) + " " + shlex.quote(str(run_dir / "config.toml"))
    result = int(ssh(ssh_transfer, remote, capture=True).stdout.strip())
    if result <= 0:
        raise ValueError(f"Invalid epochs in {run_dir}")
    return result


def sync_and_submit(ssh_login: str, ssh_transfer: str, workspace: Path, run_dir: Path, epochs: int, dry_run: bool) -> str | None:
    if dry_run:
        print(f"DRY RUN: would resume {run_dir} for {epochs} epochs")
        return None
    for local, remote in (
        (REPO_ROOT / "scripts/continue_neural_ae_training.py", workspace / "scripts/continue_neural_ae_training.py"),
        (REPO_ROOT / "scripts/BSC/continue_neural_ae_remote.sh", workspace / "scripts/BSC/continue_neural_ae_remote.sh"),
        (REPO_ROOT / "src/v1tovideo/neural_autoencoder/trainer.py", workspace / "src/v1tovideo/neural_autoencoder/trainer.py"),
    ):
        command(["scp", str(local), f"{ssh_transfer}:{remote}"])
    sbatch = workspace / "scripts/BSC/continue_neural_ae_remote.sh"
    exports = f"ALL,NEURAL_AE_RUN_DIR={run_dir},NEURAL_AE_ADDITIONAL_EPOCHS={epochs}"
    remote = (
        f"cd {shlex.quote(str(workspace))} && "
        f"sbatch --parsable --chdir={shlex.quote(str(workspace))} "
        f"--export={shlex.quote(exports)} {shlex.quote(str(sbatch))}"
    )
    job_id = ssh(ssh_login, remote, capture=True).stdout.strip().split(";", 1)[0]
    print(f"Submitted continuation: {workspace.name}, job {job_id}, epochs {epochs}")
    return job_id


def reserve_local_run() -> Path:
    base = REPO_ROOT / "outputs/neural_autoencoder"
    base.mkdir(parents=True, exist_ok=True)
    indices = [int(m.group(1)) for p in base.iterdir() if (m := re.fullmatch(r"run_(\d+)", p.name))]
    output = base / f"run_{max(indices, default=-1) + 1}"
    output.mkdir()
    return output


def download(ssh_transfer: str, workspace: Path, run_dir: Path, dry_run: bool) -> str | None:
    output = reserve_local_run()
    if dry_run:
        output.rmdir()
        print(f"DRY RUN: would download {run_dir} -> {output}")
        return None
    command(["scp", "-r", f"{ssh_transfer}:{run_dir}/.", str(output)])
    (output / "download_source.txt").write_text(f"{workspace}\n", encoding="utf-8")
    print(f"Downloaded {workspace.name}/{RUN_NAME} -> {output}")
    return str(output)


def read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"workspaces": {}}


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def check(args: argparse.Namespace, conf: dict[str, str], state: dict[str, Any]) -> None:
    login, transfer = conf["SSH_LOGIN"], conf["SSH_TRANSFER"]
    details = active_jobs(login, login.split("@", 1)[0])
    base = Path(conf["REMOTE_ROOT"]).parent.parent
    seen = state.setdefault("workspaces", {})
    for index in range(args.first_workspace, args.last_workspace + 1):
        workspace = base / f"transformer_arch_{index}" / "transformer_arch"
        run_dir = workspace / "outputs/neural_autoencoder" / RUN_NAME
        key = str(workspace)
        if key in details:
            print(f"[{index:02d}] active")
            continue
        files = remote_files(transfer, run_dir)
        if "summary.json" in files:
            if seen.get(key, {}).get("downloaded"):
                continue
            path = download(transfer, workspace, run_dir, args.dry_run)
            if not args.dry_run:
                seen[key] = {"downloaded": path, "at": datetime.now(timezone.utc).isoformat()}
                write_state(args.state_file, state)
        elif {"config.toml", "model.pt", "val_indices.json"} <= files:
            epochs = original_epochs(transfer, run_dir)
            job_id = sync_and_submit(login, transfer, workspace, run_dir, epochs, args.dry_run)
            if not args.dry_run:
                seen[key] = {"last_job": job_id, "epochs": epochs, "at": datetime.now(timezone.utc).isoformat()}
                write_state(args.state_file, state)
        elif files:
            print(f"[{index:02d}] inactive but not resumable: {', '.join(sorted(files))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--first-workspace", type=int, default=0)
    parser.add_argument("--last-workspace", type=int, default=34)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-file", type=Path, default=REPO_ROOT / "outputs/neural_autoencoder/bsc_watchdog_state.json")
    args = parser.parse_args()
    if args.first_workspace < 0 or args.last_workspace < args.first_workspace or args.interval <= 0:
        parser.error("invalid workspace range or interval")
    args.conf, args.state_file = args.conf.resolve(), args.state_file.resolve()
    conf, state = load_conf(args.conf), read_state(args.state_file)
    while True:
        print(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] checking BSC")
        check(args, conf, state)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
