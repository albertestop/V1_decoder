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


def remote_test(ssh_target: str, test_expr: str) -> bool:
    result = subprocess.run(
        ["ssh", ssh_target, f"bash -lc 'if {test_expr}; then echo YES; else echo NO; fi'"],
        text=True,
        capture_output=True,
        check=True,
    )
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


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def dump_toml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    root_scalars: dict[str, Any] = {}
    sections: list[tuple[str, dict[str, Any]]] = []

    for key, value in data.items():
        if isinstance(value, dict):
            sections.append((key, value))
        else:
            root_scalars[key] = value

    for key, value in root_scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")

    if root_scalars and sections:
        lines.append("")

    for idx, (section_name, section_body) in enumerate(sections):
        lines.append(f"[{section_name}]")
        for key, value in section_body.items():
            if isinstance(value, dict):
                lines.append("")
                lines.append(f"[{section_name}.{key}]")
                for sub_key, sub_val in value.items():
                    lines.append(f"{sub_key} = {_toml_value(sub_val)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        if idx < len(sections) - 1:
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_model_file_from_target(target: str) -> Path | None:
    if ":" not in target:
        return None
    module, _symbol = target.split(":", 1)
    return REPO_ROOT / "src" / Path(*module.split(".")).with_suffix(".py")


def main() -> None:
    conf_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CONF
    conf = load_conf(conf_path)

    ssh_login = conf["SSH_LOGIN"]
    ssh_transfer = conf["SSH_TRANSFER"]
    remote_root = Path(conf["REMOTE_ROOT"])
    remote_data_root = Path(conf["REMOTE_DATA_ROOT"])
    sync_whole_repo = conf.get("SYNC_WHOLE_REPO", "true").lower() in {"1", "true", "yes", "y"}

    local_config = (REPO_ROOT / conf["LOCAL_EXPERIMENT_CONFIG"]).resolve()
    remote_config = remote_root / conf["REMOTE_EXPERIMENT_CONFIG"]
    remote_sbatch = remote_root / conf["REMOTE_SBATCH_SCRIPT"]
    remote_bsc_dir = remote_root / "scripts" / "BSC"

    with local_config.open("rb") as fh:
        experiment_cfg = tomllib.load(fh)

    dataset_id = str(experiment_cfg["data"]["dataset_id"])
    local_data_root = Path(str(experiment_cfg["data"]["data_root_path"]))
    local_dataset_dir = local_data_root / dataset_id
    remote_dataset_dir = remote_data_root / dataset_id

    if not local_dataset_dir.exists():
        raise FileNotFoundError(f"Local dataset not found: {local_dataset_dir}")

    local_output_base = REPO_ROOT / "outputs" / "neural_autoencoder"
    local_run_dir = reserve_next_local_run_dir(local_output_base)
    remote_run_dir = remote_root / "outputs" / "neural_autoencoder" / local_run_dir.name

    print(f"Local run directory: {local_run_dir}")
    print(f"Remote run directory: {remote_run_dir}")

    run_cmd(["ssh", ssh_transfer, f"mkdir -p '{remote_data_root}' '{remote_bsc_dir}'"])

    if sync_whole_repo:
        run_cmd(
            [
                "rsync",
                "-az",
                "--delete",
                "--exclude",
                ".git/",
                "--exclude",
                "outputs/",
                "--exclude",
                "__pycache__/",
                "--exclude",
                "*.pyc",
                f"{REPO_ROOT}/",
                f"{ssh_transfer}:{remote_root}/",
            ]
        )

    dataset_exists_remote = remote_test(ssh_transfer, f"test -d '{remote_dataset_dir}'")
    if not dataset_exists_remote:
        run_cmd(
            [
                "scp",
                "-r",
                str(local_dataset_dir),
                f"{ssh_transfer}:{remote_data_root}",
            ]
        )
    else:
        print(f"Dataset already present on remote: {remote_dataset_dir}")

    architecture = str(experiment_cfg.get("architecture", experiment_cfg["data"].get("architecture", ""))).lower()
    if architecture == "custom":
        target = str(experiment_cfg["custom_model"]["target"])
        local_model_file = parse_model_file_from_target(target)
        if local_model_file is not None and local_model_file.exists():
            remote_model_file = remote_root / local_model_file.relative_to(REPO_ROOT)
            model_exists_remote = remote_test(ssh_transfer, f"test -f '{remote_model_file}'")
            if not model_exists_remote:
                run_cmd(
                    [
                        "ssh",
                        ssh_transfer,
                        f"mkdir -p '{remote_model_file.parent}'",
                    ]
                )
                run_cmd(["scp", str(local_model_file), f"{ssh_transfer}:{remote_model_file}"])
            else:
                print(f"Model file already present on remote: {remote_model_file}")

    remote_cfg = dict(experiment_cfg)
    remote_cfg["data"] = dict(experiment_cfg["data"])
    remote_cfg["output"] = dict(experiment_cfg["output"])
    remote_cfg["train"] = dict(experiment_cfg["train"])
    remote_cfg["data"]["data_root_path"] = str(remote_data_root)
    remote_cfg["output"]["dir"] = str(remote_run_dir)
    train_device = str(remote_cfg["train"].get("device", "cuda"))
    if train_device.startswith("cuda:"):
        remote_cfg["train"]["device"] = "cuda:0"

    local_generated_cfg = Path(__file__).resolve().parent / "generated_remote_config.toml"
    dump_toml(remote_cfg, local_generated_cfg)

    run_cmd(["scp", str(local_generated_cfg), f"{ssh_transfer}:{remote_config}"])
    if not sync_whole_repo:
        run_cmd(["scp", str(Path(__file__).resolve().parent / "train_ae.py"), f"{ssh_transfer}:{remote_bsc_dir / 'train_ae.py'}"])
        run_cmd(["scp", str(Path(__file__).resolve().parent / "run_neural_ae_remote.sh"), f"{ssh_transfer}:{remote_bsc_dir / 'run_neural_ae_remote.sh'}"])
        run_cmd(
            [
                "scp",
                str(Path(__file__).resolve().parent / "requirements_sci_albert_full.txt"),
                f"{ssh_transfer}:{remote_bsc_dir / 'requirements_sci_albert_full.txt'}",
            ]
        )

    run_cmd(["ssh", ssh_transfer, f"chmod +x '{remote_sbatch}' '{remote_bsc_dir / 'train_ae.py'}'"])

    submit = run_cmd(
        [
            "ssh",
            ssh_login,
            f"cd '{remote_bsc_dir}' && sbatch --parsable '{remote_sbatch}'",
        ],
        capture=True,
    )
    job_id = submit.stdout.strip().split(";")[0]
    print(f"Submitted job id: {job_id}")



if __name__ == "__main__":
    main()
