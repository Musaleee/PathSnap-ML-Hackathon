"""Configuration loading."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = str((REPO_ROOT / cfg["data_root"]).resolve())
    return cfg


def partition_dir(cfg, partition):
    return Path(cfg["data_root"]) / cfg["partitions"][partition]


def list_trips(cfg, partition):
    return sorted(p.name for p in partition_dir(cfg, partition).iterdir() if p.is_dir())
