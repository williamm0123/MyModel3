# utils/config.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

_DEFAULT_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "mvs.json"


def load_cfg() -> Dict[str, Any]:
    cfg_path = _DEFAULT_CFG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_runs_dir(cfg: Dict[str, Any]) -> Path:
    runs_dir = Path(cfg.get("output", {}).get("runs_dir", "runs"))
    # 让 runs 相对 MyModel3 根目录
    root = Path(__file__).resolve().parents[1]
    return ensure_dir(root / runs_dir)
