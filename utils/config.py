# utils/config.py
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

_DEFAULT_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "mvs.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_in_string(value: str, *, strict: bool = True) -> str:
    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            if strict:
                raise KeyError(
                    f"Missing environment variable: {var_name}. "
                    f"Please export it before running (e.g. `export {var_name}=...`)."
                )
            return match.group(0)
        return env_value

    return _ENV_VAR_RE.sub(repl, value)


def _expand_env_vars(obj: Any, *, strict: bool = True) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v, strict=strict) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v, strict=strict) for v in obj]
    if isinstance(obj, str):
        return _expand_env_in_string(obj, strict=strict)
    return obj


def _resolve_path_if_relative(path_like: str) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())


def _resolve_known_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)

    for key in ("datapath", "train_data_list", "val_data_list"):
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = _resolve_path_if_relative(value)

    dinov3_cfg = out.get("dinov3")
    if isinstance(dinov3_cfg, dict):
        d = dict(dinov3_cfg)
        weights = d.get("weights")
        if isinstance(weights, str) and weights:
            d["weights"] = _resolve_path_if_relative(weights)
        out["dinov3"] = d

    return out


def load_cfg(cfg_path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = _DEFAULT_CFG_PATH if cfg_path is None else Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    cfg = _expand_env_vars(raw_cfg, strict=True)
    cfg = _resolve_known_paths(cfg)
    return cfg


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_runs_dir(cfg: Dict[str, Any]) -> Path:
    runs_dir = Path(cfg.get("output", {}).get("runs_dir", "../log"))
    # 让 runs 相对 MyModel3 根目录
    return ensure_dir(_PROJECT_ROOT / runs_dir)
