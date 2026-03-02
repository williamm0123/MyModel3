# utils/vis_module.py
"""
Visualization module for MyModel3 network outputs.

✅ 适配你当前的 Network.forward() 输出结构：
- 默认只返回: stage1-4
- return_intermediate=True 时额外返回: dino_l3/l7/l11, sva_out, conv31_fused

因此这里做的核心修改是：
1) FPN 可视化：从旧的 fpn_s1/s2/s4/s8 改成 stage1-4（并兼容旧键名）
2) SVA 可视化：支持 sva_out / conv31_fused（并兼容 fused_s8）
3) DINO 可视化：保持 dino_l3/l7/l11（只在 return_intermediate=True 才会出现）
4) visualize_all：自动根据 outputs 里实际存在的 key 来画，不再全是 [SKIP]
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt


def to_numpy(x: torch.Tensor) -> Optional[np.ndarray]:
    """Convert tensor to numpy, handling device and grad."""
    if x is None:
        return None
    return x.detach().cpu().numpy()


def normalize_feature(feat: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize feature map for visualization.
    Args:
        feat: (H, W) or (C, H, W)
        method: 'minmax' or 'std'
    """
    if method == "minmax":
        vmin, vmax = feat.min(), feat.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(feat)
        return (feat - vmin) / (vmax - vmin)
    elif method == "std":
        mean, std = feat.mean(), feat.std()
        if std < 1e-8:
            return np.zeros_like(feat)
        return np.clip((feat - mean) / (3 * std) + 0.5, 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _select_feature_slice(
    feat: np.ndarray,
    batch_idx: int = 0,
    view_idx: int = 0,
) -> np.ndarray:
    """
    Select a (C,H,W) slice from possible shapes:
      - (B, V, C, H, W)
      - (B, C, H, W)
      - (C, H, W)
    """
    if feat.ndim == 5:
        return feat[batch_idx, view_idx]
    if feat.ndim == 4:
        return feat[batch_idx]
    if feat.ndim == 3:
        return feat
    raise ValueError(f"Unsupported feature ndim={feat.ndim}, shape={feat.shape}")


def visualize_feature_map(
    feat: np.ndarray,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Visualize a single feature map with mean, max, std, and channel-0 views.
    Args:
        feat: (C, H, W) feature map
    """
    if feat.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape {feat.shape}")

    C, H, W = feat.shape

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f"{title}  |  shape: ({C}, {H}, {W})", fontsize=12)

    # 1) Mean across channels
    mean_map = feat.mean(axis=0)
    axes[0].imshow(normalize_feature(mean_map), cmap="viridis")
    axes[0].set_title(f"Mean (μ={mean_map.mean():.3f})")
    axes[0].axis("off")

    # 2) Max across channels
    max_map = feat.max(axis=0)
    axes[1].imshow(normalize_feature(max_map), cmap="hot")
    axes[1].set_title(f"Max (max={max_map.max():.3f})")
    axes[1].axis("off")

    # 3) Std across channels
    std_map = feat.std(axis=0)
    axes[2].imshow(normalize_feature(std_map), cmap="plasma")
    axes[2].set_title(f"Std (σ={std_map.mean():.3f})")
    axes[2].axis("off")

    # 4) Channel 0
    ch0 = feat[0]
    axes[3].imshow(normalize_feature(ch0), cmap="gray")
    axes[3].set_title(f"Ch0 (range: {ch0.min():.2f}~{ch0.max():.2f})")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VIS] Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Module-specific visualizers
# ============================================================

def visualize_stage_outputs(
    outputs: Dict[str, torch.Tensor],
    save_dir: Optional[Path] = None,
    show: bool = True,
    batch_idx: int = 0,
    view_idx: int = 0,
) -> None:
    """
    Visualize multi-scale outputs.
    Prefer new keys: stage1-4.
    Compatible with legacy keys: fpn_s8/s4/s2/s1.
    """
    # Preferred (new) keys
    stage_keys = ["stage1", "stage2", "stage3", "stage4"]

    # Legacy fallback mapping
    legacy_map = {
        "stage1": "fpn_s8",
        "stage2": "fpn_s4",
        "stage3": "fpn_s2",
        "stage4": "fpn_s1",
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Multi-Scale Outputs Visualization (stage1-4)")
    print("=" * 60)

    for k in stage_keys:
        key_to_use = k if (k in outputs and outputs[k] is not None) else legacy_map.get(k)
        if not key_to_use or key_to_use not in outputs or outputs[key_to_use] is None:
            print(f"[SKIP] {k}: not found")
            continue

        feat = to_numpy(outputs[key_to_use])
        feat = _select_feature_slice(feat, batch_idx=batch_idx, view_idx=view_idx)

        save_path = (Path(save_dir) / f"{k}.png") if save_dir else None
        visualize_feature_map(feat, title=f"{k} (view={view_idx})", save_path=save_path, show=show)


def visualize_dino_outputs(
    outputs: Dict[str, torch.Tensor],
    save_dir: Optional[Path] = None,
    show: bool = True,
    batch_idx: int = 0,
    view_idx: int = 0,
) -> None:
    """Visualize DINO layer outputs (only available when return_intermediate=True)."""
    dino_keys = ["dino_l3", "dino_l7", "dino_l11"]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DINO Outputs Visualization")
    print("=" * 60)

    for key in dino_keys:
        if key not in outputs or outputs[key] is None:
            print(f"[SKIP] {key}: not found")
            continue

        feat = to_numpy(outputs[key])
        feat = _select_feature_slice(feat, batch_idx=batch_idx, view_idx=view_idx)

        save_path = (Path(save_dir) / f"{key}.png") if save_dir else None
        visualize_feature_map(feat, title=f"{key} (view={view_idx})", save_path=save_path, show=show)


def visualize_sva_outputs(
    outputs: Dict[str, torch.Tensor],
    save_dir: Optional[Path] = None,
    show: bool = True,
    batch_idx: int = 0,
    view_idx: int = 0,
) -> None:
    """
    Visualize SVA and fusion outputs.
    Current Network(return_intermediate=True) provides:
      - sva_out: (B, V, C, H, W)
      - conv31_fused: (B, V, C, H, W)
    Keep compatibility with:
      - fused_s8 (legacy)
    """
    # Prefer new keys
    sva_keys = ["sva_out", "conv31_fused", "fused_s8"]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SVA / Fusion Outputs Visualization")
    print("=" * 60)

    for key in sva_keys:
        if key not in outputs or outputs[key] is None:
            print(f"[SKIP] {key}: not found or None")
            continue

        feat = to_numpy(outputs[key])
        feat = _select_feature_slice(feat, batch_idx=batch_idx, view_idx=view_idx)

        save_path = (Path(save_dir) / f"{key}.png") if save_dir else None
        visualize_feature_map(feat, title=f"{key} (view={view_idx})", save_path=save_path, show=show)


# ============================================================
# Orchestrator
# ============================================================

def visualize_all(
    outputs: Dict[str, torch.Tensor],
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    batch_idx: int = 0,
    view_idx: int = 0,
) -> None:
    """
    Visualize all outputs:
      - stage1-4 (always)
      - DINO outputs (if present)
      - SVA/Fusion outputs (if present)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("Network Output Summary")
    print("=" * 60)
    for k, v in outputs.items():
        if v is None:
            print(f"  {k}: None")
        else:
            print(f"  {k}: {tuple(v.shape)}, dtype={v.dtype}, device={v.device}")

    # Visualize stage outputs
    stage_dir = (save_dir / "stage") if save_dir else None
    visualize_stage_outputs(outputs, stage_dir, show, batch_idx, view_idx)

    # Visualize DINO outputs (only if present)
    dino_dir = (save_dir / "dino") if save_dir else None
    visualize_dino_outputs(outputs, dino_dir, show, batch_idx, view_idx)

    # Visualize SVA outputs (only if present)
    sva_dir = (save_dir / "sva") if save_dir else None
    visualize_sva_outputs(outputs, sva_dir, show, batch_idx, view_idx)

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    if save_dir:
        print(f"Figures saved to: {save_dir}")
    print("=" * 60)


def print_output_stats(outputs: Dict[str, torch.Tensor]) -> None:
    """Print statistics for all outputs (no visualization)."""
    print("\n" + "=" * 70)
    print(f"{'Key':<20} {'Shape':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("=" * 70)

    for k, v in sorted(outputs.items()):
        if v is None:
            print(f"{k:<20} {'None':<25}")
            continue

        arr = to_numpy(v)
        shape_str = str(tuple(v.shape))
        print(f"{k:<20} {shape_str:<25} {arr.mean():>10.4f} {arr.std():>10.4f} {arr.min():>10.4f} {arr.max():>10.4f}")

    print("=" * 70)
