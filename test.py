#!/usr/bin/env python3
# test.py
"""
Test script for MyModel3 depth pipeline.
Run: python test.py

This script:
1. Loads config from config/mvs.json
2. Loads one batch from DTU dataset
3. Runs full depth estimation (Network + DepthEstimator)
4. Saves depth maps for stage1-4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Setup paths (ensure imports work from project root)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_runs_dir
from data.dtu_data import DTUData
from models.network.network import Network
from models.network.Depth_estimator import DepthEstimator, DepthEstimatorCfg
from models.losses import MultiStageLoss, LossCfg, create_multiscale_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MyModel3 with optional trained checkpoint")
    parser.add_argument("--config", type=str, default="config/mvs.json", help="Path to config json")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to best/latest checkpoint (.pth)")
    parser.add_argument("--use_ckpt_cfg", action="store_true", help="Use cfg stored in checkpoint if available")
    parser.add_argument("--data_root", type=str, default="", help="Override cfg['datapath']")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def load_cfg_from_path(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inference_checkpoint(
    ckpt_path: Path,
    network: Network,
    depth_estimator: DepthEstimator,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location=device)

    net_state: Optional[Dict[str, torch.Tensor]] = payload.get("network_state_dict")
    depth_state: Optional[Dict[str, torch.Tensor]] = payload.get("depth_estimator_state_dict")

    # Backward compatibility: old checkpoints only stored MVSModel.state_dict().
    if (net_state is None or depth_state is None) and "model_state_dict" in payload:
        full_state = payload["model_state_dict"]
        if net_state is None:
            net_state = {
                k[len("network."):]: v for k, v in full_state.items() if k.startswith("network.")
            }
        if depth_state is None:
            depth_state = {
                k[len("depth_estimator."):]: v
                for k, v in full_state.items()
                if k.startswith("depth_estimator.")
            }

    if not net_state:
        raise KeyError(f"No network weights found in checkpoint: {ckpt_path}")
    if not depth_state:
        raise KeyError(f"No depth_estimator weights found in checkpoint: {ckpt_path}")

    net_msg = network.load_state_dict(net_state, strict=False)
    depth_msg = depth_estimator.load_state_dict(depth_state, strict=False)

    print(f"  Checkpoint loaded: {ckpt_path}")
    print(f"    epoch={payload.get('epoch')} global_step={payload.get('global_step')} best_metric={payload.get('best_metric')}")
    print(f"    network: missing={len(net_msg.missing_keys)} unexpected={len(net_msg.unexpected_keys)}")
    print(f"    depth_estimator: missing={len(depth_msg.missing_keys)} unexpected={len(depth_msg.unexpected_keys)}")
    return payload


def save_depth_map(
    depth_hw: torch.Tensor,
    save_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    """Save a single depth map tensor (H,W) as image with fixed vmin/vmax."""
    depth_np = depth_hw.detach().cpu().numpy().astype(np.float32)
    valid = depth_np > 0

    if vmax <= vmin:
        raise ValueError(f"Invalid vmin/vmax for depth visualization: vmin={vmin}, vmax={vmax}")

    depth_clip = np.clip(depth_np, vmin, vmax)
    depth_vis = (depth_clip - vmin) / (vmax - vmin)
    depth_vis[~valid] = 0.0

    plt.figure(figsize=(8, 6))
    plt.imshow(depth_vis, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title}  vmin={vmin:.3f} vmax={vmax:.3f}")
    plt.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_rgb_image(image_chw: torch.Tensor, save_path: Path, title: str) -> None:
    """Save a single RGB tensor (3,H,W) to image file."""
    image_np = image_chw.detach().cpu().numpy().astype(np.float32)
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = np.clip(image_np, 0.0, 1.0)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    print("="*60)
    print("MyModel3 Test Script")
    print("="*60)
    
    # --------------------------------------------------------
    # 1. Load config
    # --------------------------------------------------------
    print("\n[1/4] Loading config...")
    cfg = load_cfg_from_path(args.config)

    ckpt_payload: Optional[Dict[str, Any]] = None
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_payload = torch.load(ckpt_path, map_location="cpu")
        if args.use_ckpt_cfg and isinstance(ckpt_payload.get("cfg"), dict):
            cfg = ckpt_payload["cfg"]
            print(f"  using cfg from checkpoint: {ckpt_path}")

    if args.data_root:
        cfg["datapath"] = args.data_root

    print(f"  datapath: {cfg.get('datapath')}")
    print(f"  train_data_list: {cfg.get('train_data_list')}")
    
    dataset_cfg = cfg.get("dataset", {})
    print(f"  rectified_dir: {dataset_cfg.get('rectified_dir')}")
    print(f"  views(top-level): {cfg.get('views')}")
    print(f"  light: {dataset_cfg.get('light')}")
    
    # --------------------------------------------------------
    # 2. Load dataset and get one sample
    # --------------------------------------------------------
    print("\n[2/4] Loading dataset...")
    dataset = DTUData(cfg, split=args.split, sample_mode="mvs")
    print(f"  Number of scans: {len(dataset)}")
    print(f"  Scans: {dataset.scans[:5]}{'...' if len(dataset.scans) > 5 else ''}")
    
    # Get first sample
    sample_idx = max(0, min(args.sample_idx, len(dataset) - 1))
    sample = dataset[sample_idx]
    images = sample["images"]  # (V, 3, H, W)
    proj_matrices = sample["proj_matrices"]  # Dict[stageX] -> (V, 2, 4, 4)
    depth_range = sample["depth_range"]  # (2,)
    depth_gt = sample["depth_gt"]  # (H, W)
    mask = sample["mask"]  # (H, W)
    depth_interval = sample["depth_interval"]  # ()
    meta = sample["meta"]
    
    print(f"\n  Sample meta: {meta}")
    print(f"  sample_idx: {sample_idx}")
    print(f"  images shape: {tuple(images.shape)}")
    print(f"  depth_range: {tuple(depth_range.tolist())}")
    print(f"  views used by sample: {meta.get('views')}")
    print(f"  ref_view: {meta.get('ref_view')}")
    
    # Add batch dimension
    images = images.unsqueeze(0)
    full_h, full_w = images.shape[-2], images.shape[-1]
    depth_range = depth_range.unsqueeze(0)
    depth_gt = depth_gt.unsqueeze(0)
    mask = mask.unsqueeze(0)
    depth_interval = depth_interval.unsqueeze(0)
    proj_matrices = {k: v.unsqueeze(0) for k, v in proj_matrices.items()}
    print(f"  images shape (with batch): {tuple(images.shape)}")
    
    # --------------------------------------------------------
    # 3. Build model and forward
    # --------------------------------------------------------
    print("\n[3/4] Building model and running forward pass...")
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    network = Network(cfg, device=device).to(device).eval()
    depth_cfg = cfg.get("depth", {})
    depth_estimator = DepthEstimator(DepthEstimatorCfg(
        ndepths=tuple(depth_cfg.get("ndepths", [32, 16, 8, 4])),
        base_chs=tuple(depth_cfg.get("base_chs", [64, 32, 16, 8])),
        depth_interval_ratios=tuple(depth_cfg.get("depth_interval_ratios", [4.0, 2.0, 1.0, 0.5])),
        inverse_depth=depth_cfg.get("inverse_depth", True),
        depth_type=depth_cfg.get("depth_type", "regression"),
        temperatures=tuple(depth_cfg.get("temperatures", [5.0, 5.0, 5.0, 1.0])),
    )).to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters()) + sum(p.numel() for p in depth_estimator.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad) + \
                      sum(p.numel() for p in depth_estimator.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    if ckpt_path is not None:
        if ckpt_payload is None:
            ckpt_payload = {}
        # reload on target device and apply to model modules.
        load_inference_checkpoint(ckpt_path, network, depth_estimator, device)
    
    # Forward pass
    images = images.to(device)
    depth_range = depth_range.to(device)
    depth_gt = depth_gt.to(device)
    mask = mask.to(device)
    depth_interval = depth_interval.to(device)
    proj_matrices = {k: v.to(device) for k, v in proj_matrices.items()}
    
    with torch.no_grad():
        features = network(images)
        outputs = depth_estimator(features, proj_matrices, depth_range)
    
    print(f"\n  Forward pass complete!")
    print(f"  Output keys: {list(outputs.keys())}")

    # Loss sanity check
    loss_cfg = cfg.get("loss", {})
    depth_cfg = cfg.get("depth", {})
    loss_fn = MultiStageLoss(LossCfg(
        depth_types=tuple(loss_cfg.get("depth_types", ["reg", "reg", "reg", "reg"])),
        loss_weights=tuple(loss_cfg.get("loss_weights", [1.0, 1.0, 1.0, 1.0])),
        inverse_depth=depth_cfg.get("inverse_depth", True),
    )).to(device)
    depth_gt_ms, mask_ms = create_multiscale_gt(depth_gt, mask)
    loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
    print("  Loss sanity:")
    for key in ["stage1", "stage2", "stage3", "stage4", "total"]:
        if key in loss_dict:
            value = float(loss_dict[key].detach().cpu().item())
            print(f"    {key}: {value:.6f}")
    
    # --------------------------------------------------------
    # 4. Save stage depth maps
    # --------------------------------------------------------
    print("\n[4/4] Saving stage depth maps...")
    
    # Setup save directory
    runs_dir = get_runs_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_root = runs_dir / f"vis_{timestamp}"
    save_dir = vis_root / "depth_stages"
    input_dir = vis_root / "input_views"

    # Save original input images used by all stages.
    num_views = images.shape[1]
    print(f"  input views used by all stages: V={num_views}, ids={meta.get('views')}")
    for view_idx in range(num_views):
        view_img = images[0, view_idx]
        view_path = input_dir / f"view_{view_idx:02d}.png"
        save_rgb_image(view_img, view_path, title=f"Input view {view_idx} (id={meta.get('views')[view_idx]})")
        print(f"    input image saved -> {view_path}")

    vmin = float(depth_range[0, 0].item())
    vmax = float(depth_range[0, 1].item())
    print(f"  depth visualization vmin/vmax: {vmin:.3f}/{vmax:.3f}")

    stage_keys = ["stage1", "stage2", "stage3", "stage4"]
    for stage_key in stage_keys:
        stage_out = outputs.get(stage_key)
        if not isinstance(stage_out, dict) or "depth" not in stage_out:
            print(f"  [SKIP] {stage_key}: depth not found")
            continue
        depth_map_native = stage_out["depth"][0]  # (H_s, W_s)
        depth_map_full = torch.nn.functional.interpolate(
            depth_map_native.unsqueeze(0).unsqueeze(0),
            size=(full_h, full_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        print(
            f"  {stage_key}: native_shape={tuple(depth_map_native.shape)} "
            f"native_min={depth_map_native.min().item():.3f} native_max={depth_map_native.max().item():.3f}"
        )
        save_path_full = save_dir / f"{stage_key}_depth.png"
        save_depth_map(depth_map_full, save_path_full, title=f"{stage_key} depth (upsampled)", vmin=vmin, vmax=vmax)
        print(f"    depth(full) -> {save_path_full}")

        # Keep the native resolution depth too (useful when debugging stage behavior).
        save_path_native = save_dir / f"{stage_key}_depth_native.png"
        save_depth_map(depth_map_native, save_path_native, title=f"{stage_key} depth (native)", vmin=vmin, vmax=vmax)
        print(f"    depth(native) -> {save_path_native}")

        # Save a reference view image at full resolution for easy side-by-side comparison.
        stage_img_path = save_dir / f"{stage_key}_image.png"
        save_rgb_image(images[0, 0], stage_img_path, title=f"{stage_key} image (ref view)")
        print(f"    image -> {stage_img_path}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
