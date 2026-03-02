#!/usr/bin/env python3
"""
Quick training script for MyModel3 (MVSFormer++).

Default behavior is debug-friendly:
- 1 epoch
- TensorBoard logs under runs/tensorboard/
- optional step limits for fast sanity training
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


def _ensure_tensorboard_numpy_compat() -> None:
    """
    Compatibility shim for TensorBoard packages that still depend on removed NumPy aliases.
    This keeps `SummaryWriter` usable in environments with NumPy 2.x.
    """
    if not hasattr(builtins, "bool_"):
        builtins.bool_ = bool
    if not hasattr(builtins, "bool8"):
        builtins.bool8 = bool

    np_aliases = {
        "object": object,
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
    }
    for name, value in np_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


_ensure_tensorboard_numpy_compat()
from torch.utils.tensorboard import SummaryWriter

from data.dtu_data import DTUData
from models.network.network import Network
from models.network.Depth_estimator import DepthEstimator, DepthEstimatorCfg
from models.losses import MultiStageLoss, LossCfg, create_multiscale_gt


class MVSModel(nn.Module):
    """Feature extractor + multi-stage depth estimator."""

    def __init__(self, cfg_dict: Dict[str, Any], device: torch.device):
        super().__init__()
        self.network = Network(cfg_dict, device=device)
        depth_cfg = cfg_dict.get("depth", {})
        self.depth_estimator = DepthEstimator(DepthEstimatorCfg(
            ndepths=tuple(depth_cfg.get("ndepths", [32, 16, 8, 4])),
            base_chs=tuple(depth_cfg.get("base_chs", [64, 32, 16, 8])),
            depth_interval_ratios=tuple(depth_cfg.get("depth_interval_ratios", [4.0, 2.0, 1.0, 0.5])),
            inverse_depth=depth_cfg.get("inverse_depth", True),
            depth_type=depth_cfg.get("depth_type", "regression"),
            temperatures=tuple(depth_cfg.get("temperatures", [5.0, 5.0, 5.0, 1.0])),
        ))

    def forward(
        self,
        images: torch.Tensor,
        proj_matrices: Dict[str, torch.Tensor],
        depth_range: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        net_outputs = self.network(images, return_intermediate=return_intermediate)
        depth_features = {
            "stage1": net_outputs["stage1"],
            "stage2": net_outputs["stage2"],
            "stage3": net_outputs["stage3"],
            "stage4": net_outputs["stage4"],
        }
        outputs = self.depth_estimator(depth_features, proj_matrices, depth_range)
        if return_intermediate:
            outputs["intermediates"] = {
                key: value for key, value in net_outputs.items()
                if key not in {"stage1", "stage2", "stage3", "stage4"}
            }
        return outputs


class MockMVSDataset(Dataset):
    """Mock dataset for quick pipeline checks."""

    def __init__(self, num_samples: int = 64, num_views: int = 3, h: int = 256, w: int = 320):
        self.num_samples = num_samples
        self.num_views = num_views
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> Dict[str, Any]:
        v = self.num_views
        h, w = self.h, self.w
        images = torch.rand(v, 3, h, w)
        depth_gt = torch.rand(h, w) * 480.0 + 425.0
        mask = torch.ones(h, w)
        depth_range = torch.tensor([425.0, 905.0], dtype=torch.float32)
        depth_interval = torch.tensor((905.0 - 425.0) / 192.0, dtype=torch.float32)

        proj_matrices = {}
        for i in range(1, 5):
            proj_matrices[f"stage{i}"] = torch.eye(4).view(1, 1, 4, 4).repeat(v, 2, 1, 1)

        return {
            "images": images,
            "depth_gt": depth_gt,
            "mask": mask,
            "depth_range": depth_range,
            "depth_interval": depth_interval,
            "proj_matrices": proj_matrices,
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    images = batch["images"].to(device)
    depth_gt = batch["depth_gt"].to(device)
    mask = batch["mask"].to(device)
    depth_range = batch["depth_range"].to(device)
    depth_interval = batch["depth_interval"].to(device)
    proj_matrices = {key: value.to(device) for key, value in batch["proj_matrices"].items()}
    return images, depth_gt, mask, depth_range, depth_interval, proj_matrices


def build_loss_fn(cfg_dict: Dict[str, Any]) -> MultiStageLoss:
    loss_cfg = cfg_dict.get("loss", {})
    depth_cfg = cfg_dict.get("depth", {})
    return MultiStageLoss(LossCfg(
        depth_types=tuple(loss_cfg.get("depth_types", ["reg", "reg", "reg", "reg"])),
        loss_weights=tuple(loss_cfg.get("loss_weights", [1.0, 1.0, 1.0, 1.0])),
        inverse_depth=depth_cfg.get("inverse_depth", True),
        clip_func=loss_cfg.get("clip_func", None),
    ))


def _normalize_for_vis(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float()
    min_val = image.min()
    max_val = image.max()
    if (max_val - min_val) < 1e-8:
        return torch.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def _feature_map_to_image(feature: torch.Tensor) -> torch.Tensor:
    """
    Convert feature map (C,H,W) to a single-channel normalized image (1,H,W).
    """
    if feature.dim() != 3:
        raise ValueError(f"Expected feature dim=3, got {tuple(feature.shape)}")
    image = feature.abs().mean(dim=0, keepdim=True)
    return _normalize_for_vis(image)


def _depth_to_image(depth: torch.Tensor) -> torch.Tensor:
    """
    Convert depth (H,W) to a single-channel normalized image (1,H,W).
    """
    if depth.dim() != 2:
        raise ValueError(f"Expected depth dim=2, got {tuple(depth.shape)}")
    valid = depth > 0
    if valid.any():
        min_val = depth[valid].min()
        max_val = depth[valid].max()
        image = (depth - min_val) / (max_val - min_val + 1e-8)
        image = torch.clamp(image, 0.0, 1.0)
    else:
        image = torch.zeros_like(depth)
    return image.unsqueeze(0)


def log_training_images(
    writer: SummaryWriter,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    global_step: int,
) -> None:
    """
    Log training images/features for TensorBoard.
    """
    images = batch["images"]
    writer.add_image("train_images/ref_view", images[0, 0].detach().cpu().clamp(0, 1), global_step)
    if images.shape[1] > 1:
        writer.add_image("train_images/src_view1", images[0, 1].detach().cpu().clamp(0, 1), global_step)

    intermediates = outputs.get("intermediates", {})
    for stage_idx in range(1, 5):
        sva_key = f"sva_stage{stage_idx}"
        fused_key = f"fpn_sva_stage{stage_idx}"
        if sva_key in intermediates:
            sva_feat = intermediates[sva_key][0, 0].detach().cpu()
            writer.add_image(
                f"train_features/sva_stage{stage_idx}",
                _feature_map_to_image(sva_feat),
                global_step,
            )
        if fused_key in intermediates:
            fused_feat = intermediates[fused_key][0, 0].detach().cpu()
            writer.add_image(
                f"train_features/fpn_sva_stage{stage_idx}",
                _feature_map_to_image(fused_feat),
                global_step,
            )

    for stage_idx in range(1, 5):
        stage_key = f"stage{stage_idx}"
        if stage_key in outputs and isinstance(outputs[stage_key], dict):
            stage_depth = outputs[stage_key]["depth"][0].detach().cpu()
            writer.add_image(
                f"train_depth/stage{stage_idx}",
                _depth_to_image(stage_depth),
                global_step,
            )

    if "depth" in outputs:
        writer.add_image(
            "train_depth/final",
            _depth_to_image(outputs["depth"][0].detach().cpu()),
            global_step,
        )


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    cfg_dict: Dict[str, Any],
    args_dict: Dict[str, Any],
    tb_log_dir: str,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    network_state = model.network.state_dict() if hasattr(model, "network") else None
    depth_estimator_state = model.depth_estimator.state_dict() if hasattr(model, "depth_estimator") else None
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "network_state_dict": network_state,
        "depth_estimator_state_dict": depth_estimator_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg_dict,
        "args": args_dict,
        "tb_log_dir": tb_log_dir,
    }
    torch.save(payload, ckpt_path)


def dump_run_params(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    loss_fn: nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    grad_clip: float,
    log_interval: int,
    image_log_interval: int,
    max_steps: int,
    global_step: int,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
) -> Tuple[Dict[str, float], int]:
    model.train()
    total_stats: Dict[str, float] = {}
    num_steps = 0
    start = time.time()

    for step_idx, batch in enumerate(loader):
        if max_steps > 0 and step_idx >= max_steps:
            break

        images, depth_gt, mask, depth_range, depth_interval, proj_matrices = move_batch_to_device(batch, device)
        depth_gt_ms, mask_ms = create_multiscale_gt(depth_gt, mask)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            should_log_images = (global_step % max(1, image_log_interval) == 0)
            with autocast(enabled=use_amp):
                outputs = model(images, proj_matrices, depth_range, return_intermediate=should_log_images)
                loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
                total_loss = loss_dict["total"]
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            should_log_images = (global_step % max(1, image_log_interval) == 0)
            outputs = model(images, proj_matrices, depth_range, return_intermediate=should_log_images)
            loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
            total_loss = loss_dict["total"]
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        num_steps += 1
        global_step += 1

        step_stats = {k: float(v.detach().item()) for k, v in loss_dict.items()}
        for key, value in step_stats.items():
            total_stats[key] = total_stats.get(key, 0.0) + value
            writer.add_scalar(f"train/{key}", value, global_step)

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/depth_mean", float(outputs["depth"].mean().detach().item()), global_step)
        writer.add_scalar("train/depth_std", float(outputs["depth"].std().detach().item()), global_step)
        if should_log_images:
            cpu_batch = {
                "images": images.detach().cpu(),
            }
            log_training_images(writer, cpu_batch, outputs, global_step)

        if step_idx % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1} Step {step_idx}/{len(loader)} "
                f"loss={step_stats['total']:.6f} lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
            )

    if num_steps == 0:
        return {"total": 0.0}, global_step

    avg_stats = {k: v / num_steps for k, v in total_stats.items()}
    return avg_stats, global_step


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_steps: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_stats: Dict[str, float] = {}
    num_steps = 0

    for step_idx, batch in enumerate(loader):
        if max_steps > 0 and step_idx >= max_steps:
            break
        images, depth_gt, mask, depth_range, depth_interval, proj_matrices = move_batch_to_device(batch, device)
        depth_gt_ms, mask_ms = create_multiscale_gt(depth_gt, mask)

        with autocast(enabled=use_amp):
            outputs = model(images, proj_matrices, depth_range)
            loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)

        for key, value in loss_dict.items():
            total_stats[key] = total_stats.get(key, 0.0) + float(value.detach().item())
        num_steps += 1

    if num_steps == 0:
        return {"total": 0.0}
    return {k: v / num_steps for k, v in total_stats.items()}


def resolve_tb_log_dir(project_root: Path, tb_root: str, run_name: str) -> Path:
    tb_root_path = Path(tb_root)
    if not tb_root_path.is_absolute():
        tb_root_path = project_root / tb_root_path
    tb_root_path.mkdir(parents=True, exist_ok=True)
    return tb_root_path / run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick MyModel3 training")
    parser.add_argument("--config", type=str, default="config/mvs.json", help="Path to mvs.json")
    parser.add_argument("--data_root", type=str, default="", help="Override datapath in config")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=-1, help="Override dataloader workers")
    parser.add_argument("--lr", type=float, default=0.0, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, default=-1.0, help="Override weight decay")
    parser.add_argument("--max_train_steps", type=int, default=0, help="Limit steps per epoch (0 = full epoch)")
    parser.add_argument("--max_val_steps", type=int, default=0, help="Limit val steps (0 = full val)")
    parser.add_argument("--no_val", action="store_true", help="Skip validation")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--log_interval", type=int, default=10, help="Console log interval")
    parser.add_argument("--image_log_interval", type=int, default=50, help="TensorBoard image log interval (steps)")
    parser.add_argument("--tb_root", type=str, default="runs/tensorboard", help="TensorBoard root directory")
    parser.add_argument("--ckpt_root", type=str, default="runs/checkpoints", help="Checkpoint root directory")
    parser.add_argument("--run_name", type=str, default="", help="TensorBoard run name")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    with open(args.config, "r", encoding="utf-8") as file:
        cfg_dict = json.load(file)

    cfg_dict.setdefault("train", {})
    cfg_dict["train"]["epochs"] = args.epochs
    if args.batch_size > 0:
        cfg_dict["train"]["batch_size"] = args.batch_size
    if args.num_workers >= 0:
        cfg_dict["train"]["num_workers"] = args.num_workers
    if args.lr > 0:
        cfg_dict["train"]["lr"] = args.lr
    if args.weight_decay >= 0:
        cfg_dict["train"]["weight_decay"] = args.weight_decay

    if args.data_root:
        cfg_dict["datapath"] = args.data_root

    train_cfg = cfg_dict["train"]
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 2))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    use_amp = bool(train_cfg.get("use_amp", True)) and (not args.no_amp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.mock:
        num_views = len(cfg_dict.get("views", [0, 1, 2]))
        train_dataset = MockMVSDataset(num_samples=64, num_views=num_views)
        val_dataset = MockMVSDataset(num_samples=16, num_views=num_views)
        # Debug-friendly default: avoid multiprocessing issues for quick mock runs.
        if args.num_workers < 0:
            num_workers = 0
    else:
        if not cfg_dict.get("datapath", ""):
            raise ValueError("No datapath configured. Set 'datapath' in config or pass --data_root.")
        train_dataset = DTUData(cfg_dict, split="train", sample_mode="mvs")
        val_dataset = DTUData(cfg_dict, split="val", sample_mode="mvs")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = None if args.no_val else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MVSModel(cfg_dict, device=device).to(device)
    loss_fn = build_loss_fn(cfg_dict).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader) if args.max_train_steps <= 0 else min(len(train_loader), args.max_train_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * max(1, steps_per_epoch)),
        eta_min=lr * 0.01,
    )
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = resolve_tb_log_dir(project_root, args.tb_root, run_name)
    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.is_absolute():
        ckpt_root = project_root / ckpt_root
    ckpt_dir = ckpt_root / run_name
    latest_ckpt = ckpt_dir / "latest.pth"
    best_ckpt = ckpt_dir / "best.pth"
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    run_meta = {
        "run_name": run_name,
        "tb_log_dir": str(tb_log_dir),
        "ckpt_dir": str(ckpt_dir),
        "latest_ckpt": str(latest_ckpt),
        "best_ckpt": str(best_ckpt),
        "cfg": cfg_dict,
        "args": vars(args),
    }
    dump_run_params(ckpt_dir / "run_params.json", run_meta)

    print(f"[TensorBoard] logdir: {tb_log_dir}")
    print("[TensorBoard] VSCode: Python: Launch TensorBoard -> select this logdir.")
    print(f"[Checkpoint] dir: {ckpt_dir}")
    print(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val batches: {len(val_loader)}")

    global_step = 0
    best_metric = float("inf")
    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        train_stats, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler if scaler.is_enabled() else None,
            loss_fn=loss_fn,
            writer=writer,
            device=device,
            epoch=epoch,
            use_amp=(use_amp and device.type == "cuda"),
            grad_clip=grad_clip,
            log_interval=max(1, args.log_interval),
            image_log_interval=max(1, args.image_log_interval),
            max_steps=args.max_train_steps,
            global_step=global_step,
            scheduler=scheduler,
        )
        print(f"Train: {train_stats}")
        for key, value in train_stats.items():
            writer.add_scalar(f"train_epoch/{key}", value, epoch)

        if val_loader is not None:
            val_stats = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                max_steps=args.max_val_steps,
                use_amp=(use_amp and device.type == "cuda"),
            )
            print(f"Val: {val_stats}")
            for key, value in val_stats.items():
                writer.add_scalar(f"val_epoch/{key}", value, epoch)
            current_metric = float(val_stats.get("total", train_stats.get("total", float("inf"))))
        else:
            current_metric = float(train_stats.get("total", float("inf")))

        # Always save latest checkpoint each epoch.
        save_checkpoint(
            ckpt_path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            cfg_dict=cfg_dict,
            args_dict=vars(args),
            tb_log_dir=str(tb_log_dir),
        )

        # Save best checkpoint.
        if current_metric < best_metric:
            best_metric = current_metric
            save_checkpoint(
                ckpt_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                cfg_dict=cfg_dict,
                args_dict=vars(args),
                tb_log_dir=str(tb_log_dir),
            )
            print(f"Best checkpoint updated: metric={best_metric:.6f}")

        writer.flush()

    writer.close()
    print("\nTraining finished.")
    if best_metric < float("inf"):
        print(f"Best metric: {best_metric:.6f}")
    print(f"Latest checkpoint: {latest_ckpt}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
