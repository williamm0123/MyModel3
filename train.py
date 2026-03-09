#!/usr/bin/env python3
"""
Quick training script for MyModel3 (MVSFormer++).

Default behavior is debug-friendly:
- 1 epoch
- TensorBoard logs under ../log/tensorboard/
- optional step limits for fast sanity training
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid


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
    # Use module dict lookup instead of hasattr(np, name) to avoid
    # triggering NumPy FutureWarning on deprecated alias access.
    for name, value in np_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


_ensure_tensorboard_numpy_compat()
from torch.utils.tensorboard import SummaryWriter

from data.dtu_data import DTUData
from utils.config import load_cfg
from models.network.network import Network
from models.network.Depth_estimator import DepthEstimator, DepthEstimatorCfg
from models.losses import MultiStageLoss, LossCfg, create_multiscale_gt


def _dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _dist_enabled() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_enabled() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _maybe_set_slurm_dist_env() -> None:
    """
    Allow launching with SLURM `srun` without torchrun.

    Works reliably for single-node multi-GPU jobs. For multi-node jobs, set
    MASTER_ADDR/MASTER_PORT explicitly (or use torchrun).
    """
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    if "MASTER_ADDR" not in os.environ:
        # For single-node jobs 127.0.0.1 is sufficient.
        os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"


def _setup_distributed() -> Tuple[bool, int]:
    _maybe_set_slurm_dist_env()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    if _is_main_process():
        print(
            f"[DDP] enabled world_size={_get_world_size()} "
            f"backend={backend} MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}"
        )
    return True, local_rank


def _cleanup_distributed() -> None:
    if _dist_enabled():
        dist.destroy_process_group()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _normalize_input_path(path_like: str, *, project_root: Path) -> str:
    """
    Normalize user-input path from CLI.
    Supports: absolute path, relative path, ~/, $VAR, ${VAR}, and common typo '$/...'.
    """
    raw = str(path_like).strip()
    if raw.startswith("$/"):
        raw = raw[1:]
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in path: {path_like}. "
            "Please use an absolute path or export the variable first."
        )
    p = Path(expanded)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return str(p)


def _wrap_ddp(
    model: nn.Module,
    local_rank: int,
    *,
    find_unused_parameters: bool,
    static_graph: bool,
) -> nn.Module:
    """
    Wrap model with DDP using performant defaults while keeping compatibility
    across PyTorch versions.
    """
    ddp_kwargs: Dict[str, Any] = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "find_unused_parameters": find_unused_parameters,
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
    }
    ddp_sig = inspect.signature(DDP.__init__)
    if static_graph and ("static_graph" in ddp_sig.parameters):
        ddp_kwargs["static_graph"] = True
    return DDP(model, **ddp_kwargs)


def _normalize_depth_type(depth_type: Any) -> str:
    """Normalize depth type aliases to {'ce', 'reg'}."""
    key = str(depth_type).strip().lower()
    mapping = {
        "ce": "ce",
        "cross_entropy": "ce",
        "cross-entropy": "ce",
        "classification": "ce",
        "cls": "ce",
        "reg": "reg",
        "re": "reg",
        "regression": "reg",
        "l1": "reg",
        "smooth_l1": "reg",
        "smoothl1": "reg",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported depth type: {depth_type}")
    return mapping[key]


def _to_stage_list(raw: Any, *, default: Any, nstage: int = 4) -> List[Any]:
    if raw is None:
        values = [default]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    if len(values) == 0:
        values = [default]
    if len(values) < nstage:
        values = values + [values[-1]] * (nstage - len(values))
    else:
        values = values[:nstage]
    return values


def _normalize_model_and_loss_cfg(cfg_dict: Dict[str, Any]) -> None:
    """
    Normalize/align model depth type and loss depth types.
    This prevents silent mismatch across different config schemas.
    """
    loss_cfg = dict(cfg_dict.get("loss", {}))
    depth_cfg = dict(cfg_dict.get("depth", {}))
    arch_cfg = cfg_dict.get("arch", {})
    arch_args = arch_cfg.get("args", {}) if isinstance(arch_cfg, dict) else {}
    arch_loss = arch_cfg.get("loss", {}) if isinstance(arch_cfg, dict) else {}

    # Fill missing depth config from common legacy keys.
    if "inverse_depth" not in depth_cfg:
        depth_cfg["inverse_depth"] = bool(arch_args.get("inverse_depth", True))
    if "ndepths" not in depth_cfg and isinstance(arch_args.get("ndepths"), (list, tuple)):
        depth_cfg["ndepths"] = list(arch_args["ndepths"])
    if "depth_interval_ratios" not in depth_cfg and isinstance(arch_args.get("depth_interals_ratio"), (list, tuple)):
        depth_cfg["depth_interval_ratios"] = list(arch_args["depth_interals_ratio"])

    if "base_chs" not in depth_cfg:
        fpn_cfg = cfg_dict.get("fpn", {})
        feat_chs = fpn_cfg.get("feat_chs") if isinstance(fpn_cfg, dict) else None
        if isinstance(feat_chs, (list, tuple)) and len(feat_chs) >= 4:
            depth_cfg["base_chs"] = [int(feat_chs[3]), int(feat_chs[2]), int(feat_chs[1]), int(feat_chs[0])]

    # Resolve model depth type from depth/loss/legacy keys.
    model_depth_raw = depth_cfg.get("depth_type", None)
    if model_depth_raw is None:
        model_depth_raw = loss_cfg.get("depth_type", None)
    if model_depth_raw is None:
        model_depth_raw = arch_args.get("depth_type", None)
    if isinstance(model_depth_raw, (list, tuple)):
        model_depth_raw = model_depth_raw[0] if len(model_depth_raw) > 0 else None

    # Resolve loss depth types from loss/depth/legacy keys.
    loss_depth_raw = loss_cfg.get("depth_types", None)
    if loss_depth_raw is None:
        loss_depth_raw = loss_cfg.get("depth_type", None)
    if loss_depth_raw is None:
        loss_depth_raw = model_depth_raw
    if loss_depth_raw is None:
        loss_depth_raw = arch_args.get("depth_type", None)

    loss_depth_types = [_normalize_depth_type(v) for v in _to_stage_list(loss_depth_raw, default="ce", nstage=4)]
    # Current depth estimator only supports a single depth type across stages.
    if len(set(loss_depth_types)) != 1:
        loss_depth_types = [loss_depth_types[0]] * 4

    model_depth_norm = _normalize_depth_type(model_depth_raw) if model_depth_raw is not None else loss_depth_types[0]
    if model_depth_norm != loss_depth_types[0]:
        # Enforce consistency: use the loss type as source of truth.
        model_depth_norm = loss_depth_types[0]

    # Resolve loss weights (support both loss_weights and legacy dlossw).
    loss_weights_raw = loss_cfg.get("loss_weights", None)
    if loss_weights_raw is None:
        loss_weights_raw = loss_cfg.get("dlossw", None)
    if loss_weights_raw is None and isinstance(arch_loss, dict):
        loss_weights_raw = arch_loss.get("dlossw", None)
    loss_weights = [float(v) for v in _to_stage_list(loss_weights_raw, default=1.0, nstage=4)]

    if "clip_func" not in loss_cfg and isinstance(arch_loss, dict) and ("clip_func" in arch_loss):
        loss_cfg["clip_func"] = arch_loss.get("clip_func")

    loss_cfg["depth_types"] = loss_depth_types
    loss_cfg["loss_weights"] = loss_weights
    depth_cfg["depth_type"] = "ce" if model_depth_norm == "ce" else "regression"

    cfg_dict["loss"] = loss_cfg
    cfg_dict["depth"] = depth_cfg


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """AdamW with no-weight-decay for bias/norm/1D parameters."""
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_l = name.lower()
        if param.ndim <= 1 or name_l.endswith(".bias") or ("norm" in name_l) or ("bn" in name_l):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: List[Dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return optim.AdamW(param_groups, lr=lr)


def _all_reduce_epoch_stats(
    total_stats: Dict[str, float],
    num_steps: int,
    device: torch.device,
) -> Tuple[Dict[str, float], int]:
    if not _dist_enabled():
        return total_stats, num_steps

    keys = sorted(total_stats.keys())
    values = [float(total_stats[k]) for k in keys]
    payload = torch.tensor(values + [float(num_steps)], device=device, dtype=torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)

    reduced_steps = int(payload[-1].item())
    reduced_stats = {k: float(payload[i].item()) for i, k in enumerate(keys)}
    return reduced_stats, reduced_steps


def _log_cuda_memory(tag: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.max_memory_allocated(device=device)
    reserved = torch.cuda.max_memory_reserved(device=device)
    payload = torch.tensor([allocated, reserved], device=device, dtype=torch.float64)
    if _dist_enabled():
        dist.all_reduce(payload, op=dist.ReduceOp.MAX)
    if _is_main_process():
        alloc_gb = float(payload[0].item()) / (1024**3)
        res_gb = float(payload[1].item()) / (1024**3)
        print(f"[CUDA][{tag}] max_alloc={alloc_gb:.2f}GiB max_reserved={res_gb:.2f}GiB")
    torch.cuda.reset_peak_memory_stats(device=device)


class MVSModel(nn.Module):
    """Feature extractor + multi-stage depth estimator."""

    def __init__(self, cfg_dict: Dict[str, Any], device: torch.device):
        super().__init__()
        self.network = Network(cfg_dict, device=device)
        depth_cfg = cfg_dict.get("depth", {})
        depth_type_norm = _normalize_depth_type(depth_cfg.get("depth_type", "ce"))
        ndepths = tuple(int(v) for v in _to_stage_list(depth_cfg.get("ndepths", [32, 16, 8, 4]), default=4, nstage=4))
        base_chs = tuple(int(v) for v in _to_stage_list(depth_cfg.get("base_chs", [64, 32, 16, 8]), default=8, nstage=4))
        depth_interval_ratios = tuple(
            float(v) for v in _to_stage_list(depth_cfg.get("depth_interval_ratios", [4.0, 2.67, 1.5, 1.0]), default=1.0, nstage=4)
        )
        self.depth_estimator = DepthEstimator(DepthEstimatorCfg(
            ndepths=ndepths,
            base_chs=base_chs,
            depth_interval_ratios=depth_interval_ratios,
            inverse_depth=depth_cfg.get("inverse_depth", True),
            depth_type="ce" if depth_type_norm == "ce" else "regression",
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
    depth_types = tuple(_normalize_depth_type(v) for v in _to_stage_list(loss_cfg.get("depth_types"), default="ce", nstage=4))
    loss_weights = tuple(float(v) for v in _to_stage_list(loss_cfg.get("loss_weights"), default=1.0, nstage=4))
    return MultiStageLoss(LossCfg(
        depth_types=depth_types,
        loss_weights=loss_weights,
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


def _depth_to_image_range(
    depth: torch.Tensor,
    depth_min: float,
    depth_max: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert depth (H,W) to a single-channel normalized image (1,H,W) using a fixed range.
    """
    if depth.dim() != 2:
        raise ValueError(f"Expected depth dim=2, got {tuple(depth.shape)}")
    denom = max(float(depth_max - depth_min), 1e-8)
    image = (depth.float() - float(depth_min)) / denom
    image = torch.clamp(image, 0.0, 1.0)
    if valid_mask is not None:
        if valid_mask.dim() != 2:
            raise ValueError(f"Expected valid_mask dim=2, got {tuple(valid_mask.shape)}")
        invalid = valid_mask <= 0.5
        image = image.masked_fill(invalid, 0.0)
    return image.unsqueeze(0)


def _upsample_2d_to(
    x_hw: torch.Tensor,
    size_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample a single 2D tensor (H,W) to size_hw and return (H2,W2).
    """
    if x_hw.dim() != 2:
        raise ValueError(f"Expected (H,W), got {tuple(x_hw.shape)}")
    return F.interpolate(
        x_hw.unsqueeze(0).unsqueeze(0),
        size=size_hw,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    ).squeeze(0).squeeze(0)


def log_training_images(
    writer: Optional[SummaryWriter],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    global_step: int,
) -> None:
    """
    Log training images/features for TensorBoard.
    """
    if writer is None:
        return
    images = batch["images"]
    # Log a grid of all views for the first sample (B=0).
    views = images[0].detach().cpu().clamp(0, 1)  # (V,3,H,W)
    writer.add_image("train_images/ref_view", views[0], global_step)
    for v in range(1, min(int(views.shape[0]), 6)):
        writer.add_image(f"train_images/src_view{v}", views[v], global_step)
    try:
        writer.add_image("train_images/views_grid", make_grid(views, nrow=int(views.shape[0])), global_step)
    except Exception:
        # Fallback to per-view images only.
        pass

    depth_gt = batch.get("depth_gt", None)
    mask = batch.get("mask", None)
    depth_range = batch.get("depth_range", None)
    depth_min, depth_max = None, None
    if depth_range is not None and depth_range.numel() >= 2:
        depth_min = float(depth_range[0, 0].detach().cpu().item())
        depth_max = float(depth_range[0, 1].detach().cpu().item())

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

    # Pred depth per stage (stage1..4) and final depth.
    # We treat stage{1..4} as the "different scales" (1/8, 1/4, 1/2, 1).
    stage_pred_up: List[torch.Tensor] = []
    stage_gt_up: List[torch.Tensor] = []
    H_full, W_full = int(views.shape[-2]), int(views.shape[-1])
    stage_scales = {1: "1_8", 2: "1_4", 3: "1_2", 4: "1_1"}

    for stage_idx in range(1, 5):
        stage_key = f"stage{stage_idx}"
        if stage_key in outputs and isinstance(outputs[stage_key], dict):
            stage_depth = outputs[stage_key]["depth"][0].detach().cpu()  # (H_s,W_s)
            stage_conf = outputs[stage_key].get("photometric_confidence", None)
            stage_conf = None if stage_conf is None else stage_conf[0].detach().cpu()

            tag_scale = stage_scales.get(stage_idx, f"stage{stage_idx}")
            if depth_min is not None and depth_max is not None:
                writer.add_image(
                    f"train_depth/pred_stage{stage_idx}_{tag_scale}",
                    _depth_to_image_range(stage_depth, depth_min, depth_max),
                    global_step,
                )
                # Backward-compatible tag.
                writer.add_image(
                    f"train_depth/stage{stage_idx}",
                    _depth_to_image_range(stage_depth, depth_min, depth_max),
                    global_step,
                )
            else:
                writer.add_image(
                    f"train_depth/pred_stage{stage_idx}_{tag_scale}",
                    _depth_to_image(stage_depth),
                    global_step,
                )
                writer.add_image(
                    f"train_depth/stage{stage_idx}",
                    _depth_to_image(stage_depth),
                    global_step,
                )
            if stage_conf is not None:
                writer.add_image(
                    f"train_depth/conf_stage{stage_idx}_{tag_scale}",
                    _normalize_for_vis(stage_conf).unsqueeze(0),
                    global_step,
                )

            # Upsampled-to-full-res grid for quick comparison across stages.
            stage_depth_up = _upsample_2d_to(stage_depth, (H_full, W_full), mode="bilinear")
            stage_pred_up.append(stage_depth_up)

            if depth_gt is not None:
                gt_full = depth_gt[0].detach().cpu()
                gt_stage = _upsample_2d_to(gt_full, tuple(stage_depth.shape), mode="bilinear")
                stage_gt_up.append(_upsample_2d_to(gt_stage, (H_full, W_full), mode="bilinear"))

                # Per-stage error at native stage resolution.
                if mask is not None:
                    mask_full = mask[0].detach().cpu()
                    mask_stage = _upsample_2d_to(mask_full, tuple(stage_depth.shape), mode="nearest")
                else:
                    mask_stage = None
                err = (stage_depth - gt_stage).abs()
                if mask_stage is not None:
                    err = err.masked_fill(mask_stage <= 0.5, 0.0)
                writer.add_image(
                    f"train_depth/abs_err_stage{stage_idx}_{tag_scale}",
                    _normalize_for_vis(err).unsqueeze(0),
                    global_step,
                )

    if "depth" in outputs:
        pred_full = outputs["depth"][0].detach().cpu()
        if depth_min is not None and depth_max is not None:
            writer.add_image(
                "train_depth/pred_final",
                _depth_to_image_range(pred_full, depth_min, depth_max),
                global_step,
            )
            writer.add_image(
                "train_depth/final",
                _depth_to_image_range(pred_full, depth_min, depth_max),
                global_step,
            )
        else:
            writer.add_image("train_depth/pred_final", _depth_to_image(pred_full), global_step)
            writer.add_image("train_depth/final", _depth_to_image(pred_full), global_step)

        if depth_gt is not None:
            gt_full = depth_gt[0].detach().cpu()
            if mask is not None:
                mask_full = mask[0].detach().cpu()
            else:
                mask_full = None

            if depth_min is not None and depth_max is not None:
                writer.add_image(
                    "train_depth/gt_final",
                    _depth_to_image_range(gt_full, depth_min, depth_max, valid_mask=mask_full),
                    global_step,
                )
            else:
                writer.add_image("train_depth/gt_final", _depth_to_image(gt_full), global_step)

            err = (pred_full - gt_full).abs()
            if mask_full is not None:
                err = err.masked_fill(mask_full <= 0.5, 0.0)
                writer.add_image("train_depth/mask_final", mask_full.unsqueeze(0).float(), global_step)
            writer.add_image("train_depth/abs_err_final", _normalize_for_vis(err).unsqueeze(0), global_step)

    # Stage prediction grids (upsampled to full resolution so they tile nicely).
    if len(stage_pred_up) == 4:
        pred_stack = torch.stack(stage_pred_up, dim=0).unsqueeze(1)  # (4,1,H,W)
        try:
            writer.add_image(
                "train_depth/pred_stages_up_grid",
                make_grid(pred_stack, nrow=2),
                global_step,
            )
        except Exception:
            pass
    if len(stage_gt_up) == 4:
        gt_stack = torch.stack(stage_gt_up, dim=0).unsqueeze(1)  # (4,1,H,W)
        try:
            writer.add_image(
                "train_depth/gt_stages_up_grid",
                make_grid(gt_stack, nrow=2),
                global_step,
            )
        except Exception:
            pass


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
    model_to_save = _unwrap_model(model)
    network_state = model_to_save.network.state_dict() if hasattr(model_to_save, "network") else None
    depth_estimator_state = model_to_save.depth_estimator.state_dict() if hasattr(model_to_save, "depth_estimator") else None
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model_state_dict": model_to_save.state_dict(),
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
    writer: Optional[SummaryWriter],
    device: torch.device,
    epoch: int,
    use_amp: bool,
    grad_clip: float,
    log_interval: int,
    image_log_interval: int,
    log_intermediates: bool,
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

        # Never request intermediates unless we're actually going to log them (saves memory).
        should_log_images = (
            writer is not None
            and (image_log_interval > 0)
            and (global_step % image_log_interval == 0)
        )
        request_intermediate = bool(should_log_images and log_intermediates)

        if scaler is not None:
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images, proj_matrices, depth_range, return_intermediate=request_intermediate)
                loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
                total_loss = loss_dict["total"]
            
            # Check for NaN/Inf in loss before backward
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                if _is_main_process():
                    print(f"[WARNING] Step {step_idx}: Loss is NaN/Inf, skipping this step...")
                optimizer.zero_grad(set_to_none=True)
                num_steps += 1
                global_step += 1
                continue
            
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= prev_scale:
                scheduler.step()
        else:
            outputs = model(images, proj_matrices, depth_range, return_intermediate=request_intermediate)
            loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
            total_loss = loss_dict["total"]
            
            # Check for NaN/Inf in loss before backward
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                if _is_main_process():
                    print(f"[WARNING] Step {step_idx}: Loss is NaN/Inf, skipping this step...")
                optimizer.zero_grad(set_to_none=True)
                num_steps += 1
                global_step += 1
                continue
            
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
            if writer is not None:
                writer.add_scalar(f"train/{key}", value, global_step)
        if writer is not None and "total" in step_stats:
            # Dedicated loss curve tags for easier TensorBoard filtering.
            writer.add_scalar("loss/train_total_step", step_stats["total"], global_step)

        if writer is not None:
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/depth_mean", float(outputs["depth"].mean().detach().item()), global_step)
            writer.add_scalar("train/depth_std", float(outputs["depth"].std().detach().item()), global_step)
        if should_log_images:
            cpu_batch = {
                "images": images.detach().cpu(),
                "depth_gt": depth_gt.detach().cpu(),
                "mask": mask.detach().cpu(),
                "depth_range": depth_range.detach().cpu(),
            }
            log_training_images(writer, cpu_batch, outputs, global_step)

        if _is_main_process() and (step_idx % log_interval == 0):
            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1} Step {step_idx}/{len(loader)} "
                f"loss={step_stats['total']:.6f} lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
            )

    if num_steps == 0:
        return {"total": 0.0}, global_step

    reduced_stats, reduced_steps = _all_reduce_epoch_stats(total_stats, num_steps, device)
    if reduced_steps <= 0:
        return {"total": 0.0}, global_step
    avg_stats = {k: v / reduced_steps for k, v in reduced_stats.items()}
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

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, proj_matrices, depth_range)
            loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)

        for key, value in loss_dict.items():
            total_stats[key] = total_stats.get(key, 0.0) + float(value.detach().item())
        num_steps += 1

    if num_steps == 0:
        reduced_stats, reduced_steps = _all_reduce_epoch_stats({"total": 0.0}, 0, device)
        return {"total": 0.0} if reduced_steps == 0 else {k: v / reduced_steps for k, v in reduced_stats.items()}

    reduced_stats, reduced_steps = _all_reduce_epoch_stats(total_stats, num_steps, device)
    if reduced_steps <= 0:
        return {"total": 0.0}
    return {k: v / reduced_steps for k, v in reduced_stats.items()}


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
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs (default: 16)")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size (per GPU)")
    parser.add_argument("--num_workers", type=int, default=-1, help="Override dataloader workers")
    parser.add_argument("--lr", type=float, default=0.0, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, default=-1.0, help="Override weight decay")
    parser.add_argument("--max_train_steps", type=int, default=0, help="Limit steps per epoch (0 = full epoch)")
    parser.add_argument("--max_val_steps", type=int, default=0, help="Limit val steps (0 = full val)")
    parser.add_argument("--no_val", action="store_true", help="Skip validation")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--log_interval", type=int, default=10, help="Console log interval")
    parser.add_argument(
        "--image_log_interval",
        type=int,
        default=-1,
        help="TensorBoard image log interval in steps; <=0 disables image logging to save memory",
    )
    parser.add_argument(
        "--log_intermediates",
        action="store_true",
        help="Also request and log network intermediate features (higher memory); off by default",
    )
    parser.add_argument("--tb_root", type=str, default="../log/tensorboard", help="TensorBoard root directory")
    parser.add_argument("--ckpt_root", type=str, default="../log/checkpoints", help="Checkpoint root directory")
    parser.add_argument("--run_name", type=str, default="", help="TensorBoard run name")
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection (slower; only use when needed).",
    )
    parser.add_argument(
        "--no_static_graph",
        action="store_true",
        help="Disable DDP static_graph optimization.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    distributed, local_rank = _setup_distributed()

    try:
        cfg_dict = load_cfg(args.config)

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
            cfg_dict["datapath"] = _normalize_input_path(args.data_root, project_root=project_root)

        # Keep model depth type and loss depth types consistent across config variants.
        _normalize_model_and_loss_cfg(cfg_dict)

        train_cfg = cfg_dict["train"]
        batch_size = int(train_cfg.get("batch_size", 1))
        num_workers = int(train_cfg.get("num_workers", 2))
        lr = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        use_amp = bool(train_cfg.get("use_amp", True)) and (not args.no_amp)
        lr_scheduler_name = str(train_cfg.get("lr_scheduler", "cosine")).lower()
        warmup_steps_cfg = int(train_cfg.get("warmup_steps", 0))
        lr_min_ratio = float(train_cfg.get("lr_min_ratio", 0.1))
        short_cosine_steps = int(train_cfg.get("short_cosine_steps", 2000))
        short_cosine_lr_floor = float(train_cfg.get("short_cosine_lr_floor", 0.2))
        short_cosine_warmup_ratio = float(train_cfg.get("short_cosine_warmup_ratio", 0.1))
        persistent_workers_cfg = bool(train_cfg.get("persistent_workers", True))

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")

        if _is_main_process():
            print(f"Using device: {device}")
            if device.type == "cuda":
                idx = 0 if device.index is None else int(device.index)
                print(f"GPU: {torch.cuda.get_device_name(idx)}")
                print(f"[CUDA] visible_device_count={torch.cuda.device_count()}")
        if not distributed and _is_main_process() and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("[DDP] disabled (WORLD_SIZE=1). Launch with torchrun or srun to use multiple GPUs.")

        if args.mock:
            num_views = len(cfg_dict.get("views", [0, 1, 2]))
            train_dataset = MockMVSDataset(num_samples=64, num_views=num_views)
            val_dataset = MockMVSDataset(num_samples=16, num_views=num_views) if not args.no_val else None
            if args.num_workers < 0:
                num_workers = 0
        else:
            if not cfg_dict.get("datapath", ""):
                raise ValueError("No datapath configured. Set 'datapath' in config or pass --data_root.")
            data_root_path = Path(str(cfg_dict["datapath"]))
            if not data_root_path.exists():
                raise FileNotFoundError(
                    f"Dataset root not found: {data_root_path}. "
                    "Please check --data_root or config.datapath."
                )
            train_dataset = DTUData(cfg_dict, split="train", sample_mode="mvs")
            val_dataset = None if args.no_val else DTUData(cfg_dict, split="val", sample_mode="mvs")

        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if distributed else None
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            if (distributed and val_dataset is not None)
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(persistent_workers_cfg and num_workers > 0),
            drop_last=True,
        )
        val_loader = None if val_dataset is None else DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(persistent_workers_cfg and num_workers > 0),
        )

        model: nn.Module = MVSModel(cfg_dict, device=device).to(device)
        if distributed:
            model = _wrap_ddp(
                model,
                local_rank,
                find_unused_parameters=bool(args.find_unused_parameters),
                static_graph=(not args.no_static_graph),
            )

        loss_fn = build_loss_fn(cfg_dict).to(device)
        optimizer = _build_optimizer(_unwrap_model(model), lr=lr, weight_decay=weight_decay)

        steps_per_epoch = len(train_loader) if args.max_train_steps <= 0 else min(len(train_loader), args.max_train_steps)
        total_train_steps = max(1, args.epochs * max(1, steps_per_epoch))
        warmup_steps = max(0, min(warmup_steps_cfg, max(0, total_train_steps - 1)))
        effective_lr_min_ratio = lr_min_ratio
        if (lr_scheduler_name == "cosine") and (total_train_steps <= short_cosine_steps):
            # Avoid decaying LR too aggressively for short debug/quick runs.
            effective_lr_min_ratio = max(lr_min_ratio, short_cosine_lr_floor)
            if warmup_steps == 0:
                warmup_steps = max(1, int(total_train_steps * short_cosine_warmup_ratio))
                warmup_steps = min(warmup_steps, max(0, total_train_steps - 1))
        if lr_scheduler_name == "cosine":
            if warmup_steps > 0:
                def _warmup_cosine_lambda(step: int) -> float:
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return effective_lr_min_ratio + (1.0 - effective_lr_min_ratio) * cosine

                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine_lambda)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_train_steps,
                    eta_min=lr * effective_lr_min_ratio,
                )
        elif lr_scheduler_name in {"none", "constant"}:
            scheduler = None
        else:
            raise ValueError(f"Unsupported train.lr_scheduler={lr_scheduler_name}. Use cosine/none.")
        scaler = GradScaler(device=device.type, enabled=(use_amp and device.type == "cuda"))

        run_name = args.run_name.strip()
        if not run_name and _is_main_process():
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if distributed:
            name_list: List[str] = [run_name]
            dist.broadcast_object_list(name_list, src=0)
            run_name = str(name_list[0])

        tb_log_dir = resolve_tb_log_dir(project_root, args.tb_root, run_name) if _is_main_process() else None
        ckpt_root = Path(args.ckpt_root)
        if not ckpt_root.is_absolute():
            ckpt_root = project_root / ckpt_root
        ckpt_dir = ckpt_root / run_name
        latest_ckpt = ckpt_dir / "latest.pth"
        best_ckpt = ckpt_dir / "best.pth"
        writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(tb_log_dir)) if tb_log_dir is not None else None

        if _is_main_process():
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
            print(f"Train batches (per-rank): {len(train_loader)} world_size={_get_world_size()}")
            model_depth_type = _unwrap_model(model).depth_estimator.cfg.depth_type
            print(f"[Depth/Loss] model_depth_type={model_depth_type} loss_depth_types={list(loss_fn.cfg.depth_types)}")
            if distributed:
                print(
                    f"[DDP] find_unused_parameters={bool(args.find_unused_parameters)} "
                    f"static_graph={not args.no_static_graph}"
                )
            print(
                f"[Scheduler] type={lr_scheduler_name} warmup_steps={warmup_steps} "
                f"lr_min_ratio={effective_lr_min_ratio:.4f} total_steps={total_train_steps}"
            )
            if val_loader is not None:
                print(f"Val batches (per-rank): {len(val_loader)}")

        global_step = 0
        best_metric = float("inf")
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            if _is_main_process():
                print("\n" + "=" * 60)
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
                image_log_interval=args.image_log_interval,
                log_intermediates=args.log_intermediates,
                max_steps=args.max_train_steps,
                global_step=global_step,
                scheduler=scheduler,
            )
            if _is_main_process():
                print(f"Train: {train_stats}")
            if writer is not None:
                for key, value in train_stats.items():
                    writer.add_scalar(f"train_epoch/{key}", value, epoch)
                if "total" in train_stats:
                    writer.add_scalar("loss/train_total_epoch", float(train_stats["total"]), epoch)

            if val_loader is not None:
                val_stats = evaluate_one_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    max_steps=args.max_val_steps,
                    use_amp=(use_amp and device.type == "cuda"),
                )
                if _is_main_process():
                    print(f"Val: {val_stats}")
                if writer is not None:
                    for key, value in val_stats.items():
                        writer.add_scalar(f"val_epoch/{key}", value, epoch)
                    if ("total" in train_stats) and ("total" in val_stats):
                        writer.add_scalar("loss/total_epoch_compare/train", float(train_stats["total"]), epoch)
                        writer.add_scalar("loss/total_epoch_compare/val", float(val_stats["total"]), epoch)
                    # Also compare per-stage losses when both sides are available.
                    for key, train_value in train_stats.items():
                        if (key == "total") or (key not in val_stats):
                            continue
                        writer.add_scalar(f"loss/{key}_epoch_compare/train", float(train_value), epoch)
                        writer.add_scalar(f"loss/{key}_epoch_compare/val", float(val_stats[key]), epoch)
                current_metric = float(val_stats.get("total", train_stats.get("total", float("inf"))))
            else:
                current_metric = float(train_stats.get("total", float("inf")))

            if _is_main_process():
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

            if writer is not None:
                writer.flush()
            _log_cuda_memory(tag=f"epoch{epoch+1}", device=device)

        if writer is not None:
            writer.close()
        if _is_main_process():
            print("\nTraining finished.")
            if best_metric < float("inf"):
                print(f"Best metric: {best_metric:.6f}")
            print(f"Latest checkpoint: {latest_ckpt}")
            print(f"Best checkpoint: {best_ckpt}")
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
