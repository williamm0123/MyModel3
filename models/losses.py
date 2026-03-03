# models/losses.py
"""
Loss functions for MVS depth estimation.

Implements:
    - Regression loss: Smooth L1 loss for depth regression
    - Cross-entropy loss: Classification loss for depth bin prediction
    - Multi-stage loss: Weighted combination across stages
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class LossCfg:
    """Configuration for loss functions."""
    # Loss type per stage: "ce" (cross-entropy) or "reg" (regression)
    depth_types: Tuple[str, ...] = ("reg", "reg", "reg", "reg")
    
    # Loss weights per stage
    loss_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    
    # Whether to use inverse depth
    inverse_depth: bool = True
    
    # Clip function for regression loss: None, "dynamic"
    clip_func: Optional[str] = None


# ============================================================
# Regression Loss
# ============================================================

def regression_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    depth_interval: Optional[torch.Tensor] = None,
    depth_values: Optional[torch.Tensor] = None,
    clip_func: Optional[str] = None,
    inverse_depth: bool = False,
) -> torch.Tensor:
    """
    Smooth L1 regression loss for depth.
    
    Args:
        depth_pred: (B, H, W) predicted depth
        depth_gt: (B, H, W) ground truth depth
        mask: (B, H, W) valid mask
        depth_interval: (B,) or (B, 1, 1) depth interval for normalization
        depth_values: (B, D, H, W) depth hypotheses for dynamic clipping
        clip_func: "dynamic" to clip loss by depth range
        inverse_depth: whether depth_values are in inverse order
    
    Returns:
        loss: scalar loss value
    """
    B, H, W = depth_pred.shape
    mask_bool = mask > 0.5
    
    # NaN detection for numerical stability
    if torch.isnan(depth_pred).any():
        print("Warning: depth_pred contains NaN, returning zero loss")
        return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)
    if torch.isnan(depth_gt).any():
        print("Warning: depth_gt contains NaN, returning zero loss")
        return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)
    
    # Compute smooth L1 loss
    if not mask_bool.any():
        return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)

    loss = F.smooth_l1_loss(depth_pred[mask_bool], depth_gt[mask_bool], reduction='none')
    
    # Check for NaN in loss
    if torch.isnan(loss).any():
        print("Warning: loss contains NaN after computation, returning zero loss")
        return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)
    
    # Normalize by depth interval AFTER computing the loss
    # This keeps the loss magnitude reasonable
    if depth_interval is not None:
        if depth_interval.dim() == 1:
            # depth_interval: (B,) -> need to select per-pixel values
            # Create per-pixel depth_interval using the mask
            # Expand depth_interval to (B, H, W) then use mask to get (N,)
            depth_interval_expanded = depth_interval.view(B, 1, 1).expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
        elif depth_interval.dim() == 3:
            # depth_interval: (B, 1, 1) -> expand to (B, H, W) then use mask
            depth_interval_expanded = depth_interval.expand(B, H, W)
            loss = loss / depth_interval_expanded[mask_bool]
    
    # Dynamic clipping
    if clip_func == 'dynamic' and depth_values is not None:
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
        depth_range = (depth_values[:, -1] - depth_values[:, 0])  # (B,)
        # Don't divide by depth_interval again since we already normalized the loss
        if depth_range.dim() == 1:
            depth_range_expanded = depth_range.view(B, 1, 1).expand(B, H, W)
        else:
            depth_range_expanded = depth_range.expand(B, H, W)
        depth_range_selected = depth_range_expanded[mask_bool]
        loss = torch.clamp_max(loss, depth_range_selected)
    
    return loss.mean()


# ============================================================
# Cross-Entropy Loss
# ============================================================

def cross_entropy_loss(
    prob_volume_pre: torch.Tensor,
    depth_values: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    inverse_depth: bool = True,
) -> torch.Tensor:
    """
    Cross-entropy classification loss for depth bins.
    
    Args:
        prob_volume_pre: (B, D, H, W) pre-softmax probability volume
        depth_values: (B, D, H, W) depth hypotheses
        depth_gt: (B, H, W) ground truth depth
        mask: (B, H, W) valid mask
        inverse_depth: whether to flip depth order
    
    Returns:
        loss: scalar loss value
    """
    mask = (mask > 0.5).float()
    B, D, H, W = depth_values.shape
    
    # Expand depth_gt
    depth_gt = depth_gt.unsqueeze(1)  # (B, 1, H, W)
    depth_gt_volume = depth_gt.expand_as(depth_values)  # (B, D, H, W)
    
    # Handle inverse depth ordering
    if inverse_depth:
        depth_values = torch.flip(depth_values, dims=[1])
        prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
    
    # Compute bin intervals
    intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2  # (B, D-1, H, W)
    intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # (B, D, H, W)
    
    # Compute bin boundaries
    min_depth_values = depth_values[:, 0:1] - intervals[:, 0:1]
    max_depth_values = depth_values[:, -1:] + intervals[:, -1:]
    depth_values_right = depth_values + intervals
    
    # Find out-of-range pixels
    out_of_range_left = (depth_gt < min_depth_values).float()
    out_of_range_right = (depth_gt > max_depth_values).float()
    out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
    in_range_mask = 1 - out_of_range_mask
    
    # Final mask
    final_mask = in_range_mask.squeeze(1) * mask  # (B, H, W)
    
    # Find GT bin index
    gt_index_volume = (depth_values_right <= depth_gt_volume).float().sum(dim=1, keepdim=True).long()
    gt_index_volume = torch.clamp_max(gt_index_volume, max=D - 1).squeeze(1)  # (B, H, W)
    
    # Apply mask and compute loss
    final_mask = final_mask.bool()
    gt_index = gt_index_volume[final_mask]  # (N,)
    prob_pre = prob_volume_pre.permute(0, 2, 3, 1)[final_mask, :]  # (N, D)
    
    if gt_index.numel() == 0:
        return torch.tensor(0.0, device=depth_gt.device, requires_grad=True)
    
    loss = F.cross_entropy(prob_pre, gt_index, reduction='mean')
    return loss


# ============================================================
# Multi-Stage Loss
# ============================================================

class MultiStageLoss(nn.Module):
    """
    Multi-stage depth loss for MVSFormer++.
    
    Computes weighted loss across all stages.
    """
    
    def __init__(self, cfg: LossCfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        depth_gt_ms: Dict[str, torch.Tensor],
        mask_ms: Dict[str, torch.Tensor],
        depth_interval: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-stage loss.
        
        Args:
            outputs: Dict with stage1-4 outputs
                Each contains: depth, depth_values, prob_volume_pre
            depth_gt_ms: Dict with stage1-4 ground truth depths
            mask_ms: Dict with stage1-4 valid masks
            depth_interval: (B,) depth interval for normalization
        
        Returns:
            Dict with:
                - stage1, stage2, stage3, stage4: per-stage losses
                - total: weighted sum of all losses
        """
        loss_dict = {}
        total_loss = 0.0
        
        stage_keys = ['stage1', 'stage2', 'stage3', 'stage4']
        
        for i, stage_key in enumerate(stage_keys):
            if stage_key not in outputs:
                continue
            
            stage_out = outputs[stage_key]
            depth_type = self.cfg.depth_types[i]
            weight = self.cfg.loss_weights[i]
            
            depth_pred = stage_out['depth']
            depth_values = stage_out['depth_values']
            depth_gt = depth_gt_ms[stage_key]
            mask = mask_ms[stage_key]
            
            if depth_type == 'ce':
                prob_volume_pre = stage_out['prob_volume_pre'].float()
                loss = cross_entropy_loss(
                    prob_volume_pre,
                    depth_values,
                    depth_gt,
                    mask,
                    inverse_depth=self.cfg.inverse_depth,
                )
            elif depth_type == 'reg':
                loss = regression_loss(
                    depth_pred,
                    depth_gt,
                    mask,
                    depth_interval=depth_interval,
                    depth_values=depth_values,
                    clip_func=self.cfg.clip_func,
                    inverse_depth=self.cfg.inverse_depth,
                )
            else:
                raise ValueError(f"Unknown depth type: {depth_type}")
            
            loss_dict[stage_key] = weight * loss
            total_loss = total_loss + weight * loss
        
        loss_dict['total'] = total_loss
        return loss_dict


# ============================================================
# Simple Loss (for quick testing)
# ============================================================

def simple_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Simple smooth L1 loss without normalization.
    
    Args:
        depth_pred: (B, H, W) predicted depth
        depth_gt: (B, H, W) ground truth depth
        mask: (B, H, W) valid mask
    
    Returns:
        loss: scalar loss value
    """
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], reduction='mean')


# ============================================================
# Utility: Create multi-scale GT
# ============================================================

def create_multiscale_gt(
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    scales: Tuple[int, ...] = (8, 4, 2, 1),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create multi-scale ground truth depth and mask.
    
    Args:
        depth_gt: (B, H, W) full resolution depth
        mask: (B, H, W) full resolution mask
        scales: downsampling scales for each stage
    
    Returns:
        depth_gt_ms: Dict with stage1-4 depths
        mask_ms: Dict with stage1-4 masks
    """
    B, H, W = depth_gt.shape
    depth_gt_ms = {}
    mask_ms = {}
    
    for i, scale in enumerate(scales):
        stage_key = f'stage{i + 1}'
        h, w = H // scale, W // scale
        
        if scale == 1:
            depth_gt_ms[stage_key] = depth_gt
            mask_ms[stage_key] = mask
        else:
            # Downsample with nearest for mask, bilinear for depth
            depth_gt_ms[stage_key] = F.interpolate(
                depth_gt.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(1)
            mask_ms[stage_key] = F.interpolate(
                mask.float().unsqueeze(1), size=(h, w), mode='nearest'
            ).squeeze(1)
    
    return depth_gt_ms, mask_ms


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, H, W = 2, 60, 80
    D = 32
    
    # Create test data
    depth_pred = torch.rand(B, H, W, device=device) * 10 + 0.5
    depth_gt = torch.rand(B, H, W, device=device) * 10 + 0.5
    mask = torch.ones(B, H, W, device=device)
    depth_interval = torch.tensor([0.1, 0.1], device=device)
    
    # Test regression loss with fix
    print("\n=== Testing regression_loss with depth_interval normalization ===")
    reg_loss = regression_loss(depth_pred, depth_gt, mask, depth_interval)
    print(f"Regression loss: {reg_loss.item():.6f}")
    
    # Test with partial mask
    print("\n=== Testing with partial mask ===")
    partial_mask = torch.zeros(B, H, W, device=device)
    partial_mask[:, :H//2, :W//2] = 1.0
    reg_loss_partial = regression_loss(depth_pred, depth_gt, partial_mask, depth_interval)
    print(f"Regression loss (partial mask): {reg_loss_partial.item():.6f}")
    
    # Test cross-entropy loss
    prob_volume_pre = torch.randn(B, D, H, W, device=device)
    depth_values = torch.linspace(0.5, 10.5, D, device=device).view(1, D, 1, 1).expand(B, D, H, W)
    
    ce_loss = cross_entropy_loss(prob_volume_pre, depth_values, depth_gt, mask)
    print(f"\nCross-entropy loss: {ce_loss.item():.6f}")
    
    # Test multi-stage loss
    print("\n=== Testing MultiStageLoss ===")
    
    cfg = LossCfg()
    loss_fn = MultiStageLoss(cfg).to(device)
    
    # Create mock outputs
    outputs = {}
    for i, stage_key in enumerate(['stage1', 'stage2', 'stage3', 'stage4']):
        scale = 8 // (2 ** i)
        h, w = H * scale // 8, W * scale // 8
        d = 32 // (2 ** i)
        
        outputs[stage_key] = {
            'depth': torch.rand(B, h, w, device=device) * 10 + 0.5,
            'depth_values': torch.linspace(0.5, 10.5, d, device=device).view(1, d, 1, 1).expand(B, d, h, w),
            'prob_volume_pre': torch.randn(B, d, h, w, device=device),
        }
    
    # Create multi-scale GT
    depth_gt_full = torch.rand(B, H * 8, W * 8, device=device) * 10 + 0.5
    mask_full = torch.ones(B, H * 8, W * 8, device=device)
    depth_gt_ms, mask_ms = create_multiscale_gt(depth_gt_full, mask_full)
    
    # Compute loss
    loss_dict = loss_fn(outputs, depth_gt_ms, mask_ms, depth_interval)
    
    print("Loss dict:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    print("\n✅ All tests passed!")
