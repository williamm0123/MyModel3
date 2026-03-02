# models/network/depth_estimator.py
"""
Multi-Stage Depth Estimation for MVSFormer++.

Implements coarse-to-fine depth estimation:
    Stage 1: 1/8 scale, 32 depth hypotheses
    Stage 2: 1/4 scale, 16 depth hypotheses  
    Stage 3: 1/2 scale, 8 depth hypotheses
    Stage 4: 1/1 scale, 4 depth hypotheses

Each stage refines the depth prediction from the previous stage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.Cost_volume import (
    StageNet, StageCfg,
    init_inverse_range
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class DepthEstimatorCfg:
    """Configuration for multi-stage depth estimation."""
    # Number of depth hypotheses per stage
    ndepths: Tuple[int, ...] = (32, 16, 8, 4)
    
    # Base channels per stage (matches FPN output)
    base_chs: Tuple[int, ...] = (64, 32, 16, 8)
    
    # Depth interval ratios for refinement
    depth_interval_ratios: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    
    # Whether to use inverse depth
    inverse_depth: bool = True
    
    # Depth type: "ce" (cross-entropy) or "regression"
    depth_type: str = "regression"
    
    # Temperature for softmax during inference
    temperatures: Tuple[float, ...] = (5.0, 5.0, 5.0, 1.0)


# ============================================================
# Depth Estimator
# ============================================================

class DepthEstimator(nn.Module):
    """
    Multi-stage depth estimation module.
    
    Takes FMT-enhanced features and predicts depth at multiple scales.
    """
    
    def __init__(self, cfg: DepthEstimatorCfg):
        super().__init__()
        self.cfg = cfg
        self.num_stages = len(cfg.ndepths)
        
        # Create stage networks
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            cost_reg_type = "PureTransformerCostReg" if i == 0 else "Normal"
            stage_cfg = StageCfg(
                ndepths=cfg.ndepths[i],
                base_ch=cfg.base_chs[i],
                depth_type=cfg.depth_type,
                cost_reg_type=cost_reg_type,
            )
            self.stages.append(StageNet(stage_cfg))
        
        print(f"[DepthEstimator] {self.num_stages} stages: ndepths={cfg.ndepths}, base_chs={cfg.base_chs}")
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proj_matrices: Dict[str, torch.Tensor],
        depth_range: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Dict with stage1-4 features
                stage1: (B, V, 64, H/8, W/8)
                stage2: (B, V, 32, H/4, W/4)
                stage3: (B, V, 16, H/2, W/2)
                stage4: (B, V, 8, H, W)
            proj_matrices: Dict with stage1-4 projection matrices
                Each: (B, V, 2, 4, 4)
            depth_range: (B, 2) [depth_min, depth_max]
        
        Returns:
            Dict with:
                stage1, stage2, stage3, stage4: Per-stage outputs
                depth: Final depth prediction (B, H, W)
                photometric_confidence: Final confidence (B, H, W)
        """
        outputs = {}
        prev_output = None
        
        for stage_idx in range(self.num_stages):
            stage_key = f'stage{stage_idx + 1}'
            
            # Get features and projections for this stage
            feat = features[stage_key]  # (B, V, C, H, W)
            proj = proj_matrices[stage_key]  # (B, V, 2, 4, 4)
            
            B, V, C, H, W = feat.shape
            
            # Initialize depth hypotheses
            if stage_idx == 0:
                # First stage: uniform sampling in depth range
                if self.cfg.inverse_depth:
                    depth_values = init_inverse_range(depth_range, self.cfg.ndepths[0], H, W)
                else:
                    depth_values = self._init_linear_range(depth_range, self.cfg.ndepths[0], H, W)
            else:
                # Subsequent stages: bilinear-upsample previous depth, then refine around it
                prev_depth = prev_output['depth'].detach()  # (B, H_prev, W_prev)
                prev_depth_up = F.interpolate(
                    prev_depth.unsqueeze(1),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)
                
                if self.cfg.inverse_depth:
                    depth_values = self._schedule_inverse_range_from_upsampled(
                        prev_depth_up,
                        self.cfg.ndepths[stage_idx],
                        self.cfg.depth_interval_ratios[stage_idx],
                        depth_range,
                    )
                else:
                    depth_values = self._schedule_linear_range(
                        prev_depth_up,
                        self.cfg.ndepths[stage_idx],
                        self.cfg.depth_interval_ratios[stage_idx],
                        depth_range[:, 1] - depth_range[:, 0],
                        H, W,
                    )
            
            # Run stage network
            temperature = self.cfg.temperatures[stage_idx]
            stage_output = self.stages[stage_idx](feat, proj, depth_values, temperature)
            
            # Store output
            outputs[stage_key] = stage_output
            prev_output = stage_output
        
        # Final outputs
        outputs['depth'] = prev_output['depth']
        outputs['photometric_confidence'] = prev_output['photometric_confidence']
        
        return outputs

    def _schedule_inverse_range_from_upsampled(
        self,
        depth_up: torch.Tensor,
        ndepths: int,
        ratio: float,
        depth_range: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build inverse-depth hypotheses centered at upsampled previous depth.
        """
        B, H, W = depth_up.shape
        device, dtype = depth_up.device, depth_up.dtype

        depth_min = depth_range[:, 0].view(B, 1, 1)
        depth_max = depth_range[:, 1].view(B, 1, 1)

        inv_min_global = 1.0 / depth_max
        inv_max_global = 1.0 / depth_min

        # Base inverse-depth interval from global range.
        base_inv_itv = (inv_max_global - inv_min_global) / max(float(self.cfg.ndepths[0] - 1), 1.0)
        half_span = 0.5 * ratio * base_inv_itv * max(float(ndepths - 1), 1.0)

        center_inv = 1.0 / torch.clamp(depth_up, min=1e-6)
        inv_left = center_inv - half_span
        inv_right = center_inv + half_span

        t = torch.linspace(0.0, 1.0, ndepths, device=device, dtype=dtype).view(1, ndepths, 1, 1)
        inv_values = inv_left.unsqueeze(1) + (inv_right - inv_left).unsqueeze(1) * t
        inv_values = torch.clamp(inv_values, min=inv_min_global.unsqueeze(1), max=inv_max_global.unsqueeze(1))

        return 1.0 / torch.clamp(inv_values, min=1e-6)
    
    def _init_linear_range(
        self,
        depth_range: torch.Tensor,
        ndepths: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Initialize depth hypotheses with linear sampling."""
        B = depth_range.shape[0]
        device, dtype = depth_range.device, depth_range.dtype
        
        depth_min = depth_range[:, 0]
        depth_max = depth_range[:, 1]
        
        interval = (depth_max - depth_min) / (ndepths - 1)
        
        depth_values = depth_min.view(B, 1, 1, 1) + \
                       torch.arange(0, ndepths, device=device, dtype=dtype).view(1, -1, 1, 1) * \
                       interval.view(B, 1, 1, 1)
        
        return depth_values.expand(B, ndepths, H, W)
    
    def _schedule_linear_range(
        self,
        depth: torch.Tensor,
        ndepths: int,
        ratio: float,
        depth_interval: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Schedule depth hypotheses for coarse-to-fine refinement (linear)."""
        B = depth.shape[0]
        device, dtype = depth.device, depth.dtype
        
        # Interval for this stage
        interval = ratio * depth_interval.view(B, 1, 1)
        
        # Range centered at predicted depth
        depth_min = depth - ndepths / 2 * interval
        depth_min = torch.clamp(depth_min, min=0.001)
        depth_max = depth + ndepths / 2 * interval
        
        new_interval = (depth_max - depth_min) / (ndepths - 1)
        
        depth_values = depth_min.unsqueeze(1) + \
                       torch.arange(0, ndepths, device=device, dtype=dtype).view(1, -1, 1, 1) * \
                       new_interval.unsqueeze(1)
        
        # Interpolate to target resolution
        depth_values = F.interpolate(
            depth_values.unsqueeze(1),
            size=[ndepths, H, W],
            mode='trilinear',
            align_corners=True
        ).squeeze(1)
        
        return depth_values


# ============================================================
# Complete MVS Model
# ============================================================

class MVSModel(nn.Module):
    """
    Complete MVS depth estimation model.
    
    Combines:
        - Network (FPN + DINO + SVA + FMT)
        - DepthEstimator (multi-stage cost volume)
    """
    
    def __init__(
        self,
        network: nn.Module,
        depth_cfg: DepthEstimatorCfg,
    ):
        super().__init__()
        self.network = network
        self.depth_estimator = DepthEstimator(depth_cfg)
    
    def forward(
        self,
        images: torch.Tensor,
        proj_matrices: Dict[str, torch.Tensor],
        depth_range: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: (B, V, 3, H, W) multi-view images
            proj_matrices: Dict with stage1-4 projection matrices
            depth_range: (B, 2) [depth_min, depth_max]
        
        Returns:
            Dict with depth predictions and intermediate outputs
        """
        # Feature extraction
        features = self.network(images)
        
        # Depth estimation
        outputs = self.depth_estimator(features, proj_matrices, depth_range)
        
        return outputs


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing Depth Estimator...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, V, H, W = 2, 3, 480, 640
    
    # Create mock features
    features = {
        'stage1': torch.randn(B, V, 64, H // 8, W // 8, device=device),
        'stage2': torch.randn(B, V, 32, H // 4, W // 4, device=device),
        'stage3': torch.randn(B, V, 16, H // 2, W // 2, device=device),
        'stage4': torch.randn(B, V, 8, H, W, device=device),
    }
    
    # Create mock projection matrices
    proj_matrices = {}
    for stage_key in ['stage1', 'stage2', 'stage3', 'stage4']:
        proj = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        proj = proj.expand(B, V, 2, 4, 4).clone()
        proj_matrices[stage_key] = proj
    
    # Depth range
    depth_range = torch.tensor([[0.5, 10.0]], device=device).expand(B, 2).clone()
    
    print(f"Features:")
    for k, v in features.items():
        print(f"  {k}: {tuple(v.shape)}")
    
    # Create depth estimator
    cfg = DepthEstimatorCfg()
    estimator = DepthEstimator(cfg).to(device)
    
    print(f"\nRunning depth estimation...")
    
    with torch.no_grad():
        outputs = estimator(features, proj_matrices, depth_range)
    
    print(f"\nOutputs:")
    for k, v in outputs.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    print(f"    {kk}: {tuple(vv.shape)}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
    
    print(f"\nFinal depth: {outputs['depth'].shape}")
    print(f"Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
    print(f"Confidence range: [{outputs['photometric_confidence'].min():.3f}, {outputs['photometric_confidence'].max():.3f}]")
    
    print("\nAll tests passed!")
