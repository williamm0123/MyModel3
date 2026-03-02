# models/network/warping.py
"""
Homography Warping for Multi-View Stereo.

Based on official implementation in warping.py.

Core operation: Warp source features to reference view at multiple depth hypotheses.
"""
from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def homo_warping_3d(
    src_feat: torch.Tensor,
    src_proj: torch.Tensor,
    ref_proj: torch.Tensor,
    depth_values: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable homography warping.
    
    Warps source features to reference view at multiple depth hypotheses.
    
    Args:
        src_feat: (B, C, H, W) source view features
        src_proj: (B, 4, 4) source projection matrix (K @ [R|t])
        ref_proj: (B, 4, 4) reference projection matrix
        depth_values: (B, D) or (B, D, H, W) depth hypotheses
    
    Returns:
        warped_feat: (B, C, D, H, W) warped source features
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    # Matrix inverse/pinv is not supported in fp16; keep geometry in fp32.
    src_proj_f = src_proj.float()
    ref_proj_f = ref_proj.float()
    depth_values_f = depth_values.float()
    
    with torch.no_grad():
        # Compute relative projection: src_proj @ inv(ref_proj)
        try:
            ref_proj_inv = torch.inverse(ref_proj_f)
        except RuntimeError:
            ref_proj_inv = torch.linalg.pinv(ref_proj_f)
        
        proj = torch.matmul(src_proj_f, ref_proj_inv)  # (B, 4, 4)
        rot = proj[:, :3, :3]   # (B, 3, 3)
        trans = proj[:, :3, 3:4]  # (B, 3, 1)
        
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=device),
            torch.arange(0, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        y, x = y.contiguous().view(-1), x.contiguous().view(-1)
        
        # Homogeneous pixel coordinates: (3, H*W)
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=0)
        xyz = xyz.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)
        
        # Rotate pixel coordinates
        rot_xyz = torch.matmul(rot, xyz)  # (B, 3, H*W)
        
        # Broadcast depth values
        if depth_values_f.dim() == 2:
            # (B, D) -> (B, 1, D, H*W)
            depth_values_expanded = depth_values_f.view(B, 1, D, 1).expand(-1, -1, -1, H * W)
        else:
            # (B, D, H, W) -> (B, 1, D, H*W)
            depth_values_expanded = depth_values_f.view(B, 1, D, H * W)
        
        # Scale by depth: (B, 3, D, H*W)
        rot_xyz_expanded = rot_xyz.unsqueeze(2).expand(-1, -1, D, -1)
        rot_depth_xyz = rot_xyz_expanded * depth_values_expanded
        
        # Add translation: (B, 3, D, H*W)
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)
        
        # Project to 2D: (B, 2, D, H*W)
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)
        
        # Normalize to [-1, 1] for grid_sample
        proj_x_norm = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        proj_y_norm = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        
        # Grid: (B, D, H*W, 2)
        grid = torch.stack([proj_x_norm, proj_y_norm], dim=3)
    
    # Warp source features
    # grid_sample expects (B, C, H_in, W_in) and grid (B, H_out, W_out, 2)
    # We reshape to (B, C, D*H, W) and grid to (B, D*H, W, 2)
    warped_feat = F.grid_sample(
        src_feat,
        grid.view(B, D * H, W, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # Reshape to (B, C, D, H, W)
    warped_feat = warped_feat.view(B, C, D, H, W)
    
    return warped_feat


def homo_warping_3d_with_mask(
    src_feat: torch.Tensor,
    src_proj: torch.Tensor,
    ref_proj: torch.Tensor,
    depth_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Homography warping with validity mask.
    
    Args:
        src_feat: (B, C, H, W) source view features
        src_proj: (B, 4, 4) source projection matrix
        ref_proj: (B, 4, 4) reference projection matrix
        depth_values: (B, D) or (B, D, H, W) depth hypotheses
    
    Returns:
        warped_feat: (B, C, D, H, W) warped source features
        mask: (B, D, H, W) validity mask (True = invalid)
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    # Matrix inverse/pinv is not supported in fp16; keep geometry in fp32.
    src_proj_f = src_proj.float()
    ref_proj_f = ref_proj.float()
    depth_values_f = depth_values.float()
    
    with torch.no_grad():
        # Compute relative projection: src_proj @ inv(ref_proj)
        # Use pseudo-inverse for numerical stability
        try:
            ref_proj_inv = torch.inverse(ref_proj_f)
        except RuntimeError:
            ref_proj_inv = torch.linalg.pinv(ref_proj_f)
        
        proj = torch.matmul(src_proj_f, ref_proj_inv)  # (B, 4, 4)
        rot = proj[:, :3, :3]   # (B, 3, 3)
        trans = proj[:, :3, 3:4]  # (B, 3, 1)
        
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=device),
            torch.arange(0, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        y, x = y.contiguous().view(-1), x.contiguous().view(-1)
        
        # Homogeneous coordinates
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=0)
        xyz = xyz.unsqueeze(0).repeat(B, 1, 1)
        
        # Rotate
        rot_xyz = torch.matmul(rot, xyz)
        
        # Broadcast depth
        if depth_values_f.dim() == 2:
            depth_values_expanded = depth_values_f.view(B, 1, D, 1).expand(-1, -1, -1, H * W)
        else:
            depth_values_expanded = depth_values_f.view(B, 1, D, H * W)
        
        # Scale by depth
        rot_xyz_expanded = rot_xyz.unsqueeze(2).expand(-1, -1, D, -1)
        rot_depth_xyz = rot_xyz_expanded * depth_values_expanded
        
        # Add translation
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)
        
        # Project to 2D
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)
        
        # Normalize
        proj_x_norm = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        proj_y_norm = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        
        # Create validity mask
        x_invalid = (proj_x_norm > 1) | (proj_x_norm < -1)
        y_invalid = (proj_y_norm > 1) | (proj_y_norm < -1)
        z_invalid = proj_xyz[:, 2:3, :, :].squeeze(1) <= 0
        
        mask = (x_invalid | y_invalid | z_invalid).view(B, D, H, W)
        
        # Grid
        grid = torch.stack([proj_x_norm, proj_y_norm], dim=3)
    
    # Warp
    warped_feat = F.grid_sample(
        src_feat,
        grid.view(B, D * H, W, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    warped_feat = warped_feat.view(B, C, D, H, W)
    
    return warped_feat, mask


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing warping module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, C, H, W = 2, 64, 60, 80
    D = 32
    
    # Create test data
    src_feat = torch.randn(B, C, H, W, device=device)
    
    # Identity projection (no transformation)
    eye = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    src_proj = eye.clone()
    ref_proj = eye.clone()
    
    # Depth hypotheses
    depth_values = torch.linspace(0.5, 10.0, D, device=device).unsqueeze(0).repeat(B, 1)
    
    # Test without mask
    warped = homo_warping_3d(src_feat, src_proj, ref_proj, depth_values)
    print(f"Input: src_feat {src_feat.shape}")
    print(f"Output: warped {warped.shape}")
    
    # Test with mask
    warped_masked, mask = homo_warping_3d_with_mask(src_feat, src_proj, ref_proj, depth_values)
    print(f"Output: warped_masked {warped_masked.shape}, mask {mask.shape}")
    print(f"Invalid ratio: {mask.float().mean():.4f}")
    
    # Test with per-pixel depth
    depth_values_hw = torch.linspace(0.5, 10.0, D, device=device).view(1, D, 1, 1).expand(B, D, H, W)
    warped_hw = homo_warping_3d(src_feat, src_proj, ref_proj, depth_values_hw)
    print(f"Output with per-pixel depth: {warped_hw.shape}")
    
    print("\nAll tests passed!")
