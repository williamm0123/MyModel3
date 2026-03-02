# models/network/cost_volume.py
"""
Cost Volume Construction and Depth Estimation for MVSFormer++.

Based on official implementation in cost_volume.py and module.py.

Pipeline:
    1. Homography warping: Warp source features to reference view
    2. Cost volume: Group-wise correlation between ref and warped src
    3. Visibility weighting: Entropy-based weights for occlusion handling
    4. Cost aggregation: Weighted sum of cost volumes
    5. Cost regularization: 3D CNN or Transformer
    6. Depth regression: Softmax + weighted sum
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.warping import homo_warping_3d_with_mask


# ============================================================
# Basic 3D Convolution Blocks
# ============================================================

class Conv3d(nn.Module):
    """3D Convolution + BatchNorm + ReLU"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        relu: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv3d(nn.Module):
    """3D Transposed Convolution + BatchNorm + ReLU"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        relu: bool = True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size, stride, padding,
            output_padding=output_padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ConvBnReLU(nn.Module):
    """2D Conv + BatchNorm + ReLU"""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ============================================================
# Cost Regularization Networks
# ============================================================

class CostRegNet(nn.Module):
    """
    3D U-Net for cost volume regularization.
    
    Downsamples in all 3 dimensions (D, H, W).
    Used when ndepths > threshold (e.g., 8).
    """
    
    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        
        # Encoder
        self.conv1 = Conv3d(in_ch, base_ch * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_ch * 2, base_ch * 2, padding=1)
        
        self.conv3 = Conv3d(base_ch * 2, base_ch * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_ch * 4, base_ch * 4, padding=1)
        
        self.conv5 = Conv3d(base_ch * 4, base_ch * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_ch * 8, base_ch * 8, padding=1)
        
        # Decoder with skip connections
        self.conv7 = Deconv3d(base_ch * 8, base_ch * 4, stride=2, padding=1, output_padding=1)
        self.conv9 = Deconv3d(base_ch * 4, base_ch * 2, stride=2, padding=1, output_padding=1)
        self.conv11 = Deconv3d(base_ch * 2, base_ch * 1, stride=2, padding=1, output_padding=1)
        
        # Skip connection projection
        if in_ch != base_ch:
            self.inner = nn.Conv3d(in_ch, base_ch, 1, 1)
        else:
            self.inner = nn.Identity()
        
        # Output
        self.prob = nn.Conv3d(base_ch, 1, 3, stride=1, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor, position3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        
        # Decoder with size alignment
        x = self.conv7(x)
        if x.shape != conv4.shape:
            x = F.interpolate(x, size=conv4.shape[2:], mode='trilinear', align_corners=False)
        x = conv4 + x
        
        x = self.conv9(x)
        if x.shape != conv2.shape:
            x = F.interpolate(x, size=conv2.shape[2:], mode='trilinear', align_corners=False)
        x = conv2 + x
        
        x = self.conv11(x)
        conv0_proj = self.inner(conv0)
        if x.shape != conv0_proj.shape:
            x = F.interpolate(x, size=conv0_proj.shape[2:], mode='trilinear', align_corners=False)
        x = conv0_proj + x
        
        x = self.prob(x)
        return x


class CostRegNet3D(nn.Module):
    """
    3D U-Net with spatial-only downsampling.
    
    Downsamples only in (H, W), preserves D dimension.
    Used when ndepths <= threshold (e.g., 8).
    """
    
    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        
        # Encoder: stride=(1,2,2) for spatial-only downsampling
        self.conv1 = Conv3d(in_ch, base_ch * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(base_ch * 2, base_ch * 2, padding=1)
        
        self.conv3 = Conv3d(base_ch * 2, base_ch * 4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(base_ch * 4, base_ch * 4, padding=1)
        
        self.conv5 = Conv3d(base_ch * 4, base_ch * 8, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(base_ch * 8, base_ch * 8, padding=1)
        
        # Decoder
        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=3, stride=(1, 2, 2),
                              padding=1, output_padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=3, stride=(1, 2, 2),
                              padding=1, output_padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=3, stride=(1, 2, 2),
                              padding=1, output_padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        
        # Skip connection
        if in_ch != base_ch:
            self.inner = nn.Conv3d(in_ch, base_ch, 1, 1)
        else:
            self.inner = nn.Identity()
        
        # Output
        self.prob = nn.Conv3d(base_ch, 1, 1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor, position3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        
        # Decoder with size alignment
        x = self.conv7(x)
        if x.shape != conv4.shape:
            x = F.interpolate(x, size=conv4.shape[2:], mode='trilinear', align_corners=False)
        x = conv4 + x
        
        x = self.conv9(x)
        if x.shape != conv2.shape:
            x = F.interpolate(x, size=conv2.shape[2:], mode='trilinear', align_corners=False)
        x = conv2 + x
        
        x = self.conv11(x)
        conv0_proj = self.inner(conv0)
        if x.shape != conv0_proj.shape:
            x = F.interpolate(x, size=conv0_proj.shape[2:], mode='trilinear', align_corners=False)
        x = conv0_proj + x
        
        x = self.prob(x)
        return x


# ============================================================
# Transformer Cost Regularization (for stage1)
# ============================================================

class FPE(nn.Module):
    """
    Feature Positional Encoding for cost volumes.

    Adds a learnable depth-axis positional bias to (B, C, D, H, W).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, _, _ = x.shape
        depth_pos = torch.linspace(-1.0, 1.0, D, device=x.device, dtype=x.dtype).view(1, 1, D, 1, 1)
        bias = self.scale.view(1, -1, 1, 1, 1) * depth_pos
        return x + bias


class CostRegTransformerDepth(nn.Module):
    """
    Cost-volume transformer regularizer along depth tokens per pixel.

    Input: (B, C, D, H, W)
    Output: (B, 1, D, H, W)
    """

    def __init__(self, channels: int, nheads: int = 8, num_layers: int = 2, use_fpe: bool = True):
        super().__init__()
        self.channels = channels
        self.use_fpe = use_fpe
        self.fpe = FPE(channels) if use_fpe else nn.Identity()

        if channels % nheads != 0:
            nheads = 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nheads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)
        self.prob = nn.Conv3d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor, position3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.fpe(x)
        B, C, D, H, W = x.shape

        seq = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        seq = self.encoder(seq)
        seq = self.norm(seq)

        x = seq.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
        x = self.prob(x)
        return x


# ============================================================
# Visibility Network (Entropy-based)
# ============================================================

class VisibilityNet(nn.Module):
    """
    Visibility-aware weighting based on cost volume entropy.
    
    Low entropy = consistent matching = high visibility
    High entropy = occlusion/noise = low visibility
    """
    
    def __init__(self, in_ch: int = None):
        super().__init__()
        # Simple 2D conv to predict visibility
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, cost_vol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_vol: (B, G, D, H, W) cost volume
        
        Returns:
            vis_weight: (B, 1, H, W) visibility weights
        """
        # Compute entropy along depth dimension
        prob = F.softmax(cost_vol.mean(dim=1, keepdim=True), dim=2)  # (B, 1, D, H, W)
        entropy = -(prob * torch.log(prob + 1e-6)).sum(dim=2)  # (B, 1, H, W)
        
        # Normalize entropy
        entropy_norm = entropy / (torch.log(torch.tensor(cost_vol.shape[2], dtype=torch.float32, device=cost_vol.device)) + 1e-6)
        
        # Predict visibility (low entropy = high visibility)
        vis_weight = self.conv(1.0 - entropy_norm)
        
        return vis_weight


# ============================================================
# Depth Regression
# ============================================================

def depth_regression(prob_volume: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """
    Weighted sum of depth hypotheses.
    
    Args:
        prob_volume: (B, D, H, W) probability volume (after softmax)
        depth_values: (B, D, H, W) or (B, D) depth hypotheses
    
    Returns:
        depth: (B, H, W) estimated depth
    """
    if depth_values.dim() == 2:
        # (B, D) -> (B, D, 1, 1)
        depth_values = depth_values.unsqueeze(-1).unsqueeze(-1)
    
    # Weighted sum
    depth = (prob_volume * depth_values).sum(dim=1)
    
    return depth


def conf_regression(prob_volume: torch.Tensor, n: int = 4) -> torch.Tensor:
    """
    Confidence estimation based on probability volume.
    
    Sum of top-n probabilities centered around the peak.
    
    Args:
        prob_volume: (B, D, H, W)
        n: number of bins around peak to sum
    
    Returns:
        confidence: (B, H, W)
    """
    B, D, H, W = prob_volume.shape
    
    # Find peak
    _, idx = torch.max(prob_volume, dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Sum probabilities in [idx-n, idx+n]
    # Create indices
    offsets = torch.arange(-n, n + 1, device=prob_volume.device)
    indices = idx + offsets.view(1, -1, 1, 1)  # (B, 2n+1, H, W)
    
    # Clamp to valid range
    indices = torch.clamp(indices, 0, D - 1)
    
    # Gather and sum
    confidence = torch.gather(prob_volume, 1, indices).sum(dim=1)  # (B, H, W)
    
    return confidence


# ============================================================
# Depth Hypotheses Initialization
# ============================================================

def init_range(depth_range: torch.Tensor, ndepths: int, H: int, W: int) -> torch.Tensor:
    """
    Initialize uniform depth hypotheses.
    
    Args:
        depth_range: (B, 2) [depth_min, depth_max]
        ndepths: number of depth hypotheses
        H, W: spatial dimensions
    
    Returns:
        depth_values: (B, D, H, W) uniform depth hypotheses
    """
    B = depth_range.shape[0]
    device = depth_range.device
    dtype = depth_range.dtype
    
    depth_min = depth_range[:, 0].view(B, 1, 1, 1)
    depth_max = depth_range[:, 1].view(B, 1, 1, 1)
    
    # Linear interpolation
    t = torch.linspace(0, 1, ndepths, device=device, dtype=dtype).view(1, -1, 1, 1)
    depth_values = depth_min + t * (depth_max - depth_min)
    
    # Expand to spatial dimensions
    depth_values = depth_values.expand(B, ndepths, H, W)
    
    return depth_values


def init_inverse_range(depth_range: torch.Tensor, ndepths: int, H: int, W: int) -> torch.Tensor:
    """
    Initialize inverse depth hypotheses (uniform in disparity space).
    
    Args:
        depth_range: (B, 2) [depth_min, depth_max]
        ndepths: number of depth hypotheses
        H, W: spatial dimensions
    
    Returns:
        depth_values: (B, D, H, W) inverse depth hypotheses
    """
    B = depth_range.shape[0]
    device = depth_range.device
    dtype = depth_range.dtype
    
    depth_min = depth_range[:, 0].view(B, 1, 1, 1)
    depth_max = depth_range[:, 1].view(B, 1, 1, 1)
    
    # Inverse depth (disparity)
    inv_min = 1.0 / depth_max
    inv_max = 1.0 / depth_min
    
    # Linear in inverse space
    t = torch.linspace(0, 1, ndepths, device=device, dtype=dtype).view(1, -1, 1, 1)
    inv_depth = inv_min + t * (inv_max - inv_min)
    
    depth_values = 1.0 / inv_depth
    depth_values = depth_values.expand(B, ndepths, H, W)
    
    return depth_values


def schedule_inverse_range(
    depth: torch.Tensor,
    prev_depth_values: torch.Tensor,
    ndepths: int,
    split_itv: float,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Schedule depth hypotheses for cascade refinement.
    
    Narrow the depth range around previous prediction.
    
    Args:
        depth: (B, H, W) previous depth prediction
        prev_depth_values: (B, D_prev, H, W) previous depth hypotheses
        ndepths: number of new depth hypotheses
        split_itv: interval ratio for narrowing
        H, W: target spatial dimensions
    
    Returns:
        depth_values: (B, D, H, W) new depth hypotheses
    """
    B = depth.shape[0]
    device = depth.device
    dtype = depth.dtype
    
    # Compute depth interval from previous stage
    last_depth_itv = 1.0 / prev_depth_values[:, 2] - 1.0 / prev_depth_values[:, 1]
    
    # New range centered at predicted depth
    inv_min_depth = 1.0 / depth + split_itv * last_depth_itv
    inv_max_depth = 1.0 / depth - split_itv * last_depth_itv
    
    # Prevent negative depth
    inv_max_depth = torch.clamp(inv_max_depth, min=0.002)
    
    # Interpolate
    H_in, W_in = depth.shape[1], depth.shape[2]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype).view(1, -1, 1, 1) / (ndepths - 1)
    itv = itv.expand(B, ndepths, H_in // 2, W_in // 2)
    
    inv_depth_hypo = inv_max_depth.unsqueeze(1)[:, :, ::2, ::2] + \
                     (inv_min_depth - inv_max_depth).unsqueeze(1)[:, :, ::2, ::2] * itv
    
    # Upsample to target resolution
    inv_depth_hypo = F.interpolate(
        inv_depth_hypo.unsqueeze(1),
        size=[ndepths, H, W],
        mode='trilinear',
        align_corners=True
    ).squeeze(1)
    
    return 1.0 / inv_depth_hypo


# ============================================================
# Stage Network
# ============================================================

@dataclass
class StageCfg:
    """Configuration for StageNet."""
    ndepths: int = 32
    base_ch: int = 8
    depth_type: str = "regression"  # "ce" or "regression"
    cost_reg_type: str = "Normal"   # "Normal" or "PureTransformerCostReg"
    transformer_heads: int = 8
    transformer_layers: int = 2
    use_fpe: bool = True


class StageNet(nn.Module):
    """
    Single stage depth estimation.
    
    Pipeline:
        1. Warp source features to reference
        2. Build cost volume (group-wise correlation)
        3. Visibility weighting
        4. Cost regularization (3D CNN)
        5. Depth regression
    """
    
    def __init__(self, cfg: StageCfg):
        super().__init__()
        self.cfg = cfg
        
        # Visibility network
        self.vis_net = VisibilityNet()
        
        # Cost regularization
        if cfg.cost_reg_type == "PureTransformerCostReg":
            self.cost_reg = CostRegTransformerDepth(
                channels=cfg.base_ch,
                nheads=cfg.transformer_heads,
                num_layers=cfg.transformer_layers,
                use_fpe=cfg.use_fpe,
            )
        else:
            # Use CostRegNet3D for small ndepths, CostRegNet for larger
            if cfg.ndepths <= 8:
                self.cost_reg = CostRegNet3D(cfg.base_ch, cfg.base_ch)
            else:
                self.cost_reg = CostRegNet(cfg.base_ch, cfg.base_ch)

    def _build_fused_cost_volume(
        self,
        features: torch.Tensor,
        proj_matrices: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        B, V, C, H, W = features.shape
        D = depth_values.shape[1]

        ref_feat = features[:, 0]  # (B, C, H, W)
        src_feats = features[:, 1:]  # (B, V-1, C, H, W)

        # Build full projection matrices P = K @ E.
        if proj_matrices.dim() == 5 and proj_matrices.shape[2] == 2:
            extr = proj_matrices[:, :, 0]  # (B,V,4,4)
            intr = proj_matrices[:, :, 1]  # (B,V,4,4)
            # Geometry ops must run in fp32 (inverse/pinv in warping do not support fp16).
            proj_full = torch.matmul(intr.float(), extr.float())
        elif proj_matrices.dim() == 4 and proj_matrices.shape[-2:] == (4, 4):
            proj_full = proj_matrices.float()
        else:
            raise ValueError(f"Unsupported proj_matrices shape: {tuple(proj_matrices.shape)}")

        ref_proj = proj_full[:, 0]   # (B,4,4)
        src_projs = proj_full[:, 1:] # (B,V-1,4,4)

        volume_sum = 0.0
        vis_sum = 0.0
        G = self.cfg.base_ch

        for i in range(V - 1):
            src_feat = src_feats[:, i].float()
            src_proj_mat = src_projs[:, i]
            ref_proj_mat = ref_proj

            warped_feat, _ = homo_warping_3d_with_mask(
                src_feat, src_proj_mat, ref_proj_mat, depth_values
            )

            if G < C:
                warped_feat = warped_feat.view(B, G, C // G, D, H, W)
                ref_volume = ref_feat.view(B, G, C // G, 1, H, W).expand(-1, -1, -1, D, -1, -1).float()
                cost_vol = (ref_volume * warped_feat).mean(dim=2)  # (B, G, D, H, W)
            else:
                ref_volume = ref_feat.view(B, G, 1, H, W).float()
                cost_vol = ref_volume * warped_feat  # (B, C, D, H, W)

            vis_weight = self.vis_net(cost_vol)  # (B, 1, H, W)
            volume_sum = volume_sum + cost_vol * vis_weight.unsqueeze(2)
            vis_sum = vis_sum + vis_weight

        return volume_sum / (vis_sum.unsqueeze(2) + 1e-6)
    
    def forward(
        self,
        features: torch.Tensor,
        proj_matrices: torch.Tensor,
        depth_values: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: (B, V, C, H, W) multi-view features
            proj_matrices:
                - Dataset format: (B, V, 2, 4, 4) where [0]=extrinsic(E), [1]=intrinsic(K_4x4)
                - Also supports legacy full-P: (B, V, 4, 4) where P = K @ E
            depth_values: (B, D, H, W) depth hypotheses
            temperature: softmax temperature for inference
        
        Returns:
            Dict with: depth, prob_volume, photometric_confidence, depth_values
        """
        B, V, C, H, W = features.shape
        D = depth_values.shape[1]
        cost_volume = self._build_fused_cost_volume(features, proj_matrices, depth_values)
        
        # Cost regularization
        cost_reg = self.cost_reg(cost_volume)  # (B, 1, D, H, W)
        prob_volume_pre = cost_reg.squeeze(1)  # (B, D, H, W)
        prob_volume = F.softmax(prob_volume_pre, dim=1)
        
        # Depth estimation
        if self.cfg.depth_type == "ce":
            # Cross-entropy: argmax during training, regression during inference
            if self.training:
                _, idx = torch.max(prob_volume, dim=1)
                depth = torch.gather(depth_values, 1, idx.unsqueeze(1)).squeeze(1)
            else:
                depth = depth_regression(
                    F.softmax(prob_volume_pre * temperature, dim=1),
                    depth_values
                )
            confidence = prob_volume.max(dim=1)[0]
        else:
            # Regression
            depth = depth_regression(prob_volume, depth_values)
            
            # Confidence
            if D >= 32:
                confidence = conf_regression(prob_volume, n=4)
            elif D >= 16:
                confidence = conf_regression(prob_volume, n=3)
            elif D >= 8:
                confidence = conf_regression(prob_volume, n=2)
            else:
                confidence = prob_volume.max(dim=1)[0]
        
        return {
            'depth': depth,
            'prob_volume': prob_volume,
            'prob_volume_pre': prob_volume_pre,
            'photometric_confidence': confidence.detach(),
            'depth_values': depth_values,
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing Cost Volume module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, V, C, H, W = 2, 3, 64, 60, 80
    D = 32
    
    # Create test data
    features = torch.randn(B, V, C, H, W, device=device)
    
    # Projection matrices: (B, V, 4, 4) - full projection matrices
    proj = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    proj = proj.expand(B, V, 4, 4).clone()
    
    # Depth hypotheses
    depth_range = torch.tensor([[0.5, 10.0]], device=device).expand(B, 2)
    depth_values = init_inverse_range(depth_range, D, H, W)
    
    print(f"Input features: {features.shape}")
    print(f"Projection matrices: {proj.shape}")
    print(f"Depth values: {depth_values.shape}")
    
    # Test StageNet
    cfg = StageCfg(ndepths=D, base_ch=C)
    stage = StageNet(cfg).to(device)
    
    with torch.no_grad():
        out = stage(features, proj, depth_values)
    
    print("\nOutputs:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
    
    print(f"\nDepth range: [{out['depth'].min():.3f}, {out['depth'].max():.3f}]")
    print(f"Confidence range: [{out['photometric_confidence'].min():.3f}, {out['photometric_confidence'].max():.3f}]")
    
    print("\nAll tests passed!")
