# models/network/fpn.py
"""
Feature Pyramid Network (FPN) for MVSFormer++.

Based on official FPNEncoder and FPNDecoder in module.py.

Architecture:
    FPNEncoder: 3-channel input → multi-scale features
        - conv01: 1/1 scale (H, W)
        - conv11: 1/2 scale (H/2, W/2)
        - conv21: 1/4 scale (H/4, W/4)
        - conv31: 1/8 scale (H/8, W/8)
    
    FPNDecoder: Top-down pathway with lateral connections
        - feat1: 1/8 scale (stage1, for FMT)
        - feat2: 1/4 scale (stage2)
        - feat3: 1/2 scale (stage3)
        - feat4: 1/1 scale (stage4)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Blocks
# ============================================================

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + Activation"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act: str = "swish",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        
        if act == "swish":
            self.act = Swish()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ============================================================
# FPN Configuration
# ============================================================

@dataclass
class FPNCfg:
    """Configuration for FPN."""
    # Channel configuration: [stage0, stage1, stage2, stage3]
    # Default matches official: [8, 16, 32, 64]
    feat_chs: Tuple[int, ...] = (8, 16, 32, 64)
    
    # Activation type
    act: str = "swish"


# ============================================================
# FPN Encoder
# ============================================================

class FPNEncoder(nn.Module):
    """
    FPN Encoder: Extracts multi-scale features from input images.
    
    Output scales:
        conv01: 1/1 (H, W)
        conv11: 1/2 (H/2, W/2)
        conv21: 1/4 (H/4, W/4)
        conv31: 1/8 (H/8, W/8)
    """
    
    def __init__(self, cfg: FPNCfg):
        super().__init__()
        chs = cfg.feat_chs
        act = cfg.act
        
        # Stage 0: 1/1 scale
        self.conv00 = ConvBnAct(3, chs[0], 7, 1, 3, act)
        self.conv01 = ConvBnAct(chs[0], chs[0], 5, 1, 2, act)
        
        # Stage 1: 1/2 scale
        self.downsample1 = ConvBnAct(chs[0], chs[1], 5, 2, 2, act)
        self.conv10 = ConvBnAct(chs[1], chs[1], 3, 1, 1, act)
        self.conv11 = ConvBnAct(chs[1], chs[1], 3, 1, 1, act)
        
        # Stage 2: 1/4 scale
        self.downsample2 = ConvBnAct(chs[1], chs[2], 5, 2, 2, act)
        self.conv20 = ConvBnAct(chs[2], chs[2], 3, 1, 1, act)
        self.conv21 = ConvBnAct(chs[2], chs[2], 3, 1, 1, act)
        
        # Stage 3: 1/8 scale
        self.downsample3 = ConvBnAct(chs[2], chs[3], 3, 2, 1, act)
        self.conv30 = ConvBnAct(chs[3], chs[3], 3, 1, 1, act)
        self.conv31 = ConvBnAct(chs[3], chs[3], 3, 1, 1, act)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            List of [conv01, conv11, conv21, conv31]
        """
        # Stage 0
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        
        # Stage 1
        down1 = self.downsample1(conv01)
        conv10 = self.conv10(down1)
        conv11 = self.conv11(conv10)
        
        # Stage 2
        down2 = self.downsample2(conv11)
        conv20 = self.conv20(down2)
        conv21 = self.conv21(conv20)
        
        # Stage 3
        down3 = self.downsample3(conv21)
        conv30 = self.conv30(down3)
        conv31 = self.conv31(conv30)
        
        return [conv01, conv11, conv21, conv31]


# ============================================================
# FPN Decoder
# ============================================================

class FPNDecoder(nn.Module):
    """
    FPN Decoder: Top-down pathway with lateral connections.
    
    Takes encoder outputs and produces multi-scale features for MVS.
    
    Output:
        feat1: 1/8 scale (for FMT stage1)
        feat2: 1/4 scale (for FMT stage2)
        feat3: 1/2 scale (for FMT stage3)
        feat4: 1/1 scale (for FMT stage4)
    """
    
    def __init__(self, cfg: FPNCfg):
        super().__init__()
        chs = cfg.feat_chs
        final_ch = chs[-1]  # 64
        act = cfg.act
        
        # Output projection for stage3 (1/8)
        self.out0 = nn.Sequential(
            nn.Conv2d(final_ch, chs[3], 1),
            nn.BatchNorm2d(chs[3]),
            Swish() if act == "swish" else nn.ReLU(inplace=True),
        )
        
        # Lateral + output for stage2 (1/4)
        self.inner1 = nn.Conv2d(chs[2], final_ch, 1)
        self.out1 = nn.Sequential(
            nn.Conv2d(final_ch, chs[2], 3, padding=1),
            nn.BatchNorm2d(chs[2]),
            Swish() if act == "swish" else nn.ReLU(inplace=True),
        )
        
        # Lateral + output for stage1 (1/2)
        self.inner2 = nn.Conv2d(chs[1], final_ch, 1)
        self.out2 = nn.Sequential(
            nn.Conv2d(final_ch, chs[1], 3, padding=1),
            nn.BatchNorm2d(chs[1]),
            Swish() if act == "swish" else nn.ReLU(inplace=True),
        )
        
        # Lateral + output for stage0 (1/1)
        self.inner3 = nn.Conv2d(chs[0], final_ch, 1)
        self.out3 = nn.Sequential(
            nn.Conv2d(final_ch, chs[0], 3, padding=1),
            nn.BatchNorm2d(chs[0]),
            Swish() if act == "swish" else nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        conv01: torch.Tensor,
        conv11: torch.Tensor,
        conv21: torch.Tensor,
        conv31: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            conv01: (B, C0, H, W) encoder output at 1/1
            conv11: (B, C1, H/2, W/2) encoder output at 1/2
            conv21: (B, C2, H/4, W/4) encoder output at 1/4
            conv31: (B, C3, H/8, W/8) encoder output at 1/8
        Returns:
            List of [feat1, feat2, feat3, feat4] at scales 1/8, 1/4, 1/2, 1/1
        """
        # Start from deepest (1/8)
        intra_feat = conv31
        out0 = self.out0(intra_feat)  # feat1: 1/8
        
        # Upsample and add lateral (1/4)
        intra_feat = F.interpolate(
            intra_feat.float(), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner1(conv21)
        out1 = self.out1(intra_feat)  # feat2: 1/4
        
        # Upsample and add lateral (1/2)
        intra_feat = F.interpolate(
            intra_feat.float(), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner2(conv11)
        out2 = self.out2(intra_feat)  # feat3: 1/2
        
        # Upsample and add lateral (1/1)
        intra_feat = F.interpolate(
            intra_feat.float(), scale_factor=2, mode="bilinear", align_corners=True
        ) + self.inner3(conv01)
        out3 = self.out3(intra_feat)  # feat4: 1/1
        
        return [out0, out1, out2, out3]


# ============================================================
# Combined FPN Module
# ============================================================

class FPN(nn.Module):
    """
    Complete FPN with Encoder and Decoder.
    
    Input: (B, 3, H, W) or (B, V, 3, H, W)
    Output: Dict with stage1-4 features
    """
    
    def __init__(self, cfg: FPNCfg = None):
        super().__init__()
        if cfg is None:
            cfg = FPNCfg()
        self.cfg = cfg
        
        self.encoder = FPNEncoder(cfg)
        self.decoder = FPNDecoder(cfg)
    
    @staticmethod
    def flatten_bv(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Reshape (B, V, 3, H, W) -> (B*V, 3, H, W)"""
        if x.ndim == 5:
            B, V, C, H, W = x.shape
            return x.view(B * V, C, H, W), B, V
        elif x.ndim == 4:
            return x, -1, -1
        else:
            raise ValueError(f"FPN expects 4D/5D tensor, got {tuple(x.shape)}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_encoder_feats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) or (B*V, 3, H, W) input images
            return_encoder_feats: if True, also return encoder features (conv01-31)
        Returns:
            Dict with:
                - stage1: (B, C3, H/8, W/8) for FMT
                - stage2: (B, C2, H/4, W/4)
                - stage3: (B, C1, H/2, W/2)
                - stage4: (B, C0, H, W)
                - (optional) conv01, conv11, conv21, conv31
        """
        x, _, _ = self.flatten_bv(x)
        
        # Encoder
        conv01, conv11, conv21, conv31 = self.encoder(x)
        
        # Decoder
        feat1, feat2, feat3, feat4 = self.decoder(conv01, conv11, conv21, conv31)
        
        out = {
            "stage1": feat1,  # 1/8, for FMT
            "stage2": feat2,  # 1/4
            "stage3": feat3,  # 1/2
            "stage4": feat4,  # 1/1
        }
        
        if return_encoder_feats:
            out.update({
                "conv01": conv01,
                "conv11": conv11,
                "conv21": conv21,
                "conv31": conv31,
            })
        
        return out
    
    def forward_encoder_only(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Only run encoder, return [conv01, conv11, conv21, conv31]."""
        x, _, _ = self.flatten_bv(x)
        return self.encoder(x)
    
    def forward_decoder_only(
        self,
        conv01: torch.Tensor,
        conv11: torch.Tensor,
        conv21: torch.Tensor,
        conv31: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Only run decoder with provided encoder features."""
        return self.decoder(conv01, conv11, conv21, conv31)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing FPN module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = FPNCfg(feat_chs=(8, 16, 32, 64))
    fpn = FPN(cfg).to(device)
    
    B, H, W = 2, 512, 640
    x = torch.randn(B, 3, H, W, device=device)
    
    with torch.no_grad():
        out = fpn(x, return_encoder_feats=True)
    
    print(f"\nInput: {x.shape}")
    print("\nOutputs:")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    
    # Verify scales
    assert out['stage1'].shape == (B, 64, H // 8, W // 8)
    assert out['stage2'].shape == (B, 32, H // 4, W // 4)
    assert out['stage3'].shape == (B, 16, H // 2, W // 2)
    assert out['stage4'].shape == (B, 8, H, W)
    
    print("\n" + "="*60)
    print("Testing with multi-view input...")
    
    V = 3
    x_mv = torch.randn(B, V, 3, H, W, device=device)
    x_flat = x_mv.view(B * V, 3, H, W)
    
    with torch.no_grad():
        out_mv = fpn(x_flat)
    
    print(f"\nMulti-view Input: {x_mv.shape} -> flattened: {x_flat.shape}")
    print("Outputs:")
    for k, v in out_mv.items():
        print(f"  {k}: {tuple(v.shape)}")
    
    print("\nAll tests passed!")