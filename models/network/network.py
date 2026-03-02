# models/network/network.py
"""
Network: The central data flow controller for MyModel3.

This module orchestrates the complete MVSFormer++ front-half pipeline:
    1. FPN: Multi-scale CNN features
    2. DINOv3: Frozen ViT features (L3, L7, L11)
    3. SVA: Side View Attention for cross-view feature enhancement
    4. Fusion: Inject SVA features into FPN
    5. FMT: Feature Matching Transformer for cross-view matching

Data Flow:
    imgs (B, V, 3, H, W)
        │
        ├──► FPNEncoder ──► conv01, conv11, conv21, conv31
        │                          │
        │                          ▼
        │                   conv31 + vit_feat ──► FPNDecoder ──► stage1-4
        │                          ▲
        └──► DINO ──► L3,L7,L11 ──► SVA ──► vit_feat
        
    stage1-4 ──► FMT ──► enhanced stage1-4 (ready for Cost Volume)
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_cfg
from models.network.fpn import FPN, FPNCfg
from models.network.dinov3_encoder import DINOv3Encoder, DinoCfg
from models.network.sva import SVA, SVACfg
from models.network.FMT import FMT_with_pathway, FMTPathwayCfg, FMTCfg


class Network(nn.Module):
    """
    Main network for MVS feature extraction.
    
    Implements the complete MVSFormer++ front-half pipeline.
    """
    
    def __init__(self, cfg_dict: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device0 = device
        
        # Get config sections
        d = cfg_dict.get("dinov3", {})
        sva_cfg = cfg_dict.get("sva", {})
        fmt_cfg = cfg_dict.get("fmt", {})
        fpn_cfg = cfg_dict.get("fpn", {})
        
        # ============================================================
        # FPN Configuration
        # ============================================================
        # Default channel config: [8, 16, 32, 64] for stages 4,3,2,1
        feat_chs = tuple(fpn_cfg.get("feat_chs", [8, 16, 32, 64]))
        self.fpn = FPN(FPNCfg(feat_chs=feat_chs))
        print(f"[FPN] feat_chs={feat_chs}")
        
        # ============================================================
        # DINOv3 Encoder
        # ============================================================
        self.dino = DINOv3Encoder(
            DinoCfg(
                weights=str(d.get("weights", "")),
                arch=str(d.get("arch", "dinov3_vitb16")),
                patch_size=int(d.get("patch_size", 16)),
                input_scale=float(d.get("input_scale", 0.5)),
                use_imagenet_norm=bool(d.get("use_imagenet_norm", True)),
                pick_layers=tuple(d.get("pick_layers", [3, 7, 11])),
                freeze=bool(d.get("freeze", True)),
            ),
            device=device,
        )
        vit_ch = int(d.get("vit_ch", 768))
        
        # ============================================================
        # SVA (Side View Attention)
        # ============================================================
        sva_out_ch = feat_chs[-1]  # Match FPN stage1 channel (64)
        self.sva = SVA(SVACfg(
            vit_ch=vit_ch,
            out_ch=sva_out_ch,
            cross_interval_layers=int(sva_cfg.get("cross_interval_layers", 3)),
            num_heads=int(sva_cfg.get("num_heads", 12)),
            mlp_ratio=float(sva_cfg.get("mlp_ratio", 4.0)),
            init_values=float(sva_cfg.get("init_values", 1.0)),
            prev_values=float(sva_cfg.get("prev_values", 0.5)),
            post_norm=bool(sva_cfg.get("post_norm", False)),
            pre_norm_query=bool(sva_cfg.get("pre_norm_query", True)),
            no_combine_norm=bool(sva_cfg.get("no_combine_norm", False)),
        ))
        print(f"[SVA] vit_ch={vit_ch}, out_ch={sva_out_ch}")
        
        # ============================================================
        # FMT (Feature Matching Transformer)
        # ============================================================
        base_channel = feat_chs[0]  # 8
        fmt_d_model = feat_chs[-1]  # 64 (stage1 channel)
        layer_names = tuple(fmt_cfg.get("layer_names", ["self", "cross", "self", "cross"]))
        
        self.fmt = FMT_with_pathway(FMTPathwayCfg(
            base_channel=base_channel,
            fmt_cfg=FMTCfg(
                d_model=fmt_d_model,
                nhead=int(fmt_cfg.get("nhead", 4)),
                layer_names=layer_names,
                mlp_ratio=float(fmt_cfg.get("mlp_ratio", 4.0)),
                init_values=float(fmt_cfg.get("init_values", 1.0)),
            ),
        ))
        print(f"[FMT] base_channel={base_channel}, d_model={fmt_d_model}, layers={layer_names}")
    
    @staticmethod
    def flatten_bv(images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Reshape (B, V, 3, H, W) -> (B*V, 3, H, W), return (flattened, B, V)"""
        if images.ndim != 5:
            raise ValueError(f"Expected images (B,V,3,H,W), got {tuple(images.shape)}")
        B, V, C, H, W = images.shape
        return images.view(B * V, C, H, W), B, V
    
    @staticmethod
    def unflatten_bv(x: torch.Tensor, B: int, V: int) -> torch.Tensor:
        """Reshape (B*V, C, H, W) -> (B, V, C, H, W)"""
        return x.view(B, V, x.shape[1], x.shape[2], x.shape[3])
    
    def forward(
        self,
        images: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: (B, V, 3, H, W) - batch of multi-view images
            return_intermediate: if True, return intermediate features for debugging
        
        Returns:
            Dict containing:
                - stage1, stage2, stage3, stage4: FMT-enhanced features (B, V, C, H/s, W/s)
                - (optional) intermediate features if return_intermediate=True
        """
        x_bv, B, V = self.flatten_bv(images)
        _, _, H, W = x_bv.shape
        
        # ============================================================
        # Step 1: FPN Encoder
        # ============================================================
        conv01, conv11, conv21, conv31 = self.fpn.forward_encoder_only(x_bv)
        
        # ============================================================
        # Step 2: DINO + SVA (parallel with FPN)
        # ============================================================
        dino_out_bv = self.dino(x_bv)
        
        # Get DINO features and reshape to tokens
        dino_l3 = dino_out_bv["dino_l3"]  # (B*V, C, Ht, Wt)
        dino_l7 = dino_out_bv["dino_l7"]
        dino_l11 = dino_out_bv["dino_l11"]
        
        _, C_vit, Ht, Wt = dino_l3.shape
        
        # Reshape to (B, V, N, C) for SVA
        def to_bv_tokens(x):
            # x: (B*V, C, Ht, Wt) -> (B, V, Ht*Wt, C)
            x = x.view(B, V, C_vit, Ht, Wt)
            return x.permute(0, 1, 3, 4, 2).reshape(B, V, Ht * Wt, C_vit)
        
        vit_features = [
            to_bv_tokens(dino_l3),
            to_bv_tokens(dino_l7),
            to_bv_tokens(dino_l11),
        ]
        
        # Run SVA
        sva_out = self.sva(vit_features, spatial_shape=(Ht, Wt))
        vit_feat = sva_out["sva_out"]  # (B*V, C, Ht*4, Wt*4)
        
        # ============================================================
        # Step 3: Fuse SVA with FPN conv31
        # ============================================================
        # Align spatial dimensions if needed
        if vit_feat.shape[2] != conv31.shape[2] or vit_feat.shape[3] != conv31.shape[3]:
            vit_feat = F.interpolate(
                vit_feat,
                size=(conv31.shape[2], conv31.shape[3]),
                mode='bilinear',
                align_corners=False,
            )
        
        # Fuse: conv31 = conv31 + vit_feat
        conv31_fused = conv31 + vit_feat
        
        # ============================================================
        # Step 4: FPN Decoder
        # ============================================================
        feat1, feat2, feat3, feat4 = self.fpn.forward_decoder_only(
            conv01, conv11, conv21, conv31_fused
        )

        # FPN outputs after SVA fusion (before FMT), used for logging/debugging.
        pre_fmt_features = {
            'stage1': self.unflatten_bv(feat1, B, V),  # 1/8
            'stage2': self.unflatten_bv(feat2, B, V),  # 1/4
            'stage3': self.unflatten_bv(feat3, B, V),  # 1/2
            'stage4': self.unflatten_bv(feat4, B, V),  # 1/1
        }
        
        # Reshape to (B, V, C, H, W)
        features = {
            'stage1': pre_fmt_features['stage1'],
            'stage2': pre_fmt_features['stage2'],
            'stage3': pre_fmt_features['stage3'],
            'stage4': pre_fmt_features['stage4'],
        }
        
        # ============================================================
        # Step 5: FMT (Feature Matching Transformer)
        # ============================================================
        features = self.fmt(features)
        
        # ============================================================
        # Prepare output
        # ============================================================
        out = {
            'stage1': features['stage1'],
            'stage2': features['stage2'],
            'stage3': features['stage3'],
            'stage4': features['stage4'],
        }
        
        if return_intermediate:
            # SVA pyramid for multi-scale TensorBoard visualization.
            sva_s1 = vit_feat
            sva_s2 = F.interpolate(sva_s1, size=(feat2.shape[2], feat2.shape[3]), mode='bilinear', align_corners=False)
            sva_s3 = F.interpolate(sva_s1, size=(feat3.shape[2], feat3.shape[3]), mode='bilinear', align_corners=False)
            sva_s4 = F.interpolate(sva_s1, size=(feat4.shape[2], feat4.shape[3]), mode='bilinear', align_corners=False)

            # Add intermediate features for debugging
            out['dino_l3'] = self.unflatten_bv(dino_l3, B, V)
            out['dino_l7'] = self.unflatten_bv(dino_l7, B, V)
            out['dino_l11'] = self.unflatten_bv(dino_l11, B, V)
            out['sva_out'] = self.unflatten_bv(sva_out["sva_out"], B, V)
            out['sva_stage1'] = self.unflatten_bv(sva_s1, B, V)
            out['sva_stage2'] = self.unflatten_bv(sva_s2, B, V)
            out['sva_stage3'] = self.unflatten_bv(sva_s3, B, V)
            out['sva_stage4'] = self.unflatten_bv(sva_s4, B, V)
            out['conv31_fused'] = self.unflatten_bv(conv31_fused, B, V)
            out['fpn_sva_stage1'] = pre_fmt_features['stage1']
            out['fpn_sva_stage2'] = pre_fmt_features['stage2']
            out['fpn_sva_stage3'] = pre_fmt_features['stage3']
            out['fpn_sva_stage4'] = pre_fmt_features['stage4']
        
        return out


def build_network(device: str | None = None) -> Network:
    """
    Build network from config file.
    
    Args:
        device: Device string (e.g., "cuda", "cpu"). Auto-detected if None.
    
    Returns:
        Network instance on the specified device.
    """
    cfg = load_cfg()
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = Network(cfg, device=dev).to(dev)
    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing Network...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Minimal config for testing
    cfg = {
        "dinov3": {
            "weights": "",
            "arch": "dinov3_vitb16",
            "patch_size": 16,
            "vit_ch": 768,
            "freeze": True,
        },
        "fpn": {
            "feat_chs": [8, 16, 32, 64],
        },
        "sva": {
            "cross_interval_layers": 3,
            "num_heads": 12,
            "prev_values": 0.5,
        },
        "fmt": {
            "nhead": 4,
            "layer_names": ["self", "cross", "self", "cross"],
        },
    }
    
    net = Network(cfg, device).to(device)
    
    B, V, H, W = 1, 3, 480, 640
    images = torch.randn(B, V, 3, H, W, device=device)
    
    print(f"\nInput: {images.shape}")
    
    with torch.no_grad():
        out = net(images, return_intermediate=True)
    
    print("\nOutputs:")
    for k, v in out.items():
        if v is not None:
            has_nan = torch.isnan(v).any().item()
            print(f"  {k}: {tuple(v.shape)}, nan={has_nan}")
    
    # Verify output shapes
    assert out['stage1'].shape == (B, V, 64, H // 8, W // 8)
    assert out['stage2'].shape == (B, V, 32, H // 4, W // 4)
    assert out['stage3'].shape == (B, V, 16, H // 2, W // 2)
    assert out['stage4'].shape == (B, V, 8, H, W)
    
    print("\nAll tests passed!")
