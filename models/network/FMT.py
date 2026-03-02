# models/network/fmt.py
"""
Feature Matching Transformer (FMT) for MVSFormer++.

Based on official implementation in FMT.py

FMT performs cross-view feature matching:
- Reference view: Self-Attention to enhance features
- Source views: Cross-Attention with reference as key/value

FMT_with_pathway adds FPN-style upsampling:
- stage1 (1/8) -> stage2 (1/4) -> stage3 (1/2) -> stage4 (1/1)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Normalized 2D Positional Encoding (for FMT)
# ============================================================

class PositionEncodingSineNorm(nn.Module):
    """
    Sinusoidal 2D positional encoding with normalization.
    
    Normalizes positions to max_shape to handle variable resolutions.
    Uses caching for efficiency.
    """
    
    def __init__(self, d_model: int, max_shape: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe_cache: Dict[str, torch.Tensor] = {}
    
    def _create_pe(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create positional encoding for given shape."""
        pe = torch.zeros((self.d_model, H, W), device=device)
        
        # Normalized positions (scaled to max_shape)
        y_position = torch.ones((H, W), device=device).cumsum(0).float() * self.max_shape[0] / H
        x_position = torch.ones((H, W), device=device).cumsum(1).float() * self.max_shape[1] / W
        
        # Frequency terms
        div_term = torch.exp(
            torch.arange(0, self.d_model // 2, 2, device=device).float() 
            * (-math.log(10000.0) / (self.d_model // 2))
        )
        div_term = div_term[:, None, None]  # (C//4, 1, 1)
        
        # Interleaved sin/cos for x and y
        pe[0::4, :, :] = torch.sin(x_position.unsqueeze(0) * div_term)
        pe[1::4, :, :] = torch.cos(x_position.unsqueeze(0) * div_term)
        pe[2::4, :, :] = torch.sin(y_position.unsqueeze(0) * div_term)
        pe[3::4, :, :] = torch.cos(y_position.unsqueeze(0) * div_term)
        
        return pe.unsqueeze(0)  # (1, C, H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C, H, W) with PE added
        """
        _, _, H, W = x.shape
        cache_key = f"{H}-{W}"
        
        if cache_key not in self.pe_cache:
            self.pe_cache[cache_key] = self._create_pe(H, W, x.device)
        
        pe = self.pe_cache[cache_key]
        if pe.device != x.device:
            pe = pe.to(x.device)
            self.pe_cache[cache_key] = pe
        
        return x + pe


# ============================================================
# Attention Block for FMT
# ============================================================

class FMTBlock(nn.Module):
    """
    Attention block for FMT supporting both Self and Cross attention.
    
    Uses Pre-LN with residual scaling (gamma).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        dropout: float = 0.0,
        post_norm: bool = False,
        pre_norm_query: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.post_norm = post_norm
        self.pre_norm_query = pre_norm_query
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_k = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gamma1 = nn.Parameter(torch.tensor(init_values), requires_grad=True)
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.gamma2 = nn.Parameter(torch.tensor(init_values), requires_grad=True)
    
    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) query
            key: (B, M, C) key for cross-attention, None for self-attention
            value: (B, M, C) value for cross-attention, None for self-attention
            attn_bias: optional attention bias
        Returns:
            (B, N, C)
        """
        if key is None:
            # Self-attention
            if self.post_norm:
                attn_out = self._self_attn(x)
                x = self.norm1(x + self.gamma1 * attn_out)
                x = self.norm2(x + self.gamma2 * self.ffn(x))
            else:
                x_norm = self.norm1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
                x = x + self.gamma1 * attn_out
                x = x + self.gamma2 * self.ffn(self.norm2(x))
        else:
            # Cross-attention
            if self.post_norm:
                attn_out = self._cross_attn(x, key, value)
                x = self.norm1(x + self.gamma1 * attn_out)
                x = self.norm2(x + self.gamma2 * self.ffn(x))
            else:
                if self.pre_norm_query:
                    q = self.norm1(x)
                else:
                    q = x
                k = self.norm1_k(key)
                v = self.norm1_v(value)
                attn_out, _ = self.attn(q, k, v, need_weights=False)
                x = x + self.gamma1 * attn_out
                x = x + self.gamma2 * self.ffn(self.norm2(x))
        
        return x
    
    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return out
    
    def _cross_attn(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if self.pre_norm_query:
            q = self.norm1(x)
        else:
            q = x
        k = self.norm1_k(key)
        v = self.norm1_v(value)
        out, _ = self.attn(q, k, v, need_weights=False)
        return out


# ============================================================
# FMT Configuration
# ============================================================

@dataclass
class FMTCfg:
    """Configuration for FMT module."""
    d_model: int = 64                    # Feature dimension
    nhead: int = 4                       # Number of attention heads
    layer_names: Tuple[str, ...] = ("self", "cross", "self", "cross")  # Attention pattern
    mlp_ratio: float = 4.0
    init_values: float = 1.0
    dropout: float = 0.0
    post_norm: bool = False
    pre_norm_query: bool = False


# ============================================================
# FMT Module
# ============================================================

class FMT(nn.Module):
    """
    Feature Matching Transformer.
    
    Processes multi-view features:
    - Reference view: Self-attention to build ref_feature_list
    - Source views: Cross-attention using ref_feature_list as key/value
    """
    
    def __init__(self, cfg: FMTCfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.layer_names = list(cfg.layer_names)
        
        # Position encoding
        self.pos_encoding = PositionEncodingSineNorm(cfg.d_model)
        
        # Attention layers
        self.layers = nn.ModuleList([
            FMTBlock(
                dim=cfg.d_model,
                num_heads=cfg.nhead,
                mlp_ratio=cfg.mlp_ratio,
                init_values=cfg.init_values,
                dropout=cfg.dropout,
                post_norm=cfg.post_norm,
                pre_norm_query=cfg.pre_norm_query,
            )
            for _ in range(len(self.layer_names))
        ])
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_ref(self, ref_feature: torch.Tensor) -> List[torch.Tensor]:
        """
        Process reference view with self-attention.
        
        Args:
            ref_feature: (B, C, H, W)
        Returns:
            List of intermediate features after each self-attention layer
        """
        B, C, H, W = ref_feature.shape
        assert C == self.d_model, f"Expected C={self.d_model}, got {C}"
        
        # Add positional encoding and flatten
        x = self.pos_encoding(ref_feature)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply self-attention layers, collect outputs
        ref_feature_list = []
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                x = layer(x)
                # Reshape back to spatial and store
                feat = x.transpose(1, 2).reshape(B, C, H, W)
                ref_feature_list.append(feat)
        
        return ref_feature_list
    
    def forward_src(
        self,
        ref_feature_list: List[torch.Tensor],
        src_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process source view with cross-attention.
        
        Args:
            ref_feature_list: List of reference features from forward_ref
            src_feature: (B, C, H, W) source feature
        Returns:
            (B, C, H, W) processed source feature
        """
        B, C, H, W = src_feature.shape
        assert C == self.d_model
        
        # Convert ref_feature_list to tokens
        ref_tokens_list = [
            f.flatten(2).transpose(1, 2)  # (B, H*W, C)
            for f in ref_feature_list
        ]
        
        # Add PE and flatten source
        x = self.pos_encoding(src_feature)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply layers
        ref_idx = 0
        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == 'self':
                x = layer(x)
            elif name == 'cross':
                # Select reference feature
                if len(ref_tokens_list) == len(self.layers):
                    r_idx = i
                else:
                    r_idx = ref_idx
                    ref_idx = min(ref_idx + 1, len(ref_tokens_list) - 1)
                
                ref_tokens = ref_tokens_list[min(r_idx, len(ref_tokens_list) - 1)]
                x = layer(x, key=ref_tokens, value=ref_tokens)
        
        # Reshape back
        return x.transpose(1, 2).reshape(B, C, H, W)
    
    def forward(
        self,
        ref_feature: Optional[torch.Tensor] = None,
        src_feature: Optional[torch.Tensor] = None,
        feat: str = "ref",
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Unified forward interface (compatible with official API).
        
        Args:
            ref_feature: For feat="ref": (B, C, H, W) reference feature
                        For feat="src": List of reference features
            src_feature: For feat="src": (B, C, H, W) source feature
            feat: "ref" or "src"
        """
        if feat == "ref":
            return self.forward_ref(ref_feature)
        elif feat == "src":
            return self.forward_src(ref_feature, src_feature)
        else:
            raise ValueError(f"Unknown feat type: {feat}")


# ============================================================
# FMT with Pathway (FPN-style upsampling)
# ============================================================

@dataclass  
class FMTPathwayCfg:
    """Configuration for FMT_with_pathway."""
    base_channel: int = 8
    fmt_cfg: FMTCfg = None
    
    def __post_init__(self):
        if self.fmt_cfg is None:
            self.fmt_cfg = FMTCfg(d_model=self.base_channel * 8)


class FMT_with_pathway(nn.Module):
    """
    FMT with FPN-style pathway for multi-scale feature fusion.
    
    Takes features from 4 stages and processes them:
    - stage1 (1/8 scale): FMT for cross-view matching
    - stage2-4: FPN-style upsampling + fusion
    
    Input:  {stage1: (B,V,64,H/8,W/8), stage2: (B,V,32,H/4,W/4), 
             stage3: (B,V,16,H/2,W/2), stage4: (B,V,8,H,W)}
    Output: Same structure, with cross-view enhanced features
    """
    
    def __init__(self, cfg: FMTPathwayCfg):
        super().__init__()
        self.cfg = cfg
        base_ch = cfg.base_channel
        
        # FMT for stage1
        self.fmt = FMT(cfg.fmt_cfg)
        
        # Dimension reduction (stage1 -> stage2 -> stage3 -> stage4)
        # stage1: 8*base_ch, stage2: 4*base_ch, stage3: 2*base_ch, stage4: base_ch
        self.dim_reduction_1 = nn.Conv2d(base_ch * 8, base_ch * 4, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1, bias=False)
        self.dim_reduction_3 = nn.Conv2d(base_ch * 2, base_ch * 1, 1, bias=False)
        
        # Smoothing convolutions
        self.smooth_1 = nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1, bias=False)
        self.smooth_3 = nn.Conv2d(base_ch * 1, base_ch * 1, 3, padding=1, bias=False)
    
    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x and add to y."""
        _, _, H, W = y.shape
        return F.interpolate(x.float(), size=(H, W), mode='bilinear', align_corners=False) + y
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process multi-view multi-scale features.
        
        Args:
            features: Dict with keys 'stage1', 'stage2', 'stage3', 'stage4'
                Each value: (B, V, C, H, W)
        Returns:
            Dict with same structure, cross-view enhanced
        """
        B, V, C, H, W = features['stage1'].shape
        
        stage1_feats = []
        stage2_feats = []
        stage3_feats = []
        stage4_feats = []
        
        ref_feat_list = None
        
        for v in range(V):
            # Get features for this view
            s1 = features['stage1'][:, v]  # (B, C, H, W)
            s2 = features['stage2'][:, v]
            s3 = features['stage3'][:, v]
            s4 = features['stage4'][:, v]
            
            if v == 0:
                # Reference view: self-attention
                ref_feat_list = self.fmt.forward(s1, feat="ref")
                s1_out = ref_feat_list[-1]
            else:
                # Source view: cross-attention
                s1_out = self.fmt.forward(ref_feat_list, s1, feat="src")
            
            stage1_feats.append(s1_out)
            
            # FPN-style upsampling
            s2_out = self.smooth_1(self._upsample_add(self.dim_reduction_1(s1_out), s2))
            stage2_feats.append(s2_out)
            
            s3_out = self.smooth_2(self._upsample_add(self.dim_reduction_2(s2_out), s3))
            stage3_feats.append(s3_out)
            
            s4_out = self.smooth_3(self._upsample_add(self.dim_reduction_3(s3_out), s4))
            stage4_feats.append(s4_out)
        
        return {
            'stage1': torch.stack(stage1_feats, dim=1),
            'stage2': torch.stack(stage2_feats, dim=1),
            'stage3': torch.stack(stage3_feats, dim=1),
            'stage4': torch.stack(stage4_feats, dim=1),
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing FMT module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test FMT
    fmt_cfg = FMTCfg(d_model=64, nhead=4)
    fmt = FMT(fmt_cfg).to(device)
    
    B, V, C, H, W = 2, 3, 64, 80, 100
    ref_feat = torch.randn(B, C, H, W, device=device)
    src_feat = torch.randn(B, C, H, W, device=device)
    
    with torch.no_grad():
        ref_list = fmt.forward(ref_feat, feat="ref")
        src_out = fmt.forward(ref_list, src_feat, feat="src")
    
    print(f"\nFMT Test:")
    print(f"  Input ref: {ref_feat.shape}")
    print(f"  Ref list: {len(ref_list)} items, each {ref_list[0].shape}")
    print(f"  Src output: {src_out.shape}")
    
    # Test FMT_with_pathway
    print("\n" + "="*60)
    print("Testing FMT_with_pathway...")
    
    base_ch = 8
    pathway_cfg = FMTPathwayCfg(
        base_channel=base_ch,
        fmt_cfg=FMTCfg(d_model=base_ch * 8, nhead=4),
    )
    fmt_pathway = FMT_with_pathway(pathway_cfg).to(device)
    
    features = {
        'stage1': torch.randn(B, V, base_ch * 8, H, W, device=device),
        'stage2': torch.randn(B, V, base_ch * 4, H * 2, W * 2, device=device),
        'stage3': torch.randn(B, V, base_ch * 2, H * 4, W * 4, device=device),
        'stage4': torch.randn(B, V, base_ch * 1, H * 8, W * 8, device=device),
    }
    
    with torch.no_grad():
        out_features = fmt_pathway(features)
    
    print(f"\nFMT_with_pathway Test:")
    for k, v in out_features.items():
        has_nan = torch.isnan(v).any().item()
        print(f"  {k}: {tuple(v.shape)}, nan={has_nan}")
    
    print("\nAll tests passed!")