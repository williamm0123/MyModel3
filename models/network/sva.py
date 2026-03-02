# models/network/sva.py
"""
Side View Attention (SVA) module for MVSFormer++.

Based on official implementation: CrossVITDecoder in module.py

Architecture:
    Input: DINOv2 features from layers [L3, L7, L11] for all views
    
    Reference path (Self-Attention):
        L3 → ref_feat_list[0]
        SelfAttn(ref_feat_list[0]) × ALS + L7 → ref_feat_list[1]
        SelfAttn(ref_feat_list[1]) × ALS + L11 → ref_feat_list[2]
    
    Source path (Cross-Attention):
        CrossAttn(L3_src, kv=ref_feat_list[0]) → src_feat
        CrossAttn(src_feat × ALS + L7_src, kv=ref_feat_list[1]) → src_feat
        CrossAttn(src_feat × ALS + L11_src, kv=ref_feat_list[2]) → src_feat
    
    Output: Upsampled features (1/32 → 1/8 scale)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Attention Block (Pre-LN with residual scaling)
# ============================================================

class CrossBlock(nn.Module):
    """
    Attention block supporting both Self-Attention and Cross-Attention.
    
    Based on official CrossBlock implementation.
    
    Self-Attention: x = x + gamma * Attn(LN(x))
    Cross-Attention: x = x + gamma * Attn(LN(x), LN(key), LN(value))
    
    Then: x = x + gamma * FFN(LN(x))
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        dropout: float = 0.0,
        post_norm: bool = False,
        pre_norm_query: bool = True,
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
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) query tokens
            key: (B, M, C) key tokens, None for self-attention
            value: (B, M, C) value tokens, None for self-attention
        Returns:
            (B, N, C) output tokens
        """
        # Self-attention or Cross-attention
        if key is None:
            # Self-attention
            if self.post_norm:
                x = self.norm1(x + self.gamma1 * self._self_attn(x))
            else:
                x = x + self.gamma1 * self._self_attn(self.norm1(x))
        else:
            # Cross-attention
            if self.post_norm:
                x = self.norm1(x + self.gamma1 * self._cross_attn(x, key, value))
            else:
                if self.pre_norm_query:
                    q = self.norm1(x)
                else:
                    q = x
                k = self.norm1_k(key)
                v = self.norm1_v(value)
                x = x + self.gamma1 * self._cross_attn_normalized(q, k, v)
        
        # FFN
        if self.post_norm:
            x = self.norm2(x + self.gamma2 * self.ffn(x))
        else:
            x = x + self.gamma2 * self.ffn(self.norm2(x))
        
        return x
    
    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out
    
    def _cross_attn(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x) if self.pre_norm_query else x
        k = self.norm1_k(key)
        v = self.norm1_v(value)
        out, _ = self.attn(q, k, v, need_weights=False)
        return out
    
    def _cross_attn_normalized(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, k, v, need_weights=False)
        return out


# ============================================================
# SVA Configuration
# ============================================================

@dataclass
class SVACfg:
    """Configuration for SVA module."""
    # DINOv2 feature dimension
    vit_ch: int = 768
    
    # Output channel (after upsampling)
    out_ch: int = 64
    
    # Number of DINOv2 layers to use
    cross_interval_layers: int = 3
    
    # Attention config
    num_heads: int = 12
    mlp_ratio: float = 4.0
    init_values: float = 1.0      # gamma initialization
    prev_values: float = 0.5      # ALS initialization
    
    # Architecture options
    post_norm: bool = False       # Pre-LN (False) vs Post-LN (True)
    pre_norm_query: bool = True   # Normalize query in cross-attention
    no_combine_norm: bool = False # Skip norm after ALS combination


# ============================================================
# SVA Module (CrossVITDecoder)
# ============================================================

class SVA(nn.Module):
    """
    Side View Attention (SVA) module.
    
    This is a re-implementation of CrossVITDecoder from MVSFormer++.
    
    Key features:
    - Reference view: Self-Attention path
    - Source views: Cross-Attention path (query=src, key/value=ref)
    - ALS (Adaptive Layer Scaling): Learnable weights for combining attention output with next layer features
    - Upsampling: 1/32 → 1/8 scale via ConvTranspose
    """
    
    def __init__(self, cfg: SVACfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.vit_ch
        n_layers = cfg.cross_interval_layers  # 3
        
        # ============================================================
        # Self-Attention blocks for Reference (n_layers - 1 = 2)
        # Used between L3→L7 and L7→L11
        # ============================================================
        self.self_attn_blocks = nn.ModuleList([
            CrossBlock(
                dim=dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                init_values=cfg.init_values,
                post_norm=cfg.post_norm,
                pre_norm_query=cfg.pre_norm_query,
            )
            for _ in range(n_layers - 1)  # 2 blocks
        ])
        
        # ============================================================
        # Cross-Attention blocks for Source (n_layers = 3)
        # One for each DINOv2 layer
        # ============================================================
        self.cross_attn_blocks = nn.ModuleList([
            CrossBlock(
                dim=dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                init_values=cfg.init_values,
                post_norm=cfg.post_norm,
                pre_norm_query=cfg.pre_norm_query,
            )
            for _ in range(n_layers)  # 3 blocks
        ])
        
        # ============================================================
        # ALS (Adaptive Layer Scaling) - prev_values in official code
        # Learnable weights for combining attention output with next layer
        # ============================================================
        self.prev_values = nn.ParameterList([
            nn.Parameter(torch.tensor(cfg.prev_values), requires_grad=True)
            for _ in range(n_layers - 1)  # 2 weights
        ])
        
        # ============================================================
        # Normalization layers (optional, after ALS combination)
        # ============================================================
        if not cfg.no_combine_norm:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(dim)
                for _ in range(n_layers - 1)  # 2 norms
            ])
        else:
            self.norm_layers = None
        
        # ============================================================
        # Upsampling: 1/32 → 1/8 (4x total)
        # proj: 768 → 256
        # upsampler0: 256 → 128 (2x)
        # upsampler1: 128 → 64 (2x)
        # ============================================================
        ch = cfg.out_ch  # 64
        self.proj = nn.Sequential(
            nn.Conv2d(dim, ch * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.SiLU(),
        )
        self.upsampler0 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.SiLU(),
        )
        self.upsampler1 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.SiLU(),
        )
    
    def forward(
        self,
        vit_features: List[torch.Tensor],
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SVA.
        
        Args:
            vit_features: List of 3 tensors [L3, L7, L11]
                Each tensor: (B, V, N, C) where N = H*W (flattened spatial)
                Or: (B, V, H, W, C) which will be reshaped
            spatial_shape: (H, W) of the spatial dimensions at 1/32 scale
                Required if vit_features are flattened (B, V, N, C)
        
        Returns:
            Dict containing:
                - sva_out: (B*V, out_ch, H*4, W*4) upsampled features
                - ref_feat: (B, N, C) final reference features
                - src_feats: (B, V-1, N, C) final source features
        """
        # Get shape info
        x0 = vit_features[0]
        if x0.dim() == 5:
            # (B, V, H, W, C) → (B, V, H*W, C)
            B, V, H, W, C = x0.shape
            vit_features = [f.reshape(B, V, H * W, C) for f in vit_features]
            spatial_shape = (H, W)
        else:
            B, V, N, C = x0.shape
            if spatial_shape is None:
                # Try to infer from N (only works for square)
                H = W = int(N ** 0.5)
                if H * W != N:
                    raise ValueError(
                        f"Cannot infer H, W from N={N}. "
                        f"Please provide spatial_shape=(H, W) argument."
                    )
                spatial_shape = (H, W)
            else:
                H, W = spatial_shape
                assert H * W == N, f"spatial_shape {spatial_shape} doesn't match N={N}"
        
        n_layers = len(vit_features)  # 3
        
        # ============================================================
        # Process each view
        # ============================================================
        ref_feat_list = []  # Reference features at each layer
        src_feat_list = []  # Final source features for each source view
        
        for v in range(V):
            if v == 0:
                # ========== Reference view: Self-Attention path ==========
                for i in range(n_layers):
                    if i == 0:
                        # First layer: directly add to list
                        ref_feat_list.append(vit_features[i][:, v])  # (B, N, C)
                    else:
                        # Subsequent layers: SelfAttn → ALS → Add next layer
                        attn_out = self.self_attn_blocks[i - 1](ref_feat_list[-1])
                        
                        # ALS: combine attention output with next layer features
                        new_ref_feat = self.prev_values[i - 1] * attn_out + vit_features[i][:, v]
                        
                        # Optional normalization
                        if self.norm_layers is not None:
                            new_ref_feat = self.norm_layers[i - 1](new_ref_feat)
                        
                        ref_feat_list.append(new_ref_feat)
            
            else:
                # ========== Source views: Cross-Attention path ==========
                src_feat = None
                
                for i in range(n_layers):
                    if i == 0:
                        # First layer: Cross-Attn with query=src, kv=ref
                        src_feat = self.cross_attn_blocks[i](
                            x=vit_features[i][:, v],      # query: source L3
                            key=ref_feat_list[i],          # key: ref L3
                            value=ref_feat_list[i],        # value: ref L3
                        )
                    else:
                        # Subsequent layers: ALS combine → Cross-Attn
                        query = self.prev_values[i - 1] * src_feat + vit_features[i][:, v]
                        
                        # Optional normalization
                        if self.norm_layers is not None:
                            query = self.norm_layers[i - 1](query)
                        
                        src_feat = self.cross_attn_blocks[i](
                            x=query,
                            key=ref_feat_list[i],
                            value=ref_feat_list[i],
                        )
                
                src_feat_list.append(src_feat.unsqueeze(1))  # (B, 1, N, C)
        
        # ============================================================
        # Combine ref and src features
        # ============================================================
        
        # Stack source features: (B, V-1, N, C)
        if len(src_feat_list) > 0:
            src_feats = torch.cat(src_feat_list, dim=1)
        else:
            src_feats = None
        
        # Final reference feature
        ref_feat = ref_feat_list[-1]  # (B, N, C)
        
        # Combine: (B, V, N, C)
        if src_feats is not None:
            combined = torch.cat([ref_feat.unsqueeze(1), src_feats], dim=1)
        else:
            combined = ref_feat.unsqueeze(1)
        
        # ============================================================
        # Upsample: 1/32 → 1/8
        # ============================================================
        
        # Reshape to spatial: (B*V, C, H, W)
        B, V, N, C = combined.shape
        H, W = spatial_shape
        
        x = combined.reshape(B * V, H, W, C).permute(0, 3, 1, 2)  # (B*V, C, H, W)
        
        # Upsample
        x = self.proj(x)       # (B*V, 256, H, W)
        x = self.upsampler0(x) # (B*V, 128, 2H, 2W)
        x = self.upsampler1(x) # (B*V, 64, 4H, 4W)
        
        return {
            "sva_out": x,              # (B*V, out_ch, H*4, W*4) = (B*V, 64, H/8, W/8)
            "ref_feat": ref_feat,      # (B, N, C) final ref features at 1/32
            "src_feats": src_feats,    # (B, V-1, N, C) final src features at 1/32
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing SVA module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SVACfg(vit_ch=768, out_ch=64)
    sva = SVA(cfg).to(device)
    
    # Test input: 3 layers, each (B, V, N, C)
    # Non-square: 37 x 50 = 1850 tokens (matches your error)
    B, V, Ht, Wt, C = 2, 3, 37, 50, 768  # 37x50 = 1850 tokens at 1/32 scale
    N = Ht * Wt
    
    vit_features = [
        torch.randn(B, V, N, C, device=device),  # L3
        torch.randn(B, V, N, C, device=device),  # L7
        torch.randn(B, V, N, C, device=device),  # L11
    ]
    
    with torch.no_grad():
        out = sva(vit_features, spatial_shape=(Ht, Wt))
    
    print("\nOutputs:")
    for k, v in out.items():
        if v is not None:
            has_nan = torch.isnan(v).any().item()
            print(f"  {k}: {tuple(v.shape)}, mean={v.mean():.4f}, std={v.std():.4f}, nan={has_nan}")
        else:
            print(f"  {k}: None")
    
    print(f"\nExpected sva_out shape: ({B * V}, {cfg.out_ch}, {Ht * 4}, {Wt * 4})")
    print(f"Actual sva_out shape: {tuple(out['sva_out'].shape)}")
    
    # Verify shapes
    assert out['sva_out'].shape == (B * V, cfg.out_ch, Ht * 4, Wt * 4)
    assert out['ref_feat'].shape == (B, N, C)
    assert out['src_feats'].shape == (B, V - 1, N, C)
    
    print("\nAll tests passed!")