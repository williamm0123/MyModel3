# models/dinov3/vision_transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .layers.patch_embed import PatchEmbed
from .layers.block import SelfAttentionBlock
from .layers.rope_position_encoding import RopePositionEmbedding


@dataclass
class ViTLocalCfg:
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    ffn_ratio: float = 4.0
    qkv_bias: bool = False
    proj_bias: bool = True
    ffn_bias: bool = True
    drop_path_rate: float = 0.0
    layerscale_init: Optional[float] = None

    n_storage_tokens: int = 0
    use_rope: bool = True
    rope_base: float = 100.0
    rope_dtype: torch.dtype = torch.float32


class DinoVisionTransformer(nn.Module):
    """
    按你现有版本裁剪：保留 prepare_tokens / RoPE / get_intermediate_layers
    关键改动：不再把 device= 透传给 PatchEmbed/Block/LayerNorm（统一用 model.to(device)）
    """
    def __init__(self, cfg: ViTLocalCfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.depth = cfg.depth
        self.num_heads = cfg.num_heads
        self.n_storage_tokens = cfg.n_storage_tokens

        # ✅ PatchEmbed：删掉 device=device（你的 PatchEmbed 不接受这个参数）
        self.patch_embed = PatchEmbed(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            flatten_embedding=False,
        )

        # ✅ 参数不要在这里绑 device，统一让外部 .to(device)
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim))
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, cfg.embed_dim))
        else:
            self.storage_tokens = None

        self.rope_embed = None
        if cfg.use_rope:
            # ✅ RopePositionEmbedding：删 device=device（按你现在实现，它 forward 时会用输入 device）
            self.rope_embed = RopePositionEmbedding(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                base=cfg.rope_base,
                dtype=cfg.rope_dtype,
            )

        # ✅ blocks：删 device=device
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                ffn_ratio=cfg.ffn_ratio,
                qkv_bias=cfg.qkv_bias,
                proj_bias=cfg.proj_bias,
                ffn_bias=cfg.ffn_bias,
                drop_path=cfg.drop_path_rate,
                init_values=cfg.layerscale_init,
            )
            for _ in range(cfg.depth)
        ])

        # ✅ LayerNorm：删 device=device
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.init_weights()

    # 下面其余函数保持你原样即可（prepare_tokens / _rope_for_hw / forward_features / get_intermediate_layers）


    def init_weights(self):
        if self.rope_embed is not None:
            self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)

    def prepare_tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        x: (B,3,H,W)
        return:
          tokens: (B, 1+n_storage+N, C)
          hw: (Hp, Wp) patch grid size (不是像素，是 patch grid 的 H,W)
        """
        x = self.patch_embed(x)  # (B, Hp, Wp, C)
        B, Hp, Wp, C = x.shape
        x = x.flatten(1, 2)      # (B, N, C)

        cls = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            st = self.storage_tokens.expand(B, -1, -1)
        else:
            st = x.new_empty((B, 0, C))

        tokens = torch.cat([cls, st, x], dim=1)
        return tokens, (Hp, Wp)

    def _rope_for_hw(self, Hp: int, Wp: int):
        if self.rope_embed is None:
            return None
        # rope_position_encoding.py 的 forward 返回 (sin,cos) tuple :contentReference[oaicite:3]{index=3}
        return self.rope_embed(H=Hp, W=Wp)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """
        返回 dict，兼容 DINOv3 风格 key：
        - x_norm_patchtokens: (B, N, C)
        - x_norm_clstoken: (B, C)
        """
        tokens, (Hp, Wp) = self.prepare_tokens(x)
        rope = self._rope_for_hw(Hp, Wp)

        for blk in self.blocks:
            tokens = blk(tokens, rope)

        tokens_norm = self.norm(tokens)
        cls = tokens_norm[:, 0]  # (B,C)
        patch = tokens_norm[:, 1 + self.n_storage_tokens:]  # (B,N,C)

        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patch,
            "hw": torch.tensor([Hp, Wp], device=x.device),
        }

    def _resolve_blocks_to_take(
        self,
        *,
        layers_1based: Optional[Sequence[int]],
        n: Optional[Union[int, Sequence[int]]],
    ) -> List[int]:
        if layers_1based is not None:
            blocks = [int(i) - 1 for i in layers_1based]
        else:
            n = 1 if n is None else n
            if isinstance(n, int):
                if n <= 0 or n > self.depth:
                    raise ValueError(f"n must be in [1, {self.depth}], got {n}")
                blocks = list(range(self.depth - n, self.depth))
            else:
                blocks = [int(i) for i in n]

        if not blocks:
            raise ValueError("No intermediate layer selected")
        if any(i < 0 or i >= self.depth for i in blocks):
            raise ValueError(f"Layer indices out of range [0, {self.depth - 1}]: {blocks}")
        return blocks

    def get_intermediate_layers(
        self,
        x: Tensor,
        layers_1based: Optional[Sequence[int]] = None,
        *,
        n: Optional[Union[int, Sequence[int]]] = None,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        return_cls: bool = False,
    ):
        """
        Supports both local and official-style APIs:
          - local:   get_intermediate_layers(x, layers_1based=[3, 7, 11], reshape=True, norm=True)
          - official:get_intermediate_layers(x, n=[2, 6, 10], reshape=True, norm=True)
        """
        if return_cls:
            return_class_token = True

        blocks_to_take = self._resolve_blocks_to_take(layers_1based=layers_1based, n=n)
        wanted = set(blocks_to_take)

        tokens, (Hp, Wp) = self.prepare_tokens(x)
        rope = self._rope_for_hw(Hp, Wp)

        outs: Dict[int, Tensor] = {}
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, rope)
            if i in wanted:
                outs[i] = self.norm(tokens) if norm else tokens

        if len(outs) != len(wanted):
            raise RuntimeError(f"Only collected {len(outs)} / {len(wanted)} layers")

        ordered = [outs[i] for i in blocks_to_take]
        class_tokens = [out[:, 0] for out in ordered]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in ordered]
        patch_tokens = [out[:, 1 + self.n_storage_tokens :] for out in ordered]

        if reshape:
            patch_tokens = [
                patch.transpose(1, 2).contiguous().view(patch.shape[0], patch.shape[2], Hp, Wp)
                for patch in patch_tokens
            ]

        if not return_class_token and not return_extra_tokens:
            return patch_tokens
        if return_class_token and not return_extra_tokens:
            return list(zip(patch_tokens, class_tokens))
        if not return_class_token and return_extra_tokens:
            return list(zip(patch_tokens, extra_tokens))
        return list(zip(patch_tokens, class_tokens, extra_tokens))
