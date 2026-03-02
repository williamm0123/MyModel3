# # models/dinov3_encoder.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Tuple, Optional, Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..dinov3.vision_transformer import DinoVisionTransformer, ViTLocalCfg
# from .fpn import SimpleImageFPN


# @dataclass
# class DINOv3Config:
#     # 你本地权重 .pth（可选；Step1先把前向跑通）
#     weights: str = ""
#     patch_size: int = 16
#     use_imagenet_norm: bool = True

#     # 取哪些 block（1-based）
#     pick_layers: Tuple[int, ...] = (3, 7, 11)

#     # FPN
#     fpn_out_ch: int = 64


# @dataclass
# class Step1Cfg:
#     # DINO
#     dino_ckpt: str = "/home/william/project/dataset/pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"                 # 你本地 .pth，可先空
#     pick_layers: Tuple[int, ...] = (3, 7, 11)  # 1-based
#     patch_size: int = 16
#     use_imagenet_norm: bool = True
#     # FPN
#     fpn_out_ch: int = 64

# models/network/dinov3_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# vendored DINOv3 (keep under models/dinov3/)
from ..dinov3.vision_transformer import DinoVisionTransformer, ViTLocalCfg


@dataclass
class DinoCfg:
    weights: str = ""
    arch: str = "dinov3_vitb16"
    patch_size: int = 16
    input_scale: float = 0.5
    use_imagenet_norm: bool = True
    pick_layers: Tuple[int, int, int] = (3, 7, 11)  # 1-based
    freeze: bool = True


def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def align_hw_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    h2, w2 = (h // m) * m, (w // m) * m
    if h2 == h and w2 == w:
        return x
    return x[..., :h2, :w2]


def unwrap_ckpt(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a dict-like state_dict")
    return obj


def clean_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("teacher.backbone."):
            k = k[len("teacher.backbone."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        out[k] = v
    return out


class DINOv3Encoder(nn.Module):
    """
    Step1 DINO branch:
      images -> half-scale -> DINO -> layers 3/7/11 feature maps

    Input:
      images: (B,V,3,H,W) or (BV,3,H,W)

    Output (BV flattened):
      dino_l3/l7/l11: (BV, 768, Hp, Wp)
    """
    def __init__(self, cfg: DinoCfg, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device0 = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # vit-b/16 default
        vit_cfg = ViTLocalCfg(
            patch_size=int(cfg.patch_size),
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4.0,
            use_rope=True,
            n_storage_tokens=0,
        )
        self.backbone = DinoVisionTransformer(vit_cfg)   # ✅ 不传 device
        self.backbone.to(self.device0)                   # ✅ 统一迁移设备


        if cfg.weights:
            raw = torch.load(cfg.weights, map_location="cpu")
            sd = clean_keys(unwrap_ckpt(raw))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DINO] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        if cfg.freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @staticmethod
    def flatten_bv(images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if images.ndim == 5:
            B, V, C, H, W = images.shape
            return images.view(B * V, C, H, W), B, V
        if images.ndim == 4:
            return images, -1, -1
        raise ValueError(f"DINOv3Encoder expects 4D/5D, got {tuple(images.shape)}")

    def get_layers(self, x: torch.Tensor, layers_1based: List[int]) -> List[torch.Tensor]:
        fn = getattr(self.backbone, "get_intermediate_layers", None)
        if fn is None:
            raise AttributeError("DINO backbone has no get_intermediate_layers()")

        # Try multiple signatures
        try:
            return list(fn(x, layers_1based=layers_1based, reshape=True, norm=True))
        except TypeError:
            pass
        try:
            n0 = [i - 1 for i in layers_1based]
            return list(fn(x, n=n0, reshape=True, norm=True))
        except TypeError:
            pass

        # last resort: positional
        return list(fn(x, layers_1based, True, True))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, _, _ = self.flatten_bv(images)

        if float(self.cfg.input_scale) != 1.0:
            x = F.interpolate(x, scale_factor=float(self.cfg.input_scale), mode="bilinear", align_corners=False)
        x = align_hw_multiple(x, int(self.cfg.patch_size))
        if self.cfg.use_imagenet_norm:
            x = imagenet_norm(x)

        f3, f7, f11 = self.get_layers(x, list(self.cfg.pick_layers))
        return {"dino_l3": f3, "dino_l7": f7, "dino_l11": f11}
