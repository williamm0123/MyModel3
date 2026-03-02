# data/dtu_data.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

from .scan_list import load_scan_list, DTUPaths, DTURules, infer_dtu_defaults


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    # img: (H,W,3) uint8 -> (3,H,W) float32 [0,1]
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

def _read_pfm(path: str) -> np.ndarray:
    """Read PFM depth file into float32 (H,W)."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        color = header == "PF"
        width, height = map(int, f.readline().decode("utf-8").split())
        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.frombuffer(f.read(), endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    if color:
        # Shouldn't happen for DTU depth maps; keep the first channel if it does.
        data = data[..., 0]
    return data.astype(np.float32)


def _read_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Robust DTU camera txt parser.
    Returns: extrinsic(4,4), intrinsic(3,3), depth_min(float), depth_max(float)
    """
    with open(path, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    # drop empty lines
    lines = [ln for ln in raw_lines if ln != ""]

    def find_idx(keyword: str) -> int:
        key = keyword.lower()
        for i, ln in enumerate(lines):
            if key in ln.lower():
                return i
        return -1

    e_idx = find_idx("extrinsic")
    if e_idx < 0:
        raise ValueError(f"[cam] missing 'extrinsic' tag in {path}")

    extrinsic = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        vals = list(map(float, lines[e_idx + 1 + i].split()))
        if len(vals) != 4:
            raise ValueError(f"[cam] bad extrinsic row at {path}: {lines[e_idx+1+i]}")
        extrinsic[i] = vals

    if abs(float(extrinsic[3, 3])) < 1e-8:
        extrinsic[3] = np.array([0, 0, 0, 1], dtype=np.float32)

    k_idx = find_idx("intrinsic")
    if k_idx < 0:
        raise ValueError(f"[cam] missing 'intrinsic' tag in {path}")

    intrinsic = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        vals = list(map(float, lines[k_idx + 1 + i].split()))
        if len(vals) < 3:
            raise ValueError(f"[cam] bad intrinsic row at {path}: {lines[k_idx+1+i]}")
        intrinsic[i] = vals[:3]

    # depth line: usually "depth_min depth_interval [depth_num]"
    depth_min: Optional[float] = None
    depth_interval: Optional[float] = None
    depth_num = 192
    for ln in reversed(lines):
        parts = ln.split()
        if len(parts) >= 2:
            try:
                d0 = float(parts[0])
                d1 = float(parts[1])
                depth_min = d0
                depth_interval = d1
                if len(parts) >= 3:
                    depth_num = int(float(parts[2]))
                break
            except Exception:
                pass

    if depth_min is None or depth_interval is None:
        raise ValueError(f"[cam] missing depth line in {path}")

    depth_max = float(depth_min + depth_interval * depth_num)
    return extrinsic, intrinsic, float(depth_min), float(depth_max)


class DTUData(Dataset):
    """
    Two modes:
      - sample_mode="scan" (default): each item is a scan with absolute view ids.
      - sample_mode="mvs": each item is (scan, ref_view) and views are offsets from ref.

    Training uses "mvs" mode and returns full supervision keys.
    返回：
      images: (V,3,H,W)
      meta: {scan, ref_view(optional), views(list[int]), light(int), image_source}
    """

    def __init__(self, cfg: Dict[str, Any], split: str = "train", sample_mode: str = "scan"):
        self.cfg = cfg
        self.split = split
        if sample_mode not in ("scan", "mvs"):
            raise ValueError(f"Unsupported sample_mode={sample_mode}. Expected 'scan' or 'mvs'.")
        self.sample_mode = sample_mode

        # 1) scan list
        list_path = cfg.get("train_data_list") if split == "train" else cfg.get("val_data_list")
        if not list_path:
            raise KeyError(f"Missing scan list for split={split}. Need train_data_list/val_data_list in mvs.json")
        self.scans: List[str] = load_scan_list(list_path)

        # 2) dataset rules: only rectified_dir decides defaults
        ds = cfg.get("dataset", {})
        rectified_dir = ds.get("rectified_dir", "Rectified")
        auto = infer_dtu_defaults(rectified_dir)

        self.img_tag = ds.get("img_tag", auto["img_tag"])
        self.img_ext = ds.get("img_ext", auto["img_ext"])
        self.light = int(ds.get("light", auto["light"]))
        self.depth_source = ds.get("depth_source", auto["depth_source"])
        self.rectified_dir = rectified_dir
        self.image_source = auto["image_source"]
        resize_hw = ds.get("resize_hw", None)
        if resize_hw is None:
            self.resize_hw: Optional[Tuple[int, int]] = None
        else:
            if not isinstance(resize_hw, (list, tuple)) or len(resize_hw) != 2:
                raise ValueError(f"dataset.resize_hw must be [H, W], got: {resize_hw}")
            self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
            if self.resize_hw[0] <= 0 or self.resize_hw[1] <= 0:
                raise ValueError(f"dataset.resize_hw must be positive, got: {self.resize_hw}")

        # 3) DTU path helper (unified naming rules live here)
        self.paths = DTUPaths(cfg["datapath"], DTURules(img_tag=self.img_tag, img_ext=self.img_ext))

        # 4) view selection
        # For mvs mode: cfg["views"] are offsets from ref.
        # For scan mode: dataset.views (or cfg["views"] fallback) are absolute view ids.
        top_views = cfg.get("views", [0])
        ds_views = ds.get("views", None)
        self.views: List[int] = list(ds_views) if ds_views is not None else list(top_views)
        self.all_views: int = int(cfg.get("all_views", 49))

        # 5) samples
        if self.sample_mode == "scan":
            self.samples: List[Tuple[str, Optional[int]]] = [(s, None) for s in self.scans]
        else:
            self.samples = []
            for s in self.scans:
                for ref_idx in range(self.all_views):
                    self.samples.append((s, ref_idx))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scan, ref_view_idx = self.samples[idx]
        imgs: List[torch.Tensor] = []
        target_hw = None  # (H, W) determined by first view
        orig_hw = None

        if self.sample_mode == "scan":
            view_indices = list(self.views)
        else:
            assert ref_view_idx is not None
            view_indices = [int((ref_view_idx + v_off) % self.all_views) for v_off in self.views]

        for v_idx in view_indices:
            img_path = self.paths.image_path(self.image_source, scan, v_idx, self.light)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            pil = Image.open(img_path).convert("RGB")
            if orig_hw is None:
                orig_hw = (pil.height, pil.width)

            # pick target size from first image
            if self.resize_hw is not None:
                target_hw = self.resize_hw
            elif target_hw is None:
                target_hw = (pil.height, pil.width)
            else:
                # resize others to match
                if (pil.height, pil.width) != target_hw:
                    pil = pil.resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)

            if (pil.height, pil.width) != target_hw:
                pil = pil.resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)

            img = np.array(pil, dtype=np.uint8)
            imgs.append(_to_tensor(img))

        images = torch.stack(imgs, dim=0)  # (V,3,H,W)

        # In scan mode we keep the old behavior (image-only dataset).
        if self.sample_mode == "scan":
            return {
                "images": images,
                "meta": {
                    "scan": scan,
                    "views": view_indices,
                    "light": self.light,
                    "image_source": self.image_source,
                    "rectified_dir": self.rectified_dir,
                },
            }

        # ---- Training mode: load cameras + depth + build projections ----
        V, _, H, W = images.shape

        intrinsics: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        depth_mins: List[float] = []
        depth_maxs: List[float] = []

        for v_idx in view_indices:
            cam_path = self.paths.cam49(v_idx)
            try:
                if not cam_path.exists():
                    raise FileNotFoundError(str(cam_path))
                ext, intr, d_min, d_max = _read_cam_file(str(cam_path))

                # basic sanity
                if abs(float(intr[0, 0])) < 1e-8 or abs(float(intr[1, 1])) < 1e-8:
                    raise ValueError(f"bad intrinsic fx/fy: fx={intr[0,0]}, fy={intr[1,1]}")
                if abs(float(ext[3, 3])) < 1e-8:
                    ext[3] = np.array([0, 0, 0, 1], dtype=np.float32)
            except Exception as e:
                # fallback to default (do not crash training)
                print(f"[WARN][cam] fallback to default: scan={scan} view={v_idx} cam={cam_path} err={e}", flush=True)
                ext = np.eye(4, dtype=np.float32)
                intr = np.array([[525, 0, W / 2], [0, 525, H / 2], [0, 0, 1]], dtype=np.float32)
                d_min, d_max = 425.0, 905.0

            if orig_hw is not None and (orig_hw[0] != H or orig_hw[1] != W):
                sy = float(H) / float(orig_hw[0])
                sx = float(W) / float(orig_hw[1])
                intr[0, :] *= sx
                intr[1, :] *= sy

            intrinsics.append(intr)
            extrinsics.append(ext)
            depth_mins.append(float(d_min))
            depth_maxs.append(float(d_max))

        depth_path = self.paths.depth_path(self.depth_source, scan, view_indices[0])
        if depth_path.exists():
            depth_gt_np = _read_pfm(str(depth_path))
            depth_gt = torch.from_numpy(depth_gt_np)  # (H0,W0)
            if depth_gt.shape[0] != H or depth_gt.shape[1] != W:
                depth_gt = F.interpolate(
                    depth_gt.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            mask = (depth_gt > 0).float()
        else:
            depth_gt = torch.zeros((H, W), dtype=torch.float32)
            mask = torch.zeros((H, W), dtype=torch.float32)

        proj_matrices: Dict[str, torch.Tensor] = {}
        for stage_idx, scale in enumerate([8, 4, 2, 1]):
            stage_key = f"stage{stage_idx + 1}"
            stage_projs: List[torch.Tensor] = []

            for i in range(V):
                K = intrinsics[i].copy()
                K[0, :] /= scale
                K[1, :] /= scale

                K_4x4 = np.eye(4, dtype=np.float32)
                K_4x4[:3, :3] = K

                proj = np.stack([extrinsics[i], K_4x4], axis=0)  # (2,4,4)
                stage_projs.append(torch.from_numpy(proj))

            proj_matrices[stage_key] = torch.stack(stage_projs, dim=0)  # (V,2,4,4)

        d_min = float(min(depth_mins))
        d_max = float(max(depth_maxs))
        depth_range = torch.tensor([d_min, d_max], dtype=torch.float32)
        depth_interval = torch.tensor((d_max - d_min) / 192.0, dtype=torch.float32)

        return {
            "images": images,
            "depth_gt": depth_gt,
            "mask": mask,
            "depth_range": depth_range,
            "depth_interval": depth_interval,
            "proj_matrices": proj_matrices,
            "meta": {
                "scan": scan,
                "ref_view": int(ref_view_idx) if ref_view_idx is not None else None,
                "views": view_indices,
                "light": self.light,
                "image_source": self.image_source,
                "rectified_dir": self.rectified_dir,
            }

        }
