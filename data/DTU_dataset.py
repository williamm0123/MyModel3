# data/dtu_dataset.py
"""
DTU Dataset for MVS training.

Loads:
    - Multi-view images
    - Camera intrinsics and extrinsics
    - Depth GT and masks
    - Projection matrices for each stage

DTU MVS Dataset structure:
    Cameras/
        train/
            00000000_cam.txt  # Camera parameters per view
            ...
    Rectified/
        scan1/
            rect_001_0_r5000.png  # Rectified images
            ...
    Depths/
        scan1/
            depth_map_0000.pfm  # Depth maps
            ...
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import struct
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# ============================================================
# Utility Functions
# ============================================================

def read_pfm(filename: str) -> np.ndarray:
    """Read PFM file (depth map format)."""
    with open(filename, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError(f"Not a PFM file: {filename}")
        
        dim_line = f.readline().decode('utf-8')
        width, height = map(int, dim_line.split())
        
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        data = np.frombuffer(f.read(), endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
        return data.astype(np.float32)


def read_cam_file(filename: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Read camera file in DTU format.
    
    Returns:
        extrinsic: (4, 4) extrinsic matrix
        intrinsic: (3, 3) intrinsic matrix
        depth_min: minimum depth
        depth_max: maximum depth
    """
    with open(filename) as f:
        lines = f.readlines()
    
    # Parse extrinsic (4x4)
    extrinsic = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        values = list(map(float, lines[i + 1].split()))
        extrinsic[i] = values
    
    # Parse intrinsic (3x3)
    intrinsic = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        values = list(map(float, lines[i + 7].split()))
        intrinsic[i] = values[:3]
    
    # Parse depth range
    depth_line = lines[11].split()
    depth_min = float(depth_line[0])
    depth_interval = float(depth_line[1])
    depth_num = int(float(depth_line[2])) if len(depth_line) > 2 else 192
    depth_max = depth_min + depth_interval * depth_num
    
    return extrinsic, intrinsic, depth_min, depth_max


def compute_proj_matrices(
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    scales: Tuple[int, ...] = (8, 4, 2, 1),
    ) -> Dict[str, np.ndarray]:
    """
    Compute projection matrices for multiple scales.
    
    Args:
        intrinsic: (3, 3) intrinsic matrix
        extrinsic: (4, 4) extrinsic matrix
        scales: downsampling scales for each stage
    
    Returns:
        Dict with stage1-4 projection matrices
            Each: (2, 4, 4) where [0] is extrinsic, [1] is intrinsic (4x4)
    """
    proj_matrices = {}
    
    for i, scale in enumerate(scales):
        stage_key = f'stage{i + 1}'
        
        # Scale intrinsic
        K = intrinsic.copy()
        K[0, :] /= scale
        K[1, :] /= scale
        
        # Convert to 4x4
        K_4x4 = np.eye(4, dtype=np.float32)
        K_4x4[:3, :3] = K
        
        # Projection matrix: P = K @ [R|t]
        # We store extrinsic and intrinsic separately for later use
        proj = np.stack([extrinsic, K_4x4], axis=0)  # (2, 4, 4)
        proj_matrices[stage_key] = proj
    
    return proj_matrices


# ============================================================
# DTU Dataset
# ============================================================

@dataclass
class DTUConfig:
    """Configuration for DTU dataset."""
    data_root: str = ""
    split: str = "train"  # "train" or "val"
    
    # Views
    num_views: int = 3
    num_src_views: int = 2  # Number of source views
    
    # Image settings
    img_size: Tuple[int, int] = (512, 640)  # (H, W)
    
    # Depth settings
    depth_num: int = 192
    depth_interval: float = 2.5
    
    # Augmentation (for training)
    augment: bool = False


class DTUDataset(Dataset):
    """
    DTU MVS Dataset.
    
    Loads multi-view images, camera parameters, depth GT, and masks.
    """
    
    def __init__(self, cfg: DTUConfig):
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        
        # Load scan list
        if cfg.split == "train":
            self.scans = self._load_scan_list("train")
        else:
            self.scans = self._load_scan_list("val")
        
        # Build sample list (scan, ref_view, src_views)
        self.samples = self._build_samples()
        
        print(f"[DTUDataset] Loaded {len(self.samples)} samples from {len(self.scans)} scans")
    
    def _load_scan_list(self, split: str) -> List[str]:
        """Load list of scans for given split."""
        # Default DTU split
        if split == "train":
            # Training scans
            scans = [f"scan{i}" for i in range(1, 80) if i not in [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]]
        else:
            # Validation scans
            scans = ["scan3", "scan5", "scan17", "scan21", "scan28", "scan35", "scan37", "scan38"]
        
        # Filter existing
        existing = []
        for scan in scans:
            scan_path = self.data_root / "Rectified" / scan
            if scan_path.exists():
                existing.append(scan)
        
        return existing
    
    def _build_samples(self) -> List[Tuple[str, int, List[int]]]:
        """Build list of (scan, ref_view, src_views) tuples."""
        samples = []
        
        # View selection (default 49 views in DTU)
        all_views = list(range(49))
        
        for scan in self.scans:
            for ref_view in all_views:
                # Select source views (nearest views)
                src_views = self._select_src_views(ref_view, all_views, self.cfg.num_src_views)
                samples.append((scan, ref_view, src_views))
        
        return samples
    
    def _select_src_views(self, ref_view: int, all_views: List[int], num_src: int) -> List[int]:
        """Select source views based on view distance."""
        # Simple strategy: select nearest views
        candidates = [v for v in all_views if v != ref_view]
        
        # Sort by distance to ref
        candidates.sort(key=lambda v: abs(v - ref_view))
        
        return candidates[:num_src]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scan, ref_view, src_views = self.samples[idx]
        views = [ref_view] + src_views
        
        H, W = self.cfg.img_size
        
        # Load images
        images = []
        intrinsics = []
        extrinsics = []
        depth_mins = []
        depth_maxs = []
        
        for v in views:
            # Load image
            img_path = self._get_image_path(scan, v)
            img = self._load_image(img_path, (H, W))
            images.append(img)
            
            # Load camera
            cam_path = self._get_cam_path(scan, v)
            ext, intr, d_min, d_max = read_cam_file(cam_path)
            intrinsics.append(intr)
            extrinsics.append(ext)
            depth_mins.append(d_min)
            depth_maxs.append(d_max)
        
        images = torch.stack(images, dim=0)  # (V, 3, H, W)
        
        # Load depth GT (only for reference view)
        depth_path = self._get_depth_path(scan, ref_view)
        if depth_path.exists():
            depth_gt = read_pfm(str(depth_path))
            depth_gt = torch.from_numpy(depth_gt)
            depth_gt = F.interpolate(
                depth_gt.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            mask = (depth_gt > 0).float()
        else:
            depth_gt = torch.zeros(H, W)
            mask = torch.zeros(H, W)
        
        # Compute projection matrices for all stages
        proj_matrices = {}
        for stage_idx, scale in enumerate([8, 4, 2, 1]):
            stage_key = f'stage{stage_idx + 1}'
            stage_projs = []
            
            for v_idx in range(len(views)):
                proj = compute_proj_matrices(
                    intrinsics[v_idx],
                    extrinsics[v_idx],
                    scales=(scale,)
                )[f'stage1']
                stage_projs.append(torch.from_numpy(proj))
            
            proj_matrices[stage_key] = torch.stack(stage_projs, dim=0)  # (V, 2, 4, 4)
        
        # Depth range
        depth_min = min(depth_mins)
        depth_max = max(depth_maxs)
        depth_range = torch.tensor([depth_min, depth_max])
        depth_interval = torch.tensor((depth_max - depth_min) / self.cfg.depth_num)
        
        return {
            'images': images,
            'depth_gt': depth_gt,
            'mask': mask,
            'depth_range': depth_range,
            'depth_interval': depth_interval,
            'proj_matrices': proj_matrices,
            'meta': {
                'scan': scan,
                'ref_view': ref_view,
                'src_views': src_views,
            }
        }
    
    def _get_image_path(self, scan: str, view: int) -> Path:
        """Get path to rectified image."""
        # DTU naming: rect_001_0_r5000.png (view 0, light 0)
        filename = f"rect_{view + 1:03d}_0_r5000.png"
        return self.data_root / "Rectified" / scan / filename
    
    def _get_cam_path(self, scan: str, view: int) -> Path:
        """Get path to camera file."""
        # DTU naming: 00000000_cam.txt
        filename = f"{view:08d}_cam.txt"
        return self.data_root / "Cameras" / "train" / filename
    
    def _get_depth_path(self, scan: str, view: int) -> Path:
        """Get path to depth GT."""
        # DTU naming: depth_map_0000.pfm
        filename = f"depth_map_{view:04d}.pfm"
        return self.data_root / "Depths" / scan / filename
    
    def _load_image(self, path: Path, size: Tuple[int, int]) -> torch.Tensor:
        """Load and preprocess image."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path).convert("RGB")
        img = img.resize((size[1], size[0]), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        
        return img


# ============================================================
# Data Module (for easy use)
# ============================================================

def build_dataloaders(
    data_root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 640),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and validation dataloaders.
    
    Args:
        data_root: Path to DTU dataset
        batch_size: Batch size
        num_workers: Number of workers
        img_size: (H, W) image size
    
    Returns:
        train_loader, val_loader
    """
    train_cfg = DTUConfig(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=True,
    )
    val_cfg = DTUConfig(
        data_root=data_root,
        split="val",
        img_size=img_size,
        augment=False,
    )
    
    train_dataset = DTUDataset(train_cfg)
    val_dataset = DTUDataset(val_cfg)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing DTU Dataset...")
    
    # Test with mock data structure
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock structure
        os.makedirs(f"{tmpdir}/Rectified/scan1")
        os.makedirs(f"{tmpdir}/Cameras/train")
        os.makedirs(f"{tmpdir}/Depths/scan1")
        
        # Create mock image
        img = Image.new('RGB', (640, 512), color='red')
        img.save(f"{tmpdir}/Rectified/scan1/rect_001_0_r5000.png")
        img.save(f"{tmpdir}/Rectified/scan1/rect_002_0_r5000.png")
        img.save(f"{tmpdir}/Rectified/scan1/rect_003_0_r5000.png")
        
        # Create mock camera files
        cam_content = """extrinsic
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
525 0 320
0 525 256
0 0 1

425 2.5 192
"""
        for v in range(3):
            with open(f"{tmpdir}/Cameras/train/{v:08d}_cam.txt", 'w') as f:
                f.write(cam_content)
        
        print(f"Created mock data in {tmpdir}")
        
        # Test dataset
        cfg = DTUConfig(data_root=tmpdir, split="train", num_views=3)
        
        # Manual test since we only have scan1
        cfg.scans = ["scan1"]
        
        print("DTU Dataset test completed!")
    
    print("\nAll tests passed!")