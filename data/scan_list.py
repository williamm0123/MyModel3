# data/scan_list.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

# -----------------------------
# Part 1) scan list utilities
# -----------------------------

def load_scan_list(txt_path: str | Path) -> List[str]:
    """
    Read a txt file where each non-empty line is a scan name, e.g.:
      scan2
      scan6
      ...
    """
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"scan list not found: {p}")
    scans: List[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        # allow comment lines
        if s.startswith("#"):
            continue
        scans.append(s)
    if len(scans) == 0:
        raise ValueError(f"empty scan list: {p}")
    return scans


# -----------------------------
# Part 2) DTU path rules
# -----------------------------

Light = int  # 0..6, and -1 means "max"


@dataclass(frozen=True)
class DTURules:
    """
    Encodes your DTU naming rules:

    - Cameras: 49 views, view0=0..48, file: Cameras/train/{view0:08d}_cam.txt
    - Depths: 49 views, scan has _train suffix, file: Depths/{scan}_train/depth_map_{view0:04d}.pfm
    - Depths_raw: 64 views, scan no suffix, file: Depths_raw/{scan}/depth_map_{view0:04d}.pfm

    - Rectified: 49 views, scan has _train suffix,
      images indexed by view1=view0+1 => rect_{view1:03d}_{light}_r5000.png
      light: 0..6 (NO max)

    - Rectified_raw: 49 views, scan no suffix,
      images indexed by view1=view0+1 => rect_{view1:03d}_{light}_r5000.png
      plus max: rect_{view1:03d}_max.png
    """
    img_tag: str = "r5000"
    img_ext: str = ".png"


class DTUPaths:
    def __init__(self, root: str | Path, rules: DTURules | None = None):
        self.root = Path(root)
        self.rules = rules or DTURules()

        # top-level dirs
        self.cameras_dir = self.root / "Cameras"
        self.depths_dir = self.root / "Depths"
        self.depths_raw_dir = self.root / "Depths_raw"
        self.rectified_dir = self.root / "Rectified"
        self.rectified_raw_dir = self.root / "Rectified_raw"

        # cameras: 49-view params
        self.cameras49_dir = self.cameras_dir / "train"

        # pair.txt
        self.pair_path = self.cameras_dir / "pair.txt"

    @staticmethod
    def scan_train_name(scan: str) -> str:
        return f"{scan}_train"

    # ---- cameras ----
    def cam49(self, view0: int) -> Path:
        return self.cameras49_dir / f"{view0:08d}_cam.txt"

    # ---- depths ----
    def depth49(self, scan: str, view0: int) -> Path:
        return self.depths_dir / self.scan_train_name(scan) / f"depth_map_{view0:04d}.pfm"

    def depth64_raw(self, scan: str, view0: int) -> Path:
        return self.depths_raw_dir / scan / f"depth_map_{view0:04d}.pfm"

    # ---- images ----
    def rectified(self, scan: str, view0: int, light: int) -> Path:
        view1 = view0 + 1
        name = f"rect_{view1:03d}_{light}_{self.rules.img_tag}{self.rules.img_ext}"
        return self.rectified_dir / self.scan_train_name(scan) / name

    def rectified_raw(self, scan: str, view0: int, light: Light) -> Path:
        view1 = view0 + 1
        if light == -1:
            name = f"rect_{view1:03d}_max{self.rules.img_ext}"
        else:
            name = f"rect_{view1:03d}_{light}_{self.rules.img_tag}{self.rules.img_ext}"
        return self.rectified_raw_dir / scan / name

    # ---- selectors ----
    def image_path(
        self,
        image_source: Literal["rectified", "rectified_raw"],
        scan: str,
        view0: int,
        light: Light,
    ) -> Path:
        if image_source == "rectified_raw":
            return self.rectified_raw(scan, view0, light)
        return self.rectified(scan, view0, int(light))

    def depth_path(
        self,
        depth_source: Literal["depths", "depths_raw"],
        scan: str,
        view0: int,
    ) -> Path:
        if depth_source == "depths_raw":
            return self.depth64_raw(scan, view0)
        return self.depth49(scan, view0)
    # data/scan_list.py 末尾追加

def infer_dtu_defaults(rectified_dir: str) -> dict:
    """
    Only rectified_dir decides the rest defaults.

    rectified_dir:
    - "Rectified"      -> rectified + depths + light=0
    - "Rectified_raw"  -> rectified_raw + depths_raw + light=-1 (max)
    """
    if rectified_dir == "Rectified_raw":
        return {
            "image_source": "rectified_raw",
            "depth_source": "depths_raw",
            "light": -1,          # default use max
            "img_tag": "r5000",
            "img_ext": ".png",
        }
    # default: Rectified
    return {
        "image_source": "rectified",
        "depth_source": "depths",
        "light": 0,
        "img_tag": "r5000",
        "img_ext": ".png",
    }

