# -*- coding: utf-8 -*-
"""DAB 染色面积量化：命令行与函数接口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from skimage import color, exposure, filters, io, morphology, util

EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_image_u8(path: Path) -> np.ndarray:
    """读取任意位深图像并拉伸到 uint8 RGB。"""
    img = io.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = util.img_as_float32(img)
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0, 1))
    return (img * 255.0 + 0.5).astype(np.uint8)


def gray_world_white_balance(img_u8: np.ndarray) -> np.ndarray:
    """灰世界白平衡：按均值缩放各通道。"""
    f = img_u8.astype(np.float32)
    mean = f.reshape(-1, 3).mean(axis=0) + 1e-6
    scale = 128.0 / mean
    f = f * scale
    return np.clip(f, 0, 255).astype(np.uint8)


def illumination_flatten(img_u8: np.ndarray, sigma_frac: float = 0.08) -> np.ndarray:
    """大尺度高斯估计背景后做逐通道归一。"""
    from skimage.filters import gaussian

    f = img_u8.astype(np.float32)
    h, w = f.shape[:2]
    sigma = max(3, int(min(h, w) * sigma_frac))
    out = np.empty_like(f)
    eps = 1e-6
    for channel in range(3):
        bg = gaussian(f[..., channel], sigma=sigma, preserve_range=True)
        out[..., channel] = f[..., channel] / (bg + eps) * np.median(bg)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_roi_mask(shape: Iterable[int], border_frac: float = 0.05) -> np.ndarray:
    """生成去除边框后的 ROI 掩膜。"""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = int(border_frac * h), int((1 - border_frac) * h)
    x0, x1 = int(border_frac * w), int((1 - border_frac) * w)
    mask[y0:y1, x0:x1] = True
    return mask


def dab_mask_from_hed(
    img_u8: np.ndarray,
    roi_mask: np.ndarray,
    manual_thr: float | None = None,
    min_obj_frac: float = 1e-4,
    open_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """HED 去卷积 -> 阈值 -> 形态学清噪。"""
    hed = color.rgb2hed(img_u8)
    dab = hed[..., 2]
    dab_01 = exposure.rescale_intensity(
        dab,
        in_range=(np.percentile(dab, 1), np.percentile(dab, 99)),
        out_range=(0, 1),
    )
    thr = float(manual_thr) if manual_thr is not None else filters.threshold_otsu(dab_01[roi_mask])
    mask = (dab_01 > thr) & roi_mask
    h, w = dab.shape
    min_obj = max(30, int(min_obj_frac * h * w))
    if open_radius > 0:
        mask = morphology.binary_opening(mask, morphology.disk(open_radius))
    mask = morphology.remove_small_objects(mask, min_obj)
    return mask, dab_01, thr


def overlay(rgb_u8: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.55) -> np.ndarray:
    """叠加彩色掩膜。"""
    out = rgb_u8.astype(np.float32)
    tint = np.array(color_rgb, np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def _iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in EXTS])


def process_one(
    path: Path,
    out_dir: Path,
    border_frac: float = 0.05,
    manual_thr: float | None = None,
    min_obj_frac: float = 1e-4,
    open_radius: int = 2,
) -> dict[str, float | int | str]:
    """处理单张图像并保存叠图。"""
    img0 = read_image_u8(path)
    img1 = gray_world_white_balance(img0)
    img2 = illumination_flatten(img1)
    roi = make_roi_mask(img2.shape, border_frac)
    mask, _, thr = dab_mask_from_hed(
        img2,
        roi,
        manual_thr=manual_thr,
        min_obj_frac=min_obj_frac,
        open_radius=open_radius,
    )
    percent = mask.mean() * 100.0
    vis = overlay(overlay(img2, roi, (0, 255, 0), 0.25), mask, (255, 0, 0), 0.60)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(out_dir / f"{path.stem}_overlay.png", vis)
    return {
        "image": path.name,
        "percent": round(percent, 4),
        "lesion_px": int(mask.sum()),
        "roi_px": int(roi.sum()),
        "thr": float(thr),
    }


def run(
    input_path: Path,
    out_dir: Path,
    border_frac: float = 0.05,
    manual_thr: float | None = None,
    min_obj_frac: float = 1e-4,
    open_radius: int = 2,
) -> pd.DataFrame:
    """批量处理并导出 CSV。"""
    paths = _iter_images(input_path)
    if not paths:
        return pd.DataFrame()
    records = [
        process_one(p, out_dir, border_frac, manual_thr, min_obj_frac, open_radius)
        for p in paths
    ]
    df = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="H2O2-DAB 视野面积测量")
    parser.add_argument("--input", required=True, type=Path, help="单张图片或目录")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--border-frac", type=float, default=0.05, help="忽略边框比例")
    parser.add_argument("--manual-thr", type=float, default=None, help="手动阈值 (0~1)")
    parser.add_argument("--min-obj-frac", type=float, default=1e-4, help="最小连通域占比")
    parser.add_argument("--open-radius", type=int, default=2, help="开运算半径")
    args = parser.parse_args()

    df = run(
        args.input,
        args.out,
        border_frac=args.border_frac,
        manual_thr=args.manual_thr,
        min_obj_frac=args.min_obj_frac,
        open_radius=args.open_radius,
    )
    if len(df):
        print(df)
    else:
        print("No images found.")


if __name__ == "__main__":
    main()
