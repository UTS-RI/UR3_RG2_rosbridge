"""Reusable processing: heatmaps, RGB overlays and coloured point clouds."""

from typing import Literal

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

from .types import PointCloud, RGBDFrame


def _check_depth_range(min_depth_m: float, max_depth_m: float) -> None:
    if not np.isfinite([min_depth_m, max_depth_m]).all() or not 0 <= min_depth_m < max_depth_m:
        raise ValueError("Depth limits must be finite: 0 <= min_depth_m < max_depth_m.")


def depth_heatmap(
    depth_m: NDArray[np.floating],
    *,
    min_depth_m: float = 0.2,
    max_depth_m: float = 3.0,
    cmap: str = "turbo_r",
) -> NDArray[np.uint8]:
    """Convert depth to an RGB heatmap (H, W, 3), uint8.

    Fixed scale: turbo_r colours nearby points red and distant points blue.
    Values outside the scale are clipped to its end colours. Zero, negative,
    NaN and infinite values appear black. Measurements are not modified.
    """
    _check_depth_range(min_depth_m, max_depth_m)
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional depth image.")
    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, min_depth_m)
    normalised = np.clip((safe_depth - min_depth_m) / (max_depth_m - min_depth_m), 0, 1)
    rgb = colormaps[cmap](normalised, bytes=True)[..., :3].copy()
    rgb[~valid] = 0
    return rgb


def depth_overlay(
    frame: RGBDFrame,
    *,
    alpha: float = 0.5,
    min_depth_m: float = 0.2,
    max_depth_m: float = 3.0,
    cmap: str = "turbo_r",
) -> NDArray[np.uint8]:
    """Blend RGB with a depth heatmap; alpha=1 shows only depth colours.

    Keep the original RGB wherever the depth measurement is missing.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    heatmap = depth_heatmap(frame.depth_m, min_depth_m=min_depth_m,
                            max_depth_m=max_depth_m, cmap=cmap)
    result = ((1 - alpha) * frame.rgb + alpha * heatmap).round().astype(np.uint8)
    result[~frame.valid_depth] = frame.rgb[~frame.valid_depth]
    return result


def make_pointcloud(
    frame: RGBDFrame,
    *,
    colour: Literal["rgb", "depth"] = "rgb",
    step: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 3.0,
    cmap: str = "turbo_r",
    color: Literal["rgb", "depth"] | None = None,
) -> PointCloud:
    """Create a point cloud with RGB colours or a depth heatmap.

    step=2 takes every second row and column (roughly a quarter of the points).
    Discard invalid points and depths outside the selected range.
    Colours are float32 in 0..1, suitable for Matplotlib or Open3D.
    """
    _check_depth_range(min_depth_m, max_depth_m)
    if color is not None:
        if colour != "rgb" and colour != color:
            raise ValueError("colour and the compatibility alias color disagree.")
        colour = color
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be an integer >= 1.")
    if colour not in ("rgb", "depth"):
        raise ValueError("colour must be 'rgb' or 'depth'.")
    depth = frame.depth_m[::step, ::step]
    points = frame.xyz_m[::step, ::step]
    valid = (np.isfinite(depth) & (depth > 0)
             & (depth >= min_depth_m) & (depth <= max_depth_m)
             & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0))
    if colour == "rgb":
        colors = frame.rgb[::step, ::step]
    else:
        colors = depth_heatmap(depth, min_depth_m=min_depth_m,
                              max_depth_m=max_depth_m, cmap=cmap)
    return PointCloud(points=points[valid].astype(np.float32),
                      colors=colors[valid].astype(np.float32) / 255.0)


def point_at(frame: RGBDFrame, x: int, y: int) -> NDArray[np.float32]:
    """Return calibrated XYZ at an aligned image pixel; reject missing depth."""
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (x, y)):
        raise ValueError("Pixel coordinates must be integers.")
    height, width = frame.depth_m.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Pixel is outside the {width} x {height} image.")
    point = frame.xyz_m[y, x]
    if not frame.valid_depth[y, x] or not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError("There is no valid 3D measurement at this pixel.")
    return point.copy()


def distance_between(frame: RGBDFrame, first: tuple[int, int], second: tuple[int, int]) -> float:
    """Euclidean distance in metres between two (x, y) image pixels."""
    return float(np.linalg.norm(point_at(frame, *first) - point_at(frame, *second)))
