"""Matplotlib visualisation: RGB, heatmaps and rotatable 3D point clouds."""

import numpy as np

from .processing import depth_heatmap, depth_overlay
from .types import PointCloud, RGBDFrame


def show_rgbd(frame: RGBDFrame, *, min_depth_m: float = 0.2, max_depth_m: float = 3.0,
              alpha: float = 0.5):
    """Show the original RGB image, a depth heatmap and an aligned overlay.

    Older snapshots without native RGB use a labelled aligned image instead.
    alpha controls depth opacity from 0 to 1. Missing depth retains RGB.
    Open with plt.show() or save with fig.savefig(...).
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    rgb = frame.native_rgb if frame.native_rgb is not None else frame.rgb
    rgb_title = "RGB (original)" if frame.native_rgb is not None else "RGB (aligned)"
    images = (rgb,
              depth_heatmap(frame.depth_m, min_depth_m=min_depth_m, max_depth_m=max_depth_m),
              depth_overlay(frame, alpha=alpha, min_depth_m=min_depth_m, max_depth_m=max_depth_m))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, image, title in zip(axes, images, (rgb_title, "Depth heatmap", "RGB + depth (aligned)")):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    scale = ScalarMappable(norm=Normalize(min_depth_m, max_depth_m), cmap="turbo_r")
    fig.colorbar(scale, ax=list(axes[1:]), label="Z-depth [m]", shrink=0.75)
    return fig


def show_pointcloud(cloud: PointCloud, *, title: str = "Point cloud"):
    """Return a rotatable 3D figure (a snapshot).

    Display (X, Y, Z) as (X, Z, -Y) so up in the image is up in the scene.
    The original point coordinates are not modified.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    if len(cloud.points):
        display = cloud.points[:, [0, 2, 1]].copy()
        display[:, 2] *= -1
        ax.scatter(*display.T, c=cloud.colors, s=2, depthshade=False)
        # Use the same scale on all axes, including for nearly flat scenes.
        center = (display.min(axis=0) + display.max(axis=0)) / 2
        half_size = max(float(np.ptp(display, axis=0).max()) / 2, 0.05)
        ax.set_xlim(center[0] - half_size, center[0] + half_size)
        ax.set_ylim(center[1] - half_size, center[1] + half_size)
        ax.set_zlim(center[2] - half_size, center[2] + half_size)
    else:
        ax.text2D(0.15, 0.5, "No valid points in the selected depth range.", transform=ax.transAxes)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X [m] – right", ylabel="Z [m] – forwards", zlabel="−Y [m] – up", title=title)
    ax.view_init(elev=15, azim=-75)
    return fig
