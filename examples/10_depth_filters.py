"""Compare native depth with SDK-filtered depth from the same capture."""

import argparse
from pathlib import Path

from _common import finish_figure
from realsense_lab import DepthFilters, RealSenseCamera, depth_heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decimation", type=int, default=2, help="Depth reduction factor, 1..8")
    parser.add_argument("--hole-filling", action="store_true", help="Estimate missing depth values")
    parser.add_argument("--playback", type=Path, help="Use an SDK recording instead of a live camera")
    parser.add_argument("--save", type=Path, help="Save a figure without opening a window")
    args = parser.parse_args()
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filters = DepthFilters(decimation=args.decimation, hole_filling=args.hole_filling)
    with RealSenseCamera(filters=filters, playback_from=args.playback) as camera:
        # Give the temporal filter several frames of history, if available.
        frame = camera.capture()
        for _ in range(9):
            try:
                frame = camera.capture()
            except EOFError:
                break
    raw_m = frame.native_depth_raw.astype("float32") * frame.depth_scale_m
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, depth, title in zip(axes, (raw_m, frame.depth_m), ("Native depth", "Filtered depth")):
        ax.imshow(depth_heatmap(depth, max_depth_m=4.0))
        ax.set_title(f"{title}: {depth.shape[1]} × {depth.shape[0]}")
        ax.set_axis_off()
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
