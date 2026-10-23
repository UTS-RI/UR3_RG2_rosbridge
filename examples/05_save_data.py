"""Capture RGB and depth, with optional image, measurement and point cloud export."""

import argparse
from datetime import datetime
from pathlib import Path

from matplotlib import image as mpl_image

from _common import add_overlay_argument
from realsense_lab import (
    DemoCamera, RealSenseCamera, depth_heatmap, depth_overlay,
    load_frame, make_pointcloud, save_frame, save_pointcloud,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Work without a camera")
    parser.add_argument("--save", "--output", dest="save", type=Path, metavar="FOLDER",
                        help="Save data in a new capture folder inside FOLDER")
    add_overlay_argument(parser)
    args = parser.parse_args()
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class() as camera:
        frame = camera.capture()

    print(f"Captured RGB {frame.native_rgb.shape} and depth {frame.depth_m.shape}, unit: metres")
    if args.save is None:
        print("No files saved. Use --save FOLDER to export this example's data.")
        return

    # Use a new folder for each run to preserve earlier captures.
    folder = args.save / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    mpl_image.imsave(folder / "rgb.png", frame.native_rgb)
    mpl_image.imsave(folder / "depth_heatmap.png", depth_heatmap(frame.depth_m))
    mpl_image.imsave(folder / "depth_overlay.png", depth_overlay(frame, alpha=args.alpha))
    save_frame(folder / "frame.npz", frame)  # Retain the original metric measurements.
    for colour in ("rgb", "depth"):
        cloud = make_pointcloud(frame, colour=colour, step=2)
        save_pointcloud(folder / f"pointcloud_{colour}.ply", cloud)

    # Process the same data later without a camera.
    loaded = load_frame(folder / "frame.npz")
    print(f"Saved to: {folder.resolve()}")
    print(f"Reloaded: {loaded.depth_m.shape}, unit: metres")


if __name__ == "__main__":
    main()
