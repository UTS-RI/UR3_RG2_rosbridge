"""Capture an image: RGB, a depth heatmap and an overlay side by side."""

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, show_rgbd


def main():
    args = example_args(__doc__, overlay=True)
    camera_class = DemoCamera if args.demo else RealSenseCamera

    with camera_class() as camera:
        frame = camera.capture()

    # Standard NumPy arrays for further processing.
    print(f"Original RGB: {frame.native_rgb.shape}, {frame.native_rgb.dtype}")
    print(f"Depth: {frame.depth_m.shape}, {frame.depth_m.dtype}, unit: metres")
    print(f"Valid depth pixels: {frame.valid_depth.mean():.1%}")
    fig = show_rgbd(frame, min_depth_m=0.2, max_depth_m=3.0, alpha=args.alpha)
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
