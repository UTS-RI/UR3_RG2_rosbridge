"""Colour a 3D point cloud by depth: near is red, far is blue."""

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, make_pointcloud, show_pointcloud


def main():
    args = example_args(__doc__)
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class() as camera:
        frame = camera.capture()

    # step=4 takes every fourth row and column for smoother rotation.
    cloud = make_pointcloud(frame, colour="depth", step=4, min_depth_m=0.2, max_depth_m=3.0)
    print(f"{len(cloud.points)} points; XYZ in metres, RGB colours in 0..1")
    fig = show_pointcloud(cloud, title="Point cloud: depth colours")

    # Use the same fixed colour scale as the 2D depth image.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    scale = ScalarMappable(norm=Normalize(0.2, 3.0), cmap="turbo_r")
    fig.colorbar(scale, ax=fig.axes[0], label="Z-depth [m]", shrink=0.6)
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
