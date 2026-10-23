"""Colour a 3D point cloud using the registered RGB image."""

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, make_pointcloud, show_pointcloud


def main():
    args = example_args(__doc__)
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class() as camera:
        frame = camera.capture()

    cloud = make_pointcloud(frame, colour="rgb", step=4, max_depth_m=3.0)
    print(f"Points: {cloud.points.shape}; Colours: {cloud.colours.shape}")
    fig = show_pointcloud(cloud, title="Point cloud: RGB colours")
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
