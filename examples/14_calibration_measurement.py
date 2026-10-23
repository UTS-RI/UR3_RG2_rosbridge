"""Measure a 3D distance, with optional calibration and image export."""

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, distance_between, point_at, save_json


def main():
    args = example_args(__doc__)
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class() as camera:
        frame = camera.capture()
        calibration = {"synthetic_intrinsics": frame.intrinsics} if args.demo else camera.calibration
    if args.save is not None:
        report = args.save.with_suffix(".json")
        save_json(report, calibration)
        print(f"Calibration saved to {report}")
    height, width = frame.depth_m.shape
    first, second = (width // 3, height // 2), (2 * width // 3, height // 2)
    try:
        print(f"First XYZ: {point_at(frame, *first)} m")
        print(f"Second XYZ: {point_at(frame, *second)} m")
        print(f"3D distance: {distance_between(frame, first, second):.3f} m")
    except ValueError as exc:
        print(f"Measurement unavailable: {exc}")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.imshow(frame.rgb)
    ax.plot([first[0], second[0]], [first[1], second[1]], "y-o")
    ax.set_title("3D distance between two registered image pixels")
    ax.set_axis_off()
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
