"""Start your own project: remove the background and measure depth in a region."""

import numpy as np

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera


def main():
    args = example_args(__doc__)
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class() as camera:
        frame = camera.capture()

    # Your own processing starts here. Only NumPy arrays are needed.
    foreground = frame.valid_depth & (frame.depth_m < 1.8)
    masked_rgb = frame.rgb.copy()
    masked_rgb[~foreground] = 0

    height, width = frame.depth_m.shape
    roi = frame.depth_m[height // 3:2 * height // 3, width // 3:2 * width // 3]
    valid_values = roi[np.isfinite(roi) & (roi > 0)]
    if valid_values.size:
        print(f"Median Z-depth in the central third: {np.median(valid_values):.3f} m")
    else:
        print("No valid depth measurements in the region.")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, image, title in zip(axes, (frame.native_rgb, masked_rgb),
                                ("RGB (original)", "Aligned RGB with Z-depth < 1.8 m")):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
