"""Display the left and right native infrared images (greyscale)."""

from _common import example_args, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera


def main():
    args = example_args(__doc__)
    camera_class = DemoCamera if args.demo else RealSenseCamera
    with camera_class(infrared=True) as camera:
        frame = camera.capture()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, image, title in zip(axes, (frame.infrared_left, frame.infrared_right), ("Left IR", "Right IR")):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.set_axis_off()
    if args.demo:
        fig.suptitle("Synthetic IR demonstration — not a calibrated stereo pair")
    finish_figure(fig, args.save)


if __name__ == "__main__":
    main()
