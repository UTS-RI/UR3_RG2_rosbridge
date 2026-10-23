"""Update RGB and depth continuously. Quit with Q, Escape or by closing the window."""

import argparse
from pathlib import Path

from _common import add_overlay_argument, finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, depth_heatmap, depth_overlay, show_rgbd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Work without a camera")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames; 0 = until the window closes")
    parser.add_argument("--save", type=Path, help="Save the last frame without a window; requires --frames")
    add_overlay_argument(parser)
    args = parser.parse_args()
    if args.frames < 0 or (args.save and args.frames == 0):
        parser.error("--frames must be >= 0, or > 0 when using --save.")
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    camera_class = DemoCamera if args.demo else RealSenseCamera
    fig = None
    try:
        with camera_class() as camera:
            frame = camera.capture()
            fig = show_rgbd(frame, max_depth_m=3.0, alpha=args.alpha)
            artists = [ax.images[0] for ax in fig.axes[:3]]
            fig.canvas.mpl_connect("key_press_event", lambda event:
                                   plt.close(fig) if event.key in ("q", "escape") else None)
            if not args.save:
                plt.show(block=False)
            count = 1
            while True:
                if not args.save:
                    plt.pause(0.03)  # Process GUI events, including in demo mode.
                    if not plt.fignum_exists(fig.number):
                        break
                if args.frames and count >= args.frames:
                    break
                frame = camera.capture()
                images = (frame.native_rgb, depth_heatmap(frame.depth_m),
                          depth_overlay(frame, alpha=args.alpha))
                for artist, image in zip(artists, images):
                    artist.set_data(image)
                if not args.save:
                    fig.canvas.draw_idle()
                count += 1
            print(f"{count} frames captured.")
        if args.save:
            finish_figure(fig, args.save)
    except KeyboardInterrupt:
        print("Capture stopped.")
    finally:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
