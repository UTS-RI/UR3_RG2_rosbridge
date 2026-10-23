"""Update a rotatable point cloud with RGB or depth colours; Q/Escape quits."""

import argparse
from pathlib import Path

from _common import finish_figure
from realsense_lab import DemoCamera, RealSenseCamera, make_pointcloud, show_pointcloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--colour", choices=("rgb", "depth"), default="rgb")
    parser.add_argument("--step", type=int, default=8, help="Take every nth row and column")
    parser.add_argument("--frames", type=int, default=0, help="0 = until the window closes")
    parser.add_argument("--save", type=Path, help="Save the last frame; requires --frames > 0")
    args = parser.parse_args()
    if args.step < 1 or args.frames < 0 or (args.save and args.frames == 0):
        parser.error("Use --step >= 1 and --frames >= 0; saving requires --frames > 0.")
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    camera_class = DemoCamera if args.demo else RealSenseCamera
    fig = None
    try:
        with camera_class() as camera:
            cloud = make_pointcloud(camera.capture(), colour=args.colour, step=args.step)
            fig = show_pointcloud(cloud, title=f"Live point cloud: {args.colour} colours")
            ax = fig.axes[0]
            fig.canvas.mpl_connect("key_press_event", lambda event:
                                   plt.close(fig) if event.key in ("q", "escape") else None)
            count = 1
            if not args.save:
                plt.show(block=False)
            while True:
                if not args.save:
                    plt.pause(0.03)
                    if not plt.fignum_exists(fig.number):
                        break
                if args.frames and count >= args.frames:
                    break
                cloud = make_pointcloud(camera.capture(), colour=args.colour, step=args.step)
                for artist in list(ax.collections):
                    artist.remove()
                points = cloud.points
                ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=cloud.colours, s=2, depthshade=False)
                if not args.save:
                    fig.canvas.draw_idle()
                count += 1
            print(f"{count} point clouds displayed; rendering speed depends on the computer.")
        if args.save:
            finish_figure(fig, args.save)
    except KeyboardInterrupt:
        print("Capture stopped.")
    finally:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
