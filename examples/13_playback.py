"""Read an SDK recording through the same capture() interface, without a camera."""

import argparse
from pathlib import Path

from _common import add_overlay_argument, finish_figure
from realsense_lab import RealSenseCamera, show_rgbd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--frames", type=int, default=1, help="Frames to read; 0 = to the end of the file")
    parser.add_argument("--all-streams", action="store_true", help="Read IR and IMU too; they must exist in the recording")
    parser.add_argument("--save", type=Path, help="Save the last frame without opening a window")
    add_overlay_argument(parser)
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be >= 0.")
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    frame, count = None, 0
    with RealSenseCamera(playback_from=args.recording,
                         infrared=args.all_streams, imu=args.all_streams) as camera:
        while args.frames == 0 or count < args.frames:
            try:
                frame = camera.capture()
            except EOFError:
                break
            count += 1
    if frame is None:
        raise SystemExit("No complete RGB/depth pair was read from this recording.")
    print(f"Read {count} video frames; final timestamp {frame.timestamp_ms:.3f} ms ({frame.timestamp_domain})")
    finish_figure(show_rgbd(frame, alpha=args.alpha), args.save)


if __name__ == "__main__":
    main()
