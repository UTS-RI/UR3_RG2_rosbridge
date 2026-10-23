"""Capture camera streams, with optional .db3 recording for later playback."""

import argparse
from pathlib import Path
from time import monotonic

import numpy as np

from realsense_lab import RealSenseCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--save", "--output", dest="save", type=Path, metavar="FILE.db3",
                        help="Record enabled streams to a new .db3 file")
    parser.add_argument("--infrared", action="store_true", help="Include both infrared streams")
    parser.add_argument("--imu", action="store_true", help="Include accelerometer and gyroscope")
    parser.add_argument("--all-streams", action="store_true", help="Enable all six D435i streams")
    parser.add_argument("--serial")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite.")
    with RealSenseCamera(record_to=args.save, serial=args.serial, warmup_frames=0,
                         infrared=args.infrared or args.all_streams, imu=args.imu or args.all_streams) as camera:
        deadline = monotonic() + args.seconds
        count = 0
        while monotonic() < deadline:
            camera.capture()
            count += 1
    # The optional recorder stores enabled native streams before image filtering.
    if args.save is not None:
        print(f"Recording closed: {args.save} ({count} video frames consumed by this script)")
    else:
        print(f"Captured {count} video frames. No recording saved. Use --save FILE.db3 to record.")


if __name__ == "__main__":
    main()
