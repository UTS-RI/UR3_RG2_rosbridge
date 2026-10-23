"""Experimental IMU capture with optional CSV export.

Live capture still needs validation on the lab computer. The demo uses synthetic data.
"""

import argparse
from pathlib import Path

import numpy as np

from realsense_lab import MotionReader, MotionSample, save_motion_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500, help="Total number of samples across both streams")
    parser.add_argument("--save", "--output", dest="save", type=Path, metavar="FILE.csv",
                        help="Save the motion samples as CSV")
    parser.add_argument("--serial", help="Select a specific camera")
    parser.add_argument("--accel-hz", type=int, default=63)
    parser.add_argument("--gyro-hz", type=int, default=200)
    parser.add_argument("--demo", action="store_true", help="Generate labelled stationary synthetic samples")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive.")
    if args.demo:
        samples = [MotionSample(
            "accel" if n % 2 == 0 else "gyro",
            np.array([0, -9.80665, 0] if n % 2 == 0 else [0, 0, 0], dtype=np.float32),
            (n // 2) * 10.0, "synthetic", n // 2,
        ) for n in range(args.samples)]
        print("Synthetic stationary data at 100 Hz per stream; not real IMU measurements.")
    else:
        print("Experimental IMU support. Live capture has not yet been verified on the lab computer.")
        try:
            with MotionReader(serial=args.serial, accel_hz=args.accel_hz, gyro_hz=args.gyro_hz) as reader:
                samples = [reader.read() for _ in range(args.samples)]
                print(f"Queue overflow: {reader.dropped_samples} samples dropped")
        except (RuntimeError, OSError) as exc:
            raise SystemExit(f"IMU capture failed: {exc}") from None
    for stream in ("accel", "gyro"):
        subset = [sample for sample in samples if sample.stream == stream]
        if subset:
            print(f"{stream}: {len(subset)} samples, latest {subset[-1].vector} {subset[-1].unit}")
    if args.save is not None:
        save_motion_csv(args.save, samples)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
