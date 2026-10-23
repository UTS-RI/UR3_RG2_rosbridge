"""Run with python -m realsense_lab.check before a lab session."""

import argparse
from importlib.metadata import version
import platform

from .camera import RealSenseCamera
from .demo import DemoCamera
from .device import inspect_devices
from .processing import make_pointcloud


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Also test a connected RGB/depth camera")
    parser.add_argument("--all-streams", action="store_true",
                        help="Test D435i RGB, depth and IR, plus experimental IMU support")
    args = parser.parse_args()
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    for name in ("numpy", "matplotlib"):
        print(f"{name}: {version(name)}")
    with DemoCamera() as camera:
        cloud = make_pointcloud(camera.capture())
    print(f"Demo processing: OK ({len(cloud.points)} points)")
    if args.camera or args.all_streams:
        try:
            reports = inspect_devices()
            if not reports:
                raise RuntimeError("No RealSense devices found.")
            for report in reports:
                print(f"Device: {report['info'].get('name', 'RealSense')}")
            with RealSenseCamera(infrared=args.all_streams, imu=args.all_streams) as camera:
                frame = camera.capture()
                motion_streams = {sample.stream for sample in frame.motion}
                if args.all_streams:
                    for _ in range(9):
                        if {"accel", "gyro"} <= motion_streams:
                            break
                        frame = camera.capture()
                        motion_streams.update(sample.stream for sample in frame.motion)
                    if not {"accel", "gyro"} <= motion_streams:
                        raise RuntimeError("Both IMU streams were requested but did not provide samples.")
            print(f"Camera capture: OK ({frame.depth_m.shape[1]} x {frame.depth_m.shape[0]})")
            if args.all_streams:
                if frame.infrared_left is None or frame.infrared_right is None:
                    raise RuntimeError("An infrared stream did not return a frame.")
                print(f"Infrared: OK; IMU streams received: {sorted(motion_streams)}")
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Camera check failed: {exc}")
            print("On Linux, check the RealSense USB and IMU/IIO permissions in README.md.")
            return 1
    print("Preflight complete. Interactive GUI operation must be checked separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
