"""A synthetic scene for exercises without a connected camera."""

import numpy as np

from .types import MotionSample, RGBDFrame


class DemoCamera:
    """The same start/capture/stop interface as RealSenseCamera.

    Generate a sphere, a rectangle and a background using an ideal pinhole
    model. These are synthetic data, not a recorded RealSense scene.
    """

    def __init__(self, width: int = 640, height: int = 480, *, infrared: bool = False, imu: bool = False):
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (width, height)):
            raise ValueError("width and height must be positive integers.")
        self.width = width
        self.height = height
        self.infrared = infrared
        self.imu = imu
        self._running = False
        self._frame_number = 0

    def start(self) -> "DemoCamera":
        self._running = True
        return self

    def capture(self) -> RGBDFrame:
        if not self._running:
            raise RuntimeError("Start DemoCamera using 'with' or start() first.")
        v, u = np.mgrid[:self.height, :self.width].astype(np.float32)
        x = (u - self.width / 2) / self.width
        y = (v - self.height / 2) / self.height
        depth = (2.2 + 0.3 * x + 0.15 * y).astype(np.float32)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[..., 0] = 60 + 100 * (x + 0.5)
        rgb[..., 1] = 90 + 80 * (y + 0.5)
        rgb[..., 2] = 180

        center_x = -0.19 + 0.04 * np.sin(self._frame_number / 12)
        radius_squared = ((x - center_x) / 0.19) ** 2 + ((y + 0.02) / 0.25) ** 2
        sphere = radius_squared < 1
        depth[sphere] = 1.3 - 0.3 * np.sqrt(1 - radius_squared[sphere])
        rgb[sphere] = [240, 85, 45]
        rectangle = (x > 0.12) & (x < 0.35) & (np.abs(y) < 0.27)
        depth[rectangle] = 1.6
        rgb[rectangle] = [45, 200, 100]

        # A small missing region: zero depth must not be treated as an object.
        depth[(x < -0.43) & (y > 0.32)] = 0
        focal_length = 0.9 * self.width
        xyz = np.stack(((u - self.width / 2) * depth / focal_length,
                        (v - self.height / 2) * depth / focal_length, depth), axis=-1)
        self._frame_number += 1
        timestamp = self._frame_number * 1000 / 30
        samples = ()
        if self.imu:
            samples = tuple(MotionSample(stream, np.array(vector, dtype=np.float32), timestamp,
                                         "synthetic", self._frame_number)
                            for stream, vector in (("accel", [0, -9.80665, 0]), ("gyro", [0, 0, 0])))
        grey = rgb.mean(axis=-1).astype(np.uint8) if self.infrared else None
        raw = np.rint(depth / 0.001).astype(np.uint16)
        return RGBDFrame(
            rgb=rgb, depth_m=depth, xyz_m=xyz.astype(np.float32), native_rgb=rgb.copy(),
            depth_raw=raw, native_depth_raw=raw.copy(), infrared_left=grey,
            infrared_right=np.roll(grey, -4, axis=1) if grey is not None else None,
            depth_scale_m=0.001, timestamp_ms=timestamp, timestamp_domain="synthetic",
            frame_number=self._frame_number, motion=samples,
            intrinsics={"width": self.width, "height": self.height, "fx": focal_length,
                        "fy": focal_length, "ppx": self.width / 2, "ppy": self.height / 2,
                        "model": "none", "coeffs": [0.0] * 5},
            metadata={"source": "Synthetic demonstration; IR shift is illustrative only."},
        )

    def stop(self) -> None:
        self._running = False

    def __enter__(self) -> "DemoCamera":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
