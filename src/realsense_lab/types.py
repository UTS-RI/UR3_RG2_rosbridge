"""NumPy data structures shared by the camera, processing and examples."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MotionSample:
    """An IMU sample: accel in m/s² (including gravity), or gyro in rad/s."""

    stream: str
    vector: NDArray[np.float32]
    timestamp_ms: float
    timestamp_domain: str
    frame_number: int

    @property
    def unit(self) -> str:
        return "m/s²" if self.stream == "accel" else "rad/s"


@dataclass
class RGBDFrame:
    """Registered RGB, metric depth and calibrated XYZ, plus optional native data.

    rgb: (H, W, 3), uint8, RGB 0..255, registered to depth_m.
    depth_m: (H, W), float32, Z-depth in metres; 0 is invalid.
    xyz_m: (H, W, 3), float32, metres; X right, Y down, Z forwards.
    Native RGB/depth/IR arrays retain their original sensor pixel grids.
    motion contains only samples present in this video frameset; use
    MotionReader to collect IMU samples independently at their native rates.
    """

    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    xyz_m: NDArray[np.float32]
    native_rgb: NDArray[np.uint8] | None = None
    depth_raw: NDArray[np.uint16] | None = None
    native_depth_raw: NDArray[np.uint16] | None = None
    infrared_left: NDArray[np.uint8] | None = None
    infrared_right: NDArray[np.uint8] | None = None
    depth_scale_m: float | None = None
    timestamp_ms: float | None = None
    timestamp_domain: str | None = None
    frame_number: int | None = None
    alignment: str = "depth"
    intrinsics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    motion: tuple[MotionSample, ...] = ()

    @property
    def valid_depth(self) -> NDArray[np.bool_]:
        """True for finite, positive depth measurements."""
        return np.isfinite(self.depth_m) & (self.depth_m > 0)


@dataclass
class PointCloud:
    """Filtered points (N, 3) in metres and RGB colours (N, 3) in 0..1."""

    points: NDArray[np.float32]
    colors: NDArray[np.float32]

    @property
    def colours(self) -> NDArray[np.float32]:
        """Australian English alias; colors remains compatible with older examples."""
        return self.colors
