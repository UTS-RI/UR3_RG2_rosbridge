"""D435i video, infrared, IMU snapshots, controls, filtering and recording playback."""

from pathlib import Path

import numpy as np

from .device import CameraControls, calibration_report, get_sdk, intrinsics_dict, stream_key
from .filters import DepthFilters
from .motion import motion_sample
from .types import RGBDFrame


def _frame_details(frame, rs, include_metadata: bool) -> dict:
    result = {"timestamp_ms": frame.get_timestamp(),
              "timestamp_domain": frame.get_frame_timestamp_domain().name,
              "frame_number": frame.get_frame_number()}
    if include_metadata:
        values = {}
        for name, field in rs.frame_metadata_value.__members__.items():
            if name != "count" and frame.supports_frame_metadata(field):
                try:
                    values[name] = frame.get_frame_metadata(field)
                except RuntimeError:
                    pass  # Metadata support can vary between individual frames.
        result["sdk_metadata"] = values
    return result


def _texture_rgb(colour: np.ndarray, texture: np.ndarray, shape: tuple) -> np.ndarray:
    """Sample valid SDK UV coordinates without smearing out-of-view edge colours."""
    uv = texture.reshape(*shape, 2)
    valid = np.isfinite(uv).all(axis=-1) & (uv >= 0).all(axis=-1) & (uv < 1).all(axis=-1)
    safe = np.where(valid[..., None], uv, 0)
    height, width = colour.shape[:2]
    x = np.minimum((safe[..., 0] * width + 0.5).astype(int), width - 1)
    y = np.minimum((safe[..., 1] * height + 0.5).astype(int), height - 1)
    rgb = colour[y, x].copy()
    rgb[~valid] = 0
    return rgb


class RealSenseCamera:
    """Capture registered NumPy arrays; optional features are explicitly enabled.

    alignment='depth' registers colour to the depth sensor; 'colour' registers
    depth to the colour sensor. Native images are retained in either case.
    Use infrared=True and imu=True for all six D435i streams. IMU entries in
    capture() are video-rate snapshots; use MotionReader for native-rate logging.
    The SDK validates the requested stream combination on start.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, *,
                 serial: str | None = None, warmup_frames: int = 15, timeout_ms: int = 5000,
                 colour_width: int | None = None, colour_height: int | None = None,
                 colour_fps: int | None = None, alignment: str = "depth",
                 infrared: bool = False, imu: bool = False, accel_hz: int = 63,
                 gyro_hz: int = 200, filters: DepthFilters | None = None,
                 record_to: str | Path | None = None, playback_from: str | Path | None = None,
                 repeat_playback: bool = False, metadata: bool = True):
        self.width, self.height, self.fps = width, height, fps
        self.colour_width = width if colour_width is None else colour_width
        self.colour_height = height if colour_height is None else colour_height
        self.colour_fps = fps if colour_fps is None else colour_fps
        for name, value in {"width": width, "height": height, "fps": fps,
                            "colour_width": self.colour_width, "colour_height": self.colour_height,
                            "colour_fps": self.colour_fps, "timeout_ms": timeout_ms,
                            "accel_hz": accel_hz, "gyro_hz": gyro_hz}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(warmup_frames, int) or isinstance(warmup_frames, bool) or warmup_frames < 0:
            raise ValueError("warmup_frames must be an integer >= 0.")
        if alignment not in ("depth", "colour", "color"):
            raise ValueError("alignment must be 'depth' or 'colour'.")
        if record_to is not None and playback_from is not None:
            raise ValueError("Recording and playback cannot be enabled together.")
        if playback_from is not None and serial is not None:
            raise ValueError("Select either a live serial number or a playback file.")
        self.serial, self.warmup_frames, self.timeout_ms = serial, warmup_frames, timeout_ms
        self.alignment = "colour" if alignment == "color" else alignment
        self.infrared, self.imu = infrared, imu
        self.accel_hz, self.gyro_hz = accel_hz, gyro_hz
        self.filter_config = filters
        self.record_to = Path(record_to) if record_to is not None else None
        self.playback_from = Path(playback_from) if playback_from is not None else None
        self.repeat_playback = repeat_playback
        self.include_metadata = metadata
        self._pipeline = self._profile = self._rs = None
        self._align = self._pointcloud = self._last_frames = None
        self._filters = []

    @property
    def profile(self):
        """Active SDK pipeline profile, for features beyond the teaching helpers."""
        if self._profile is None:
            raise RuntimeError("Start the camera using 'with RealSenseCamera() as camera:'.")
        return self._profile

    @property
    def device(self):
        return self.profile.get_device()

    @property
    def controls(self) -> CameraControls:
        return CameraControls(self.device)

    @property
    def calibration(self) -> dict:
        return calibration_report(self.profile)

    @property
    def playback(self):
        """SDK playback handle: pause(), resume(), seek(timedelta), set_real_time()."""
        if self.playback_from is None:
            raise RuntimeError("This camera is not playing a recording.")
        return self.device.as_playback()

    @property
    def recorder(self):
        """SDK recording handle, including pause() and resume()."""
        if self.record_to is None:
            raise RuntimeError("This camera is not recording.")
        return self.device.as_recorder()

    @property
    def last_frames(self):
        """Most recent native SDK frameset; capture() must have been called first."""
        if self._last_frames is None:
            raise RuntimeError("Call capture() before accessing last_frames.")
        return self._last_frames

    def start(self) -> "RealSenseCamera":
        if self._pipeline is not None:
            return self
        if self.playback_from is not None and not self.playback_from.is_file():
            raise FileNotFoundError(self.playback_from)
        if self.record_to is not None:
            if self.record_to.exists():
                raise FileExistsError(f"Recording already exists: {self.record_to}")
            self.record_to.parent.mkdir(parents=True, exist_ok=True)
        rs = get_sdk()
        config = rs.config()
        replay = self.playback_from is not None
        if replay:
            config.enable_device_from_file(str(self.playback_from.resolve()), self.repeat_playback)
        elif self.serial is not None:
            config.enable_device(self.serial)
        # Zero dimensions/rate let playback use the profiles stored in the file.
        config.enable_stream(rs.stream.depth, 0 if replay else self.width,
                             0 if replay else self.height, rs.format.z16, 0 if replay else self.fps)
        config.enable_stream(rs.stream.color, 0 if replay else self.colour_width,
                             0 if replay else self.colour_height, rs.format.rgb8, 0 if replay else self.colour_fps)
        if self.infrared:
            for index in (1, 2):
                config.enable_stream(rs.stream.infrared, index, 0 if replay else self.width,
                                     0 if replay else self.height, rs.format.y8, 0 if replay else self.fps)
        if self.imu:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 0 if replay else self.accel_hz)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 0 if replay else self.gyro_hz)
        if self.record_to is not None:
            config.enable_record_to_file(str(self.record_to.resolve()))
        pipeline = None
        try:
            pipeline = rs.pipeline()
            profile = pipeline.start(config)
        except RuntimeError as exc:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass  # start() may have failed before any stream was opened.
            raise RuntimeError(
                "Cannot start RealSense. Check USB access, close other camera applications, "
                "and check the requested streams in examples/07_inspect_device.py. "
                f"SDK: {exc}"
            ) from exc
        self._pipeline, self._profile, self._rs = pipeline, profile, rs
        try:
            self._align = rs.align(rs.stream.depth if self.alignment == "depth" else rs.stream.color)
            self._pointcloud = rs.pointcloud()
            self._filters = [] if self.filter_config is None else self.filter_config.build()
            if replay:
                self.playback.set_real_time(False)
            else:
                for _ in range(self.warmup_frames):
                    pipeline.wait_for_frames(self.timeout_ms)
        except BaseException:
            self.stop()
            raise
        return self

    def _next_frames(self):
        if self._pipeline is None:
            raise RuntimeError("Start the camera using 'with RealSenseCamera() as camera:'.")
        if self.playback_from is not None:
            frames = self._pipeline.poll_for_frames()
            if frames:
                return frames
            if self.playback.current_status() == self._rs.playback_status.stopped:
                raise EOFError("End of recording.")
        try:
            return self._pipeline.wait_for_frames(self.timeout_ms)
        except RuntimeError as exc:
            if self.playback_from is not None and self.playback.current_status() == self._rs.playback_status.stopped:
                raise EOFError("End of recording.") from exc
            raise RuntimeError(f"No camera frames received: {exc}") from exc

    def capture(self) -> RGBDFrame:
        """Read owned arrays in metres, native images and timestamped metadata."""
        frames = self._next_frames()
        native_depth, native_colour = frames.get_depth_frame(), frames.get_color_frame()
        if not native_depth or not native_colour:
            raise RuntimeError("The frameset does not contain a complete RGB/depth pair.")
        self._last_frames = frames
        aligned = self._align.process(frames)
        depth, colour = aligned.get_depth_frame(), aligned.get_color_frame()
        for operation in self._filters:
            depth = operation.process(depth).as_depth_frame()
        depth_raw = np.asanyarray(depth.get_data()).copy()
        scale = depth.get_units()
        depth_m = depth_raw.astype(np.float32) * scale
        rgb = np.asanyarray(colour.get_data()).copy()
        # Decimation changes the depth grid; use calibrated UV mapping for colour.
        if rgb.shape[:2] != depth_m.shape:
            self._pointcloud.map_to(colour)
        sdk_points = self._pointcloud.calculate(depth)
        xyz_m = np.asanyarray(sdk_points.get_vertices()).view(np.float32).reshape(*depth_m.shape, 3).copy()
        if rgb.shape[:2] != depth_m.shape:
            texture = np.asanyarray(sdk_points.get_texture_coordinates()).view(np.float32)
            rgb = _texture_rgb(rgb, texture, depth_m.shape)
        infrared = []
        for index in (1, 2):
            ir = frames.get_infrared_frame(index) if self.infrared else None
            infrared.append(np.asanyarray(ir.get_data()).copy() if ir else None)
        details = {stream_key(f.profile): _frame_details(f, self._rs, self.include_metadata) for f in frames}
        stamp = _frame_details(native_depth, self._rs, False)
        motion = tuple(motion_sample(f) for f in frames if f.is_motion_frame()) if self.imu else ()
        return RGBDFrame(
            rgb=rgb, depth_m=depth_m, xyz_m=xyz_m,
            native_rgb=np.asanyarray(native_colour.get_data()).copy(), depth_raw=depth_raw,
            native_depth_raw=np.asanyarray(native_depth.get_data()).copy(),
            infrared_left=infrared[0], infrared_right=infrared[1], depth_scale_m=scale,
            timestamp_ms=stamp["timestamp_ms"], timestamp_domain=stamp["timestamp_domain"],
            frame_number=stamp["frame_number"], alignment=self.alignment,
            intrinsics=intrinsics_dict(depth.profile.as_video_stream_profile().get_intrinsics()),
            metadata=details, motion=motion,
        )

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._profile = self._last_frames = self._align = self._pointcloud = None
        self._filters = []
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> "RealSenseCamera":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
