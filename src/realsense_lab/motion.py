"""Read accelerometer and gyroscope samples independently of video rendering."""

from queue import Empty, Full, Queue
from threading import Lock

import numpy as np

from .device import get_sdk
from .types import MotionSample


def motion_sample(frame) -> MotionSample:
    motion = frame.as_motion_frame()
    vector = motion.get_motion_data()
    return MotionSample(
        stream=frame.profile.stream_type().name,
        vector=np.array([vector.x, vector.y, vector.z], dtype=np.float32),
        timestamp_ms=frame.get_timestamp(),
        timestamp_domain=frame.get_frame_timestamp_domain().name,
        frame_number=frame.get_frame_number(),
    )


class MotionReader:
    """Queue IMU samples through an SDK callback, independently of video FPS.

    Experimental in this project. Live capture still needs hardware validation.
    Use separately or alongside a video-only RealSenseCamera on the same device.
    Do not also enable imu=True on that camera: each sensor needs one owner.
    Read promptly and check dropped_samples; the bounded queue drops the oldest
    sample if the consumer falls behind. IMU support is checked by SDK resolution.
    """

    def __init__(self, *, serial: str | None = None, accel_hz: int = 63,
                 gyro_hz: int = 200, queue_size: int = 10000):
        for value in (accel_hz, gyro_hz, queue_size):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("IMU rates and queue_size must be positive integers.")
        self.serial = serial
        self.accel_hz = accel_hz
        self.gyro_hz = gyro_hz
        self._queue = Queue(maxsize=queue_size)
        self._lock = Lock()
        self._pipeline = None
        self._last = {}
        self.dropped_samples = 0
        self._callback_error = None

    def _receive(self, frame):
        try:
            frames = frame.as_frameset() if frame.is_frameset() else [frame]
            for item in frames:
                if not item.is_motion_frame():
                    continue
                sample = motion_sample(item)
                key = (sample.frame_number, sample.timestamp_ms)
                with self._lock:
                    # A synchronised frameset can repeat the latest slower sample.
                    if self._last.get(sample.stream) == key:
                        continue
                    self._last[sample.stream] = key
                    try:
                        self._queue.put_nowait(sample)
                    except Full:
                        try:
                            self._queue.get_nowait()
                            self.dropped_samples += 1
                        except Empty:
                            pass
                        self._queue.put_nowait(sample)
        except Exception as exc:
            self._callback_error = exc

    def start(self):
        if self._pipeline is not None:
            return self
        rs = get_sdk()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.accel_hz)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.gyro_hz)
        self._last.clear()
        self.dropped_samples = 0
        self._callback_error = None
        while not self._queue.empty():
            self._queue.get_nowait()
        pipeline = rs.pipeline()
        try:
            pipeline.start(config, self._receive)
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            if "permission denied" in str(exc).casefold():
                hint = (
                    "On Linux, install the official RealSense access rules, including the "
                    "IMU/HID/IIO rules, then reconnect the camera. This requires administrator "
                    "access. See the Linux setup notes in README.md."
                )
            else:
                hint = "Check supported profiles and close other applications using the motion sensor."
            raise RuntimeError(f"Cannot start IMU streams. {hint} SDK: {exc}") from exc
        self._pipeline = pipeline
        return self

    def read(self, timeout_s: float = 5.0) -> MotionSample:
        if self._pipeline is None:
            raise RuntimeError("Start MotionReader with a 'with' block before reading.")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite.")
        if self._callback_error is not None:
            raise RuntimeError(f"IMU callback failed: {self._callback_error}") from self._callback_error
        try:
            return self._queue.get(timeout=timeout_s)
        except Empty as exc:
            raise TimeoutError("No IMU sample arrived before the timeout.") from exc

    def stop(self):
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
