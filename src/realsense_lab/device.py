"""Discover streams and controls without assuming every model has every feature."""

import json
from pathlib import Path

import numpy as np


def get_sdk():
    """Import the optional SDK only for hardware or recording playback."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            'RealSense SDK is missing. Run: python -m pip install -e ".[camera]". '
            "The basic examples also work without it using --demo."
        ) from exc
    return rs


def stream_key(profile) -> str:
    return f"{profile.stream_type().name}:{profile.stream_index()}"


def intrinsics_dict(intrinsics) -> dict:
    """Convert video intrinsics to JSON-compatible values."""
    return {"width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
            "model": intrinsics.model.name, "coeffs": list(intrinsics.coeffs)}


def profile_dict(profile) -> dict:
    result = {"stream": profile.stream_type().name, "index": profile.stream_index(),
              "format": profile.format().name, "fps": profile.fps()}
    if profile.is_video_stream_profile():
        video = profile.as_video_stream_profile()
        result.update(width=video.width(), height=video.height())
    return result


def calibration_report(profile) -> dict:
    """Active intrinsics and source-to-target extrinsics; rotations are column-major."""
    streams = list(profile.get_streams())
    result = {"streams": {}, "extrinsics": {}, "rotation_order": "column-major",
              "translation_unit": "metres"}
    for stream in streams:
        entry = profile_dict(stream)
        if stream.is_video_stream_profile():
            entry["intrinsics"] = intrinsics_dict(stream.as_video_stream_profile().get_intrinsics())
        elif stream.is_motion_stream_profile():
            intr = stream.as_motion_stream_profile().get_motion_intrinsics()
            entry["motion_intrinsics"] = {
                "data": np.asarray(intr.data).tolist(), "noise_variances": list(intr.noise_variances),
                "bias_variances": list(intr.bias_variances)}
        result["streams"][stream_key(stream)] = entry
        for target in streams:
            key = f"{stream_key(stream)} -> {stream_key(target)}"
            try:
                extr = stream.get_extrinsics_to(target)
                result["extrinsics"][key] = {
                    "rotation": list(extr.rotation), "translation": list(extr.translation)}
            except RuntimeError as exc:
                result["extrinsics"][key] = {"error": str(exc)}
    return result


class CameraControls:
    """Every standard sensor option, with support, range and read-only checks.

    Select 'depth', 'colour', 'motion' or an exact SDK sensor name.
    Option names retain SDK spelling, e.g. 'enable_auto_exposure'.
    Some settings require stopped streams; the SDK reports these restrictions.
    """

    def __init__(self, device):
        self.device = device
        self.rs = get_sdk()

    def sensor(self, name: str):
        for sensor in self.device.query_sensors():
            if sensor.get_info(self.rs.camera_info.name) == name:
                return sensor
            if name == "depth" and sensor.is_depth_sensor():
                return sensor
            if name in ("colour", "color") and sensor.is_color_sensor():
                return sensor
            if name == "motion" and sensor.is_motion_sensor():
                return sensor
        raise ValueError(f"Sensor {name!r} is not available on this device.")

    def _option(self, sensor, name: str):
        option = self.rs.option.__members__.get(name)
        if option is None or not sensor.supports(option):
            raise ValueError(f"Option {name!r} is not supported by this sensor.")
        return option

    def options(self, sensor: str = "depth") -> dict:
        """List values, ranges, descriptions, named choices and read-only status."""
        handle = self.sensor(sensor)
        result = {}
        for option in handle.get_supported_options():
            try:
                limits = handle.get_option_range(option)
                item = {"min": limits.min, "max": limits.max, "step": limits.step,
                        "default": limits.default, "read_only": handle.is_option_read_only(option),
                        "description": handle.get_option_description(option)}
                try:
                    item["value"] = handle.get_option(option)
                except RuntimeError as exc:
                    item["value_error"] = str(exc)
                if 0 < limits.step and (limits.max - limits.min) / limits.step <= 32:
                    choices = {}
                    for index in range(round((limits.max - limits.min) / limits.step) + 1):
                        value = limits.min + index * limits.step
                        label = handle.get_option_value_description(option, value)
                        if label:
                            choices[str(value)] = label
                    if choices:
                        item["choices"] = choices
                result[option.name] = item
            except RuntimeError as exc:
                result[option.name] = {"error": str(exc)}
        return result

    def get_option(self, sensor: str, name: str) -> float:
        handle = self.sensor(sensor)
        return handle.get_option(self._option(handle, name))

    def set_option(self, sensor: str, name: str, value: float) -> None:
        """Set a supported option without clamping or guessing its units."""
        handle = self.sensor(sensor)
        option = self._option(handle, name)
        if handle.is_option_read_only(option):
            raise ValueError(f"Option {name!r} is read-only.")
        limits = handle.get_option_range(option)
        if not np.isfinite(value) or not limits.min <= value <= limits.max:
            raise ValueError(f"{name} must be between {limits.min} and {limits.max}.")
        if limits.step > 0:
            steps = (value - limits.min) / limits.step
            if not np.isclose(steps, round(steps), rtol=0, atol=1e-4):
                raise ValueError(f"{name} must use increments of {limits.step} from {limits.min}.")
        handle.set_option(option, float(value))

    def get_roi(self, sensor: str = "depth") -> tuple[int, int, int, int]:
        """Auto-exposure ROI as inclusive (left, top, right, bottom)."""
        handle = self.sensor(sensor)
        if not handle.is_roi_sensor():
            raise ValueError(f"Sensor {sensor!r} does not support an exposure ROI.")
        roi = handle.as_roi_sensor().get_region_of_interest()
        return roi.min_x, roi.min_y, roi.max_x, roi.max_y

    def set_roi(self, sensor: str, left: int, top: int, right: int, bottom: int) -> None:
        """Set an exposure ROI in native sensor coordinates (inclusive bounds)."""
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (left, top, right, bottom)):
            raise ValueError("ROI coordinates must be integers.")
        if not (0 <= left < right and 0 <= top < bottom):
            raise ValueError("ROI must satisfy 0 <= left < right and 0 <= top < bottom.")
        handle = self.sensor(sensor)
        if not handle.is_roi_sensor():
            raise ValueError(f"Sensor {sensor!r} does not support an exposure ROI.")
        roi = self.rs.region_of_interest()
        roi.min_x, roi.min_y, roi.max_x, roi.max_y = left, top, right, bottom
        handle.as_roi_sensor().set_region_of_interest(roi)

    def advanced_settings_json(self) -> str:
        """Read D400 advanced settings when advanced mode is already enabled."""
        advanced = self.rs.rs400_advanced_mode(self.device)
        if not advanced.is_enabled():
            raise RuntimeError("Advanced mode is disabled; export does not toggle or restart the device.")
        return advanced.serialize_json()


def describe_device(device) -> dict:
    rs = get_sdk()
    info = {}
    for name, field in rs.camera_info.__members__.items():
        if name != "count" and device.supports(field):
            info[name] = device.get_info(field)
    controls = CameraControls(device)
    sensors = []
    for sensor in device.query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        sensors.append({"name": name, "exposure_roi": sensor.is_roi_sensor(),
                        "profiles": [profile_dict(p) for p in sensor.get_stream_profiles()],
                        "options": controls.options(name)})
    return {"info": info, "sensors": sensors}


def inspect_devices() -> list[dict]:
    """Enumerate connected devices, supported stream profiles and sensor options."""
    rs = get_sdk()
    context = rs.context()
    return [describe_device(device) for device in context.query_devices()]


def save_json(path: str | Path, data: dict | list) -> Path:
    """Write a readable JSON report using UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
