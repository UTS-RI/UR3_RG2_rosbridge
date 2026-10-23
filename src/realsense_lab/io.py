"""Lossless NPZ snapshots, coloured PLY point clouds and timestamped IMU CSV."""

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import MotionSample, PointCloud, RGBDFrame


_ARRAYS = ("rgb", "depth_m", "xyz_m", "native_rgb", "depth_raw", "native_depth_raw",
           "infrared_left", "infrared_right")
_FIELDS = ("depth_scale_m", "timestamp_ms", "timestamp_domain", "frame_number", "alignment",
           "intrinsics", "metadata")


def save_frame(path: str | Path, frame: RGBDFrame) -> Path:
    """Save all available arrays, metadata and IMU snapshots without pickle."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("The filename must end in .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(frame, name) for name in _ARRAYS if getattr(frame, name) is not None}
    details = {name: getattr(frame, name) for name in _FIELDS}
    details["schema_version"] = 2
    details["motion"] = [dict(stream=s.stream, vector=s.vector.tolist(), timestamp_ms=s.timestamp_ms,
                              timestamp_domain=s.timestamp_domain, frame_number=s.frame_number)
                         for s in frame.motion]
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays, details_json=np.array(json.dumps(details)))
    return path


def load_frame(path: str | Path) -> RGBDFrame:
    """Load a snapshot, including the original three-array teaching format."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in _ARRAYS if name in data}
        details = json.loads(str(data["details_json"])) if "details_json" in data else {}
        fields = {name: details[name] for name in _FIELDS if name in details}
        samples = tuple(MotionSample(**dict(item, vector=np.asarray(item["vector"], dtype=np.float32)))
                        for item in details.get("motion", []))
        return RGBDFrame(**arrays, **fields, motion=samples)


def save_pointcloud(path: str | Path, cloud: PointCloud) -> Path:
    """Export XYZ in metres and 8-bit RGB colours as ASCII PLY, e.g. for CloudCompare."""
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError("The filename must end in .ply.")
    path.parent.mkdir(parents=True, exist_ok=True)
    colours = np.rint(np.clip(cloud.colours, 0, 1) * 255).astype(np.uint8)
    header = (
        "ply\nformat ascii 1.0\ncomment XYZ in metres; X right, Y down, Z forwards\n"
        f"element vertex {len(cloud.points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write(header)
        np.savetxt(file, np.column_stack((cloud.points, colours)), fmt="%.7g %.7g %.7g %d %d %d")
    return path


def save_motion_csv(path: str | Path, samples: Iterable[MotionSample]) -> Path:
    """Write each sample's own timestamp, stream, XYZ components and units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["stream", "timestamp_ms", "timestamp_domain", "frame_number", "x", "y", "z", "unit"])
        for sample in samples:
            writer.writerow([sample.stream, sample.timestamp_ms, sample.timestamp_domain,
                             sample.frame_number, *sample.vector, sample.unit])
    return path
