"""Public helpers for the examples and students' own projects."""

from .camera import RealSenseCamera
from .demo import DemoCamera
from .device import CameraControls, inspect_devices, save_json
from .filters import DepthFilters
from .io import load_frame, save_frame, save_motion_csv, save_pointcloud
from .motion import MotionReader
from .processing import depth_heatmap, depth_overlay, distance_between, make_pointcloud, point_at
from .types import MotionSample, PointCloud, RGBDFrame
from .visualisation import show_pointcloud, show_rgbd

__all__ = [
    "RealSenseCamera", "DemoCamera", "RGBDFrame", "PointCloud",
    "depth_heatmap", "depth_overlay", "make_pointcloud",
    "show_rgbd", "show_pointcloud", "save_frame", "load_frame", "save_pointcloud",
    "CameraControls", "inspect_devices", "save_json", "DepthFilters", "MotionReader",
    "MotionSample", "save_motion_csv", "point_at", "distance_between",
]
