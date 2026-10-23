"""SDK depth post-processing, created afresh for each camera session."""

from dataclasses import dataclass

import numpy as np

from .device import get_sdk


@dataclass(frozen=True)
class DepthFilters:
    """Decimation -> threshold -> disparity -> spatial -> temporal -> depth -> holes.

    No filters run unless this configuration is passed to RealSenseCamera.
    Hole filling estimates missing measurements and is therefore opt-in.
    Temporal state belongs to one continuous camera session.
    """

    decimation: int = 1
    min_depth_m: float = 0.2
    max_depth_m: float = 4.0
    spatial: bool = True
    temporal: bool = True
    hole_filling: bool = False

    def __post_init__(self):
        if not isinstance(self.decimation, int) or isinstance(self.decimation, bool) or not 1 <= self.decimation <= 8:
            raise ValueError("decimation must be an integer from 1 to 8.")
        if not np.isfinite([self.min_depth_m, self.max_depth_m]).all() or not 0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError("Depth limits must be finite and satisfy 0 <= minimum < maximum.")

    def build(self):
        rs = get_sdk()
        filters = []
        if self.decimation > 1:
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, self.decimation)
            filters.append(decimation)
        filters.append(rs.threshold_filter(self.min_depth_m, self.max_depth_m))
        if self.spatial or self.temporal:
            filters.append(rs.disparity_transform(True))
            if self.spatial:
                filters.append(rs.spatial_filter())
            if self.temporal:
                filters.append(rs.temporal_filter())
            filters.append(rs.disparity_transform(False))
        if self.hole_filling:
            filters.append(rs.hole_filling_filter())
        return filters
