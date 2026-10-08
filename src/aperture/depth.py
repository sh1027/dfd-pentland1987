"""Metric depth conversion using the saved blur-radius coefficient K."""

import numpy as np


def depth_from_diameter(diameter_px, *, k, focus_distance_m, branch="far"):
    """Invert diameter = 2*K*abs(1/z - 1/focus).

    diameter_px is in pixels, K in metre-pixels, and focus/depth in metres.
    The default far branch uses the minus sign; near uses the plus sign.
    Negative/nonfinite diameters and nonpositive/nonfinite depths return NaN.
    K and focus must be finite and positive. Scalars and arrays are accepted.
    """
    if not all(np.isfinite(value) and value > 0 for value in (k, focus_distance_m)):
        raise ValueError("k and focus_distance_m must be finite and positive")
    if branch not in ("near", "far"):
        raise ValueError("branch must be 'near' or 'far'")
    diameter = np.asarray(diameter_px, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        offset = diameter / (2 * k)
        inverse = 1 / focus_distance_m + (offset if branch == "near" else -offset)
        valid = np.isfinite(diameter) & (diameter >= 0) & np.isfinite(inverse) & (inverse > 0)
        if branch == "far":
            valid &= diameter < 2 * k / focus_distance_m
        depth = np.divide(1, inverse, out=np.full_like(diameter, np.nan), where=valid)
    return np.where(np.isfinite(depth), depth, np.nan)
