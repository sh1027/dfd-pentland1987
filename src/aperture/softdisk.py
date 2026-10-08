"""Discrete circular PSF used by the official BokehMe classical renderer.

Reference: https://github.com/JuewenPeng/BokehMe/blob/main/classical_renderer/scatter.py
The renderer accepts a signed *radius*. This module's public API uses diameter
in pixels, consistently with geometric circle-of-confusion calibration.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


def kernel(diameter_px: float) -> NDArray[np.float64]:
    """Return the normalized, odd-sized BokehMe softdisk kernel."""
    if not np.isfinite(diameter_px) or diameter_px < 0:
        raise ValueError("diameter_px must be finite and non-negative.")
    radius = float(diameter_px) / 2
    extent = int(radius) + 1
    offsets = np.arange(-extent, extent + 1, dtype=np.float64)
    x, y = offsets[None, :], offsets[:, None]
    distance = np.sqrt(x**2 + y**2)
    weights = (0.5 + 0.5 * np.tanh(4 * (radius - distance))) / (radius**2 + 0.2)
    return weights / np.sum(weights)


def diameter_from_variance(variance_px2, max_diameter_px):
    """Dataset adaptation: Gaussian axis variance to an equal-moment softdisk.

    Invert sum(h*x²), using PSF-only samples, not depth truth or scene fitting.
    Equal second moments approximate the paper's Gaussian spectral model at
    low frequencies; they do not make a softdisk OTF Gaussian at all frequencies.
    """
    if not np.isfinite(max_diameter_px) or max_diameter_px <= 0:
        raise ValueError("max_diameter_px must be finite and positive")
    diameters, moments = _variance_table(float(max_diameter_px))
    values = np.asarray(variance_px2, dtype=float)
    valid = np.isfinite(values) & (values >= moments[0]) & (values <= moments[-1])
    return np.where(valid, np.interp(values, moments, diameters), np.nan)


@lru_cache(maxsize=8)
def _variance_table(max_diameter_px):
    """Reuse the PSF-only calibration for repeated conversions of the same range."""
    diameters = np.linspace(0, max_diameter_px, 4097)
    moments = np.empty_like(diameters)
    for index, diameter in enumerate(diameters):
        psf = kernel(float(diameter))
        x = np.arange(psf.shape[1]) - psf.shape[1] // 2
        moments[index] = np.sum(psf * x[None, :] ** 2)
    if np.any(np.diff(moments) <= 0):
        raise ValueError("PSF moment calibration must be strictly increasing")
    diameters.flags.writeable = False
    moments.flags.writeable = False
    return diameters, moments
