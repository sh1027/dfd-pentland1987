"""Section II.B: comparison across differing apertures (Pentland, TPAMI 1987).

Gaussian PSFs give log(A1/A2) = c + 2*pi²*(sigma2²-sigma1²)*frequency².
Frequencies are cycles/pixel and sigma is the one-axis Gaussian standard
deviation in pixels. c absorbs an exposure multiplier, not a depth offset.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ApertureResult:
    variance_difference_px2: np.ndarray
    fit_rmse: np.ndarray
    valid: np.ndarray


def _gray(image):
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image @ np.array([0.2126, 0.7152, 0.0722])
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D grayscale or H x W x 3 linear RGB image")
    return image


def fit_variance_difference(
    amplitude1, amplitude2, frequencies, *, amplitude_floor=1e-3, max_fit_error=0.15
):
    """Eq. (9), regression of log amplitude ratio against 2*pi²*frequency².

    Amplitudes (..., bins) are angular means at equal radial frequencies.
    A free intercept compensates brightness differences (paper footnote 6).
    Weights=min(A1,A2)² favor bins with signal in both images. Bins below a
    relative amplitude floor are omitted; at least four bins are required.
    Return (sigma2²-sigma1², log-amplitude RMSE, valid). Both signs are allowed.
    """
    first, second = np.asarray(amplitude1, dtype=float), np.asarray(amplitude2, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 4 or first.ndim < 1:
        raise ValueError("At least four frequencies and amplitude arrays are required")
    if first.shape != second.shape or first.shape[-1] != frequencies.size:
        raise ValueError("Amplitude shapes must match each other and the frequencies")
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (first, second)):
        raise ValueError("Amplitudes must be finite and nonnegative")
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("Frequencies must be finite and positive")
    if not 0 < amplitude_floor < 1 or not np.isfinite(max_fit_error) or max_fit_error <= 0:
        raise ValueError("Require 0 < amplitude_floor < 1 and positive finite max_fit_error")
    used = (first > amplitude_floor * first.max(axis=-1, keepdims=True)) & (
        second > amplitude_floor * second.max(axis=-1, keepdims=True)
    )
    signal = np.where(used, np.minimum(first, second), 0)
    scale = signal.max(axis=-1, keepdims=True)
    # Normalize before squaring to avoid overflow/underflow in amplitude units.
    weights = np.divide(signal, scale, out=np.zeros_like(signal), where=scale > 0) ** 2
    total = weights.sum(axis=-1)
    safe_total = np.where(total > 0, total, 1)
    x = 2 * np.pi**2 * frequencies**2
    y = np.log(np.where(used, first, 1)) - np.log(np.where(used, second, 1))
    xmean = (weights * x).sum(axis=-1) / safe_total
    ymean = (weights * y).sum(axis=-1) / safe_total
    dx, dy = x - xmean[..., None], y - ymean[..., None]
    denominator = (weights * dx**2).sum(axis=-1)
    slope = np.divide(
        (weights * dx * dy).sum(axis=-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    residual = dy - slope[..., None] * dx
    error = np.sqrt((weights * residual**2).sum(axis=-1) / safe_total)
    valid = (
        (used.sum(axis=-1) >= 4)
        & (denominator > np.finfo(float).eps)
        & np.isfinite(slope)
        & (error <= max_fit_error)
    )
    return np.where(valid, slope, np.nan), np.where(valid, error, np.nan), valid


def _frequency_bins(n, frequency_min, frequency_max):
    """Group equal-radius FFT samples for the angular amplitude mean in Eq. (9)."""
    frequencies = np.rint(np.fft.fftfreq(n) * n).astype(int)
    squared_radius = (frequencies[:, None] ** 2 + frequencies[None, :] ** 2).ravel()
    radial = np.sqrt(squared_radius) / n
    indices = np.flatnonzero((radial >= frequency_min) & (radial <= frequency_max))
    indices = indices[np.argsort(squared_radius[indices])]
    radius, starts, counts = np.unique(
        squared_radius[indices], return_index=True, return_counts=True
    )
    if radius.size < 4:
        raise ValueError("Patch/frequency band must include at least four radial frequencies")
    return np.sqrt(radius) / n, (indices, starts, counts)


def _radial_amplitudes(patches, window, bins, min_texture):
    """Normalize brightness, taper patches, and average Fourier amplitudes."""
    indices, starts, counts = bins
    brightness = (patches * window).sum(axis=(-2, -1))
    centered = patches - brightness[..., None, None]
    contrast = np.sqrt((centered**2 * window).sum(axis=(-2, -1)))
    normalized = np.divide(
        centered,
        brightness[..., None, None],
        out=centered,
        where=brightness[..., None, None] > 0,
    )
    # centered is an owned buffer; reuse it without modifying the input patches.
    normalized[brightness <= 0] = 0
    normalized *= window
    amplitudes = np.abs(np.fft.fft2(normalized, axes=(-2, -1)))
    samples = amplitudes.reshape(-1, window.size)[:, indices]
    amplitudes = np.add.reduceat(samples, starts, axis=-1) / counts
    return amplitudes, ((brightness > 0) & (contrast > min_texture * brightness)).ravel()


def estimate_aperture_pair(
    image1,
    image2,
    *,
    patch_size=33,
    frequency_min=0.05,
    frequency_max=0.25,
    min_texture=0.01,
    amplitude_floor=1e-3,
    max_fit_error=0.15,
):
    """Local Fourier implementation of Eq. (9); image1 is the reference.

    Use Gaussian-tapered patches as suggested after Eq. (7). The taper's sigma
    is patch_size/6. Remove its weighted DC and normalize patch brightness.
    Equal-radius FFT samples are averaged in *amplitude*, not power. Finite
    windows make the Gaussian model approximate; fit error is not a guarantee
    of depth accuracy. Whole patch borders and low-texture patches are invalid.
    No depth truth or scene-specific parameter fitting is used.
    """
    first, second = _gray(image1), _gray(image2)
    if first.shape != second.shape:
        raise ValueError("Registered image shapes must match")
    if (
        isinstance(patch_size, bool)
        or not isinstance(patch_size, (int, np.integer))
        or patch_size < 9
        or patch_size % 2 != 1
        or patch_size > min(first.shape)
    ):
        raise ValueError("patch_size must be odd, >= 9, and fit inside both images")
    if not 0 < frequency_min < frequency_max <= 0.5:
        raise ValueError("Require 0 < frequency_min < frequency_max <= .5 cycles/pixel")
    if not np.isfinite(min_texture) or min_texture < 0:
        raise ValueError("min_texture must be finite and nonnegative")
    n = int(patch_size)
    frequencies, bins = _frequency_bins(n, frequency_min, frequency_max)
    positions = np.arange(n) - n // 2
    window = np.exp(-(positions[:, None] ** 2 + positions[None, :] ** 2) / (2 * (n / 6) ** 2))
    window /= window.sum()
    patches1 = np.lib.stride_tricks.sliding_window_view(first, (n, n))
    patches2 = np.lib.stride_tricks.sliding_window_view(second, (n, n))
    width = patches1.shape[1]
    variance = np.full(first.shape, np.nan)
    error = np.full(first.shape, np.nan)
    valid = np.zeros(first.shape, dtype=bool)
    margin = n // 2

    # Process a few patch rows at a time without copying the full H*W*N*N array.
    for row in range(0, patches1.shape[0], 4):
        stop = min(row + 4, patches1.shape[0])
        a1, textured1 = _radial_amplitudes(patches1[row:stop], window, bins, min_texture)
        a2, textured2 = _radial_amplitudes(patches2[row:stop], window, bins, min_texture)
        values, errors, accepted = fit_variance_difference(
            a1, a2, frequencies, amplitude_floor=amplitude_floor, max_fit_error=max_fit_error
        )
        accepted &= textured1 & textured2
        region = (slice(row + margin, stop + margin), slice(margin, -margin))
        variance[region] = np.where(accepted, values, np.nan).reshape(-1, width)
        error[region] = np.where(accepted, errors, np.nan).reshape(-1, width)
        valid[region] = accepted.reshape(-1, width)
    return ApertureResult(variance, error, valid)


def focal_disparity(image1, image2, *, window_size=4, brightness_sigma=2.0, min_contrast=1e-3):
    """Section III.B approximation: normalized local high-frequency content ratio.

    Returns blurry/reference ratio and valid mask. The paper leaves the exact
    aggregation unspecified; use absolute four-neighbor Laplacian responses
    to avoid sign cancellation, a 4x4 average, and Gaussian local brightness.
    This disparity alone has no universal metric calibration; use Eq. (9)
    above for Gaussian variance estimates. Values >1 are retained, not clipped.
    """
    first, second = _gray(image1), _gray(image2)
    if first.shape != second.shape:
        raise ValueError("Registered image shapes must match")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, (int, np.integer))
        or window_size < 1
        or not np.isfinite(brightness_sigma)
        or brightness_sigma <= 0
        or not np.isfinite(min_contrast)
        or min_contrast < 0
    ):
        raise ValueError("Invalid aggregation, brightness scale, or contrast threshold")

    def content(image):
        laplacian = np.abs(ndimage.laplace(image, mode="reflect"))
        high = ndimage.uniform_filter(laplacian, size=window_size, mode="reflect")
        brightness = ndimage.gaussian_filter(image, brightness_sigma, mode="reflect")
        return np.divide(
            high, brightness, out=np.zeros_like(high), where=brightness > 0
        ), brightness

    (a1, b1), (a2, b2) = content(first), content(second)
    valid = (a1 > min_contrast) & (b1 > 0) & (b2 > 0)
    border = max(1 + window_size // 2, int(np.ceil(4 * brightness_sigma)))
    valid[:border] = valid[-border:] = False
    valid[:, :border] = valid[:, -border:] = False
    ratio = np.divide(a2, a1, out=np.full_like(a1, np.nan), where=valid)
    return ratio, valid
