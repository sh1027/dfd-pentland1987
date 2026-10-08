import numpy as np
import pytest
from scipy import ndimage

from aperture import (
    estimate_aperture_pair,
    fit_variance_difference,
    focal_disparity,
)
from aperture.softdisk import diameter_from_variance, kernel


def test_gaussian_amplitude_equation_and_exposure():
    frequency = np.linspace(0.03, 0.25, 21)
    scene = np.exp(-frequency) * (1 + 0.3 * np.cos(31 * frequency))
    first = 4 * scene * np.exp(-2 * np.pi**2 * 0.8**2 * frequency**2)
    second = 0.5 * scene * np.exp(-2 * np.pi**2 * 2.1**2 * frequency**2)
    delta, error, valid = fit_variance_difference(first, second, frequency)
    assert valid
    assert delta == pytest.approx(2.1**2 - 0.8**2, abs=1e-12)
    assert error < 1e-12
    reverse, _, accepted = fit_variance_difference(second, first, frequency)
    assert accepted and reverse == pytest.approx(-delta)


def test_empty_spectrum_and_model_inconsistency():
    frequencies = np.linspace(0.04, 0.25, 12)
    variance, _, valid = fit_variance_difference(np.zeros(12), np.zeros(12), frequencies)
    assert not valid and np.isnan(variance)
    inconsistent = np.tile([0.2, 0.8], 6)
    variance, _, valid = fit_variance_difference(
        np.ones(12), inconsistent, frequencies, max_fit_error=0.01
    )
    assert not valid and np.isnan(variance)


def test_local_gaussian_plane_and_exposure_invariance():
    rng = np.random.default_rng(43)
    image = 0.5 + 0.05 * rng.normal(size=(128, 128))
    blurred = ndimage.gaussian_filter(image, 1.25, mode="reflect")
    result = estimate_aperture_pair(image, blurred, patch_size=65)
    interior = np.s_[40:-40, 40:-40]
    assert result.valid[interior].mean() > 0.8
    estimate = np.nanmedian(result.variance_difference_px2[interior])
    assert estimate == pytest.approx(1.25**2, rel=0.15)
    scaled = estimate_aperture_pair(3 * image, 7 * blurred, patch_size=65)
    np.testing.assert_allclose(
        scaled.variance_difference_px2, result.variance_difference_px2, rtol=1e-10, equal_nan=True
    )
    assert not result.valid[:32].any()


def test_flat_images_and_parameter_validation():
    flat = np.ones((50, 50))
    assert not estimate_aperture_pair(flat, flat).valid.any()
    with pytest.raises(ValueError):
        estimate_aperture_pair(flat, flat, patch_size=20)
    with pytest.raises(ValueError):
        estimate_aperture_pair(flat, flat[:40])


def test_normalized_laplacian_disparity():
    x = np.arange(96)[None, :]
    image = np.broadcast_to(1 + 0.3 * np.cos(0.4 * x), (64, 96))
    attenuation = np.exp(-(1.2**2) * 0.4**2 / 2)
    blurred = 1 + (image - 1) * attenuation
    ratio, valid = focal_disparity(image, blurred)
    scaled, accepted = focal_disparity(4 * image, 3 * blurred)
    np.testing.assert_allclose(ratio, scaled, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(valid, accepted)
    assert np.nanmedian(ratio[12:-12, 12:-12]) < 1
    assert not focal_disparity(np.ones((50, 50)), np.ones((50, 50)))[1].any()


def test_softdisk_uses_axis_variance_not_radial_variance():
    psf = kernel(2.0)
    x = np.arange(psf.shape[1]) - psf.shape[1] // 2
    axis_variance = np.sum(psf * x[None, :] ** 2)
    assert diameter_from_variance(axis_variance, 4) == pytest.approx(2, abs=1e-10)
    invalid = diameter_from_variance([-1, 100, np.nan], 4)
    assert np.isnan(invalid).all()


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_fit_is_stable_when_amplitude_units_change(scale):
    frequencies = np.linspace(0.03, 0.25, 21)
    first = np.ones_like(frequencies)
    second = np.exp(-2 * np.pi**2 * 2.25 * frequencies**2)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        variance, error, valid = fit_variance_difference(first * scale, second * scale, frequencies)
    assert valid and variance == pytest.approx(2.25, abs=1e-11)
    assert error < 1e-11


def test_patch_preprocessing_does_not_modify_input_images():
    rng = np.random.default_rng(14)
    first = 0.5 + 0.05 * rng.normal(size=(64, 64))
    second = ndimage.gaussian_filter(first, 1.0)
    first[:40] = 0
    second[:40] = 0
    originals = first.copy(), second.copy()
    estimate_aperture_pair(first, second)
    np.testing.assert_array_equal(first, originals[0])
    np.testing.assert_array_equal(second, originals[1])
