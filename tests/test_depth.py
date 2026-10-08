import numpy as np
import pytest

from aperture import depth_from_diameter


@pytest.mark.parametrize("branch,depth", [("near", [1.0, 1.5, 2.0]), ("far", [2.0, 3.0, 10.0])])
def test_radius_calibration_roundtrip(branch, depth):
    depth = np.asarray(depth)
    k, focus = 16.0, 2.0
    radius = k * np.abs(1 / depth - 1 / focus)
    recovered = depth_from_diameter(2 * radius, k=k, focus_distance_m=focus, branch=branch)
    np.testing.assert_allclose(recovered, depth)


def test_focus_and_far_pole_are_not_clipped():
    k, focus = 16.0, 2.0
    pole = 2 * k / focus
    result = depth_from_diameter(
        [0, -1, np.nan, np.inf, pole, pole + 1], k=k, focus_distance_m=focus
    )
    assert result[0] == focus
    assert np.isnan(result[1:]).all()
    assert depth_from_diameter(pole, k=k, focus_distance_m=focus, branch="near") == focus / 2


@pytest.mark.parametrize(
    "parameters",
    [
        {"k": 0, "focus_distance_m": 2},
        {"k": 16, "focus_distance_m": np.nan},
        {"k": 16, "focus_distance_m": 2, "branch": "invalid"},
    ],
)
def test_invalid_camera_settings(parameters):
    with pytest.raises(ValueError):
        depth_from_diameter(1, **parameters)
