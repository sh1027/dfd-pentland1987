"""Demo outputs and evaluation stay independent of ground truth during inference."""

import json
import sys

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

import demo


def test_cli_evaluation_and_visualization(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(12)
    sharp = 0.5 + 0.1 * rng.normal(size=(64, 64))
    blurred = ndimage.gaussian_filter(sharp, 1.0, mode="reflect")
    for name, image in [("sharp.png", sharp), ("blur.png", blurred)]:
        Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(data / name)
    metadata = {
        "rgb_path": "sharp.png",
        "depth_path": "truth.npy",
        "near": 2.0,
        "far": 4.0,
        "image_width_px": 64,
        "image_height_px": 64,
        "defocus_paths": {"blur.png": {"f_number": 4}},
    }
    (data / "metadata.json").write_text(json.dumps(metadata))
    (data / "defocus.json").write_text(
        json.dumps(
            {
                "blur.png": {"f_number": 4, "K": 8.0, "gamma": 1.0, "focus_disparity": 1.0},
            }
        )
    )
    np.save(data / "truth.npy", np.full((64, 64), 2.5))
    output = tmp_path / "outputs"
    monkeypatch.setattr(
        sys, "argv", ["demo.py", "--data-dir", str(data), "--output-dir", str(output)]
    )
    demo.main()
    prediction = np.load(output / "f4/depth_m.npy")
    assert prediction.dtype == np.float32
    assert prediction.shape == (64, 64)
    assert np.isfinite(prediction).any()
    assert np.isnan(prediction[:16]).all()
    assert (prediction[np.isfinite(prediction)] > metadata["near"]).all()

    assert (output / "f4/sigma_px.npy").is_file()
    assert (output / "f4/focal_disparity.npy").is_file()
    report = json.loads((output / "f4/metrics.json").read_text())
    evaluation = report["evaluation"]
    assert evaluation["coverage"] == np.isfinite(prediction).sum() / prediction.size
    assert evaluation["mae_m"] >= 0
    with Image.open(output / "f4/comparison.png") as figure:
        assert figure.width >= 1000 and figure.height >= 600
    np.save(data / "truth.npy", np.full((64, 64), 5.0))
    demo.main()
    np.testing.assert_array_equal(np.load(output / "f4/depth_m.npy"), prediction)
    changed = json.loads((output / "f4/metrics.json").read_text())["evaluation"]
    assert changed["rmse_m"] != evaluation["rmse_m"]
    saved = {path.name: path.read_bytes() for path in (output / "f4").iterdir()}
    np.save(data / "truth.npy", np.ones((2, 2)))
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(data / "blur.png")
    with pytest.raises(ValueError, match="equally shaped"):
        demo.main()
    assert {path.name: path.read_bytes() for path in (output / "f4").iterdir()} == saved


def test_evaluation_invalid_pixels_and_absolute_scale():
    truth = np.array([[1.0, 2.0], [0.0, np.nan]])
    prediction = np.array([[2.0, np.nan], [99.0, 99.0]])
    result = demo.evaluate_depth(prediction, truth)
    assert result == {
        "count_gt_valid": 2,
        "count_prediction_valid": 1,
        "coverage": 0.5,
        "mae_m": 1.0,
        "rmse_m": 1.0,
    }


def test_evaluation_without_valid_pairs():
    result = demo.evaluate_depth(np.full((2, 2), np.nan), np.ones((2, 2)))
    assert result["coverage"] == 0
    assert result["mae_m"] is None and result["rmse_m"] is None
    json.dumps(result, allow_nan=False)


def test_evaluation_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        demo.evaluate_depth(np.ones((2, 2)), np.ones((2, 1)))
