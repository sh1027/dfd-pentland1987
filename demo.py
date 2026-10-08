"""Demo: Pentland depth estimation, evaluation, and visualization (default f/4)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from aperture import depth_from_diameter, estimate_aperture_pair, focal_disparity
from aperture.softdisk import diameter_from_variance

ROOT = Path(__file__).resolve().parent


def evaluate_depth(prediction, truth):
    """Metric errors on valid pairs and coverage of valid ground truth."""
    if prediction.shape != truth.shape or truth.ndim != 2:
        raise ValueError("Prediction and ground truth must be equally shaped 2D arrays")
    valid_truth = np.isfinite(truth) & (truth > 0)
    valid = valid_truth & np.isfinite(prediction) & (prediction > 0)
    count_truth, count = int(valid_truth.sum()), int(valid.sum())
    error = prediction[valid] - truth[valid]
    return {
        "count_gt_valid": count_truth,
        "count_prediction_valid": count,
        "coverage": count / count_truth if count_truth else 0.0,
        "mae_m": float(np.mean(np.abs(error))) if count else None,
        "rmse_m": float(np.sqrt(np.mean(error**2))) if count else None,
    }


def save_figure(path, sharp, blurred, prediction, truth, disparity, metrics, f_number):
    """Inputs, depth/error, and the Section III.B focal disparity."""
    finite = truth[np.isfinite(truth) & (truth > 0)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if low == high:
        low, high = max(0.0, low - 0.1), high + 0.1
    truth = np.where(np.isfinite(truth) & (truth > 0), truth, np.nan)
    error = np.abs(prediction - truth)
    finite_error = error[np.isfinite(error)]
    limit = max(float(np.percentile(finite_error, 95)), 0.01) if finite_error.size else 1.0
    depth_cmap = plt.get_cmap("viridis").with_extremes(bad="#d6d6d6")
    error_cmap = plt.get_cmap("magma").with_extremes(bad="#d6d6d6")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    axes[0, 0].imshow(sharp)
    axes[0, 0].set_title("All-in-focus input")
    axes[0, 1].imshow(blurred)
    axes[0, 1].set_title(f"Defocused input: f/{f_number:g}")
    for ax, values, title, cmap, vmin, vmax in [
        (axes[0, 2], truth, "Ground truth [m]", depth_cmap, low, high),
        (axes[1, 0], prediction, "Estimated depth [m]", depth_cmap, low, high),
        (axes[1, 1], error, "Absolute error [m] (95% display limit)", error_cmap, 0.0, limit),
    ]:
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    im = axes[1, 2].imshow(disparity, cmap=depth_cmap, vmin=0, vmax=1)
    axes[1, 2].set_title("Focal disparity: blurry / sharp")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.8)
    for ax in axes.flat:
        ax.set_axis_off()
    rmse = f"{metrics['rmse_m']:.4f} m" if metrics["rmse_m"] is not None else "unavailable"
    fig.suptitle(
        f"Pentland | RMSE {rmse} | coverage {metrics['coverage']:.1%}\nGray depth/error pixels: invalid",
        fontsize=13,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(data_dir: Path, output_dir: Path, f_number: float = 4.0, patch_size: int = 33) -> Path:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    entries = metadata["defocus_paths"]
    entries = entries if isinstance(entries, list) else [entries]
    selected = [
        (path, settings)
        for entry in entries
        for path, settings in entry.items()
        if np.isclose(float(settings["f_number"]), f_number)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one image for f/{f_number:g}; found {len(selected)}")
    blurred_name, settings = selected[0]
    sidecar = json.loads((data_dir / "defocus.json").read_text())
    settings = {**settings, **sidecar.get(blurred_name, {})}
    k, gamma = float(settings["K"]), float(settings.get("gamma", 4))
    qfocus = float(settings.get("focus_disparity", 1))
    near, far = float(metadata["near"]), float(metadata["far"])
    if not all(np.isfinite(value) and value > 0 for value in (k, gamma, near, far)):
        raise ValueError("K, gamma and depth bounds must be finite and positive")
    if near >= far or not np.isfinite(qfocus) or not 0 <= qfocus <= 1:
        raise ValueError("Require near < far and focus_disparity in [0, 1]")
    if not np.isclose(float(settings["f_number"]), f_number):
        raise ValueError("Saved f-number disagrees with selected image")
    focus = 1 / (1 / far + qfocus * (1 / near - 1 / far))

    def load_rgb(name):
        with Image.open(data_dir / name) as image:
            return (np.asarray(image.convert("RGB"), dtype=float) / 255) ** gamma

    sharp, blurred = load_rgb(metadata["rgb_path"]), load_rgb(blurred_name)
    shape = (metadata["image_height_px"], metadata["image_width_px"])
    if sharp.shape[:2] != shape or blurred.shape != sharp.shape:
        raise ValueError("Image dimensions disagree with metadata or each other")
    maximum = float(np.max(2 * k * np.abs(1 / np.array([near, far]) - 1 / focus))) * 1.25
    result = estimate_aperture_pair(sharp, blurred, patch_size=patch_size)
    diameter = diameter_from_variance(result.variance_difference_px2, maximum)
    depth = depth_from_diameter(diameter, k=k, focus_distance_m=focus)
    valid = result.valid & np.isfinite(depth)
    depth = np.where(valid, depth, np.nan)
    # Ground truth is read after inference and cannot affect the prediction.
    truth = np.load(data_dir / metadata["depth_path"], allow_pickle=False)
    metrics = evaluate_depth(depth, truth)
    folder = output_dir / f"f{f_number:g}".replace(".", "p")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "depth_m.npy"
    np.save(path, depth.astype(np.float32), allow_pickle=False)
    sigma = np.sqrt(np.where(valid, result.variance_difference_px2, np.nan))
    np.save(folder / "sigma_px.npy", sigma.astype(np.float32), allow_pickle=False)
    disparity, _ = focal_disparity(sharp, blurred)
    np.save(folder / "focal_disparity.npy", disparity.astype(np.float32), allow_pickle=False)
    report = {"f_number": f_number, "evaluation": metrics}
    (folder / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_figure(
        folder / "comparison.png",
        sharp ** (1 / gamma),
        blurred ** (1 / gamma),
        depth,
        truth,
        disparity,
        metrics,
        f_number,
    )
    if metrics["rmse_m"] is None:
        print(f"f/{f_number:g}: no valid evaluation pixels -> {folder}")
    else:
        print(
            f"f/{f_number:g}: MAE={metrics['mae_m']:.5f} m, "
            f"RMSE={metrics['rmse_m']:.5f} m, coverage={metrics['coverage']:.2%} -> {folder}"
        )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--f-number", type=float, default=4.0)
    parser.add_argument("--patch-size", type=int, default=33)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.f_number, args.patch_size)


if __name__ == "__main__":
    main()
