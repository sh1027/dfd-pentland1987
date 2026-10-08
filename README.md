# dfd-pentland1987

## Setup

Python 3.11 or later and uv are required. Run the commands from the project root.

```bash
uv sync --locked
```

## Data

Use the supplied `data/` directory. For another dataset, provide the following in the same format:

- `metadata.json`
- `defocus.json`
- The input images and ground-truth depth `.npy` file referenced by the metadata

If `gamma` and `focus_disparity` are omitted from `defocus.json`, they default to 4 and 1, respectively.

## Run

```bash
uv run demo.py
uv run demo.py --f-number 2.8
uv run demo.py --patch-size 49
uv run demo.py --data-dir /path/to/data --output-dir /path/to/outputs
uv run demo.py --help
```

The defaults are f/4 and a patch size of 33. Results are saved to `outputs/f4/`, or `outputs/f2p8/` for f/2.8.
Running again with the same output directory updates the results.

| File | Contents |
| --- | --- |
| `depth_m.npy` | Estimated depth [m]. Invalid pixels are NaN |
| `sigma_px.npy` | Estimated sigma [px] |
| `focal_disparity.npy` | Focal disparity |
| `metrics.json` | F-number and evaluation results |
| `comparison.png` | Comparison image |

## Check

```bash
uv run pytest -q
uv run ruff check .
```
