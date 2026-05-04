# Online Fraud Detection

This folder reproduces the online fraud detection experiment from the Double-Q paper.

To run this experiment, place the required CSV data file in this folder, execute each algorithm script first to generate its result file, then run `draw.py` to load the saved results and draw the final figure.

The compared methods are `Double-DQ`, `Double-Q`, `[24]`, and `[29]`. Each script saves its accumulated loss and hard violation results as an `.npz` file, and `draw.py` reads these files to generate the final figure.

For a fair comparison, all algorithms use the same experiment setup. We only tune the multiplicative factor of each algorithm's learning rate or step size to obtain its best empirical performance, while keeping the other parameters consistent or following the paper setting.

## Requirements

This experiment requires:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `matplotlib`

```bash
pip install numpy pandas scikit-learn torch matplotlib
```

Python 3.11 or a recent Python 3 environment is recommended.

## Data

Place this file in the `FraudDetection` folder before running the scripts:

- `creditcard.csv`

## Run

```bash
python FraudDetection/double_dq.py
python FraudDetection/double_queue.py
python FraudDetection/naive_surrogate_gd.py
python FraudDetection/coco.py
python FraudDetection/draw.py
```

Use `FraudDetection/draw.py --no-show` if you only want to save the figure without opening a Matplotlib window.
