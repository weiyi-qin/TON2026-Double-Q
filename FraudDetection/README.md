# Online Fraud Detection

This folder reproduces the online fraud detection experiment from the Double-Q paper.

To run this experiment, place the required CSV data file in this folder, execute each algorithm script first to generate its result file, then run `draw.py` to load the saved results and draw the final figure.

The compared methods are `Double-DQ`, `Double-Q`, `[24]`, and `[29]`. Here, `[24]` denotes S. Supantha and A. Sinha, "Universal Dynamic Regret and Constraint Violation Bounds for Constrained Online Convex Optimization," arXiv:2510.01867, 2025, and `[29]` denotes R. Vaze and A. Sinha, "O(sqrt(T)) Static Regret and Instance Dependent Constraint Violation for Constrained Online Convex Optimization," in Proc. Adv. Neural Info. Proc. Sys. (NeurIPS), 2025. Each script saves its accumulated loss and hard violation results as an `.npz` file, and `draw.py` reads these files to generate the final figure.

For a fair comparison, all algorithms use the same experiment setup. We only tune the multiplicative factor of each algorithm's learning rate or step size to obtain its best empirical performance, while keeping the other parameters consistent or following the paper setting.
For `[24]` and `[29]`, we use the suggested parameter settings from the original papers and fine-tune only constants that do not affect the theoretical rates.

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

## Run

```bash
python FraudDetection/double_dq.py
python FraudDetection/double_queue.py
python FraudDetection/baseline_24.py
python FraudDetection/baseline_29.py
python FraudDetection/draw.py
```

Use `FraudDetection/draw.py --no-show` if you only want to save the figure without opening a Matplotlib window.
