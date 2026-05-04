# Fixed-Constraint Online Quadratic Programming

This folder reproduces the fixed-constraint online quadratic programming experiment from the Double-Q paper.

The compared methods are `Double-DQ`, `Double-Q`, `[24]`, and `[29]`. Each script saves its accumulated loss and hard violation results as an `.npz` file, and `draw.py` reads these files to generate the final figure.

For a fair comparison, all algorithms use the same experiment setup. We only tune the multiplicative factor of each algorithm's learning rate or step size to obtain its best empirical performance, while keeping the other parameters consistent or following the paper setting.

## Requirements

This experiment requires:

- `numpy`
- `cvxpy`
- `matplotlib`

```bash
pip install numpy cvxpy matplotlib
```

Python 3.11 or a recent Python 3 environment is recommended.

## Run

```bash
python Fixed/double_dq.py
python Fixed/double_queue.py
python Fixed/naive_surrogate_gd.py
python Fixed/coco.py
python Fixed/draw.py
```

Use `Fixed/draw.py --no-show` if you only want to save the figure without opening a Matplotlib window.
