# Time-Varying Constraint Online Quadratic Programming

This folder reproduces the time-varying constraint online quadratic programming experiment from the Double-Q paper.

The compared methods are `Double-DQ`, `Double-Q`, `[24]`, and `[29]`. Here, `[24]` denotes S. Supantha and A. Sinha, "Universal Dynamic Regret and Constraint Violation Bounds for Constrained Online Convex Optimization," arXiv:2510.01867, 2025, and `[29]` denotes R. Vaze and A. Sinha, "O(sqrt(T)) Static Regret and Instance Dependent Constraint Violation for Constrained Online Convex Optimization," in Proc. Adv. Neural Info. Proc. Sys. (NeurIPS), 2025. Each script saves its accumulated loss and hard violation results as an `.npz` file, and `draw.py` reads these files to generate the final figure.

For a fair comparison, all algorithms use the same experiment setup. We only tune the multiplicative factor of each algorithm's learning rate or step size to obtain its best empirical performance, while keeping the other parameters consistent or following the paper setting.
For `[24]` and `[29]`, we use the suggested parameter settings from the original papers and fine-tune only constants that do not affect the theoretical rates.

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
python TimeVarying/double_dq.py
python TimeVarying/double_queue.py
python TimeVarying/naive_surrogate_gd.py
python TimeVarying/coco.py
python TimeVarying/draw.py
```

Use `TimeVarying/draw.py --no-show` if you only want to save the figure without opening a Matplotlib window.
