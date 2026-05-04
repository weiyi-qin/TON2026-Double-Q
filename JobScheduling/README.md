# Online Network Job Scheduling

This folder reproduces the online network job scheduling experiment from the Double-Q paper.

To run this experiment, place the required CSV data files in this folder, execute each algorithm script first to generate its result file, then run `draw.py` to load the saved results and draw the final figure.

The compared methods are `Double-DQ`, `Double-Q`, `PSGD` (`[24]`), and `SGDP` (`[29]`). Here, `[24]` denotes S. Supantha and A. Sinha, "Universal Dynamic Regret and Constraint Violation Bounds for Constrained Online Convex Optimization," arXiv:2510.01867, 2025, and `[29]` denotes R. Vaze and A. Sinha, "O(sqrt(T)) Static Regret and Instance Dependent Constraint Violation for Constrained Online Convex Optimization," in Proc. Adv. Neural Info. Proc. Sys. (NeurIPS), 2025. Each script saves its accumulated cost and delayed jobs results as an `.npz` file, and `draw.py` reads these files to generate the final figure.

For a fair comparison, all algorithms use the same experiment setup. We fine-tune multiplicative factors in the learning rates or step sizes. We also fine-tune constants that do not affect the theoretical rates. Constants that play the same role are kept identical across algorithms.

## Requirements

This experiment requires:

- `numpy`
- `cvxpy`
- `pandas`
- `matplotlib`

```bash
pip install numpy cvxpy pandas matplotlib
```

Python 3.11 or a recent Python 3 environment is recommended.

## Run

```bash
python JobScheduling/double_dq.py
python JobScheduling/double_queue.py
python JobScheduling/PSGD.py
python JobScheduling/SGDP.py
python JobScheduling/draw.py
```

Use `JobScheduling/draw.py --no-show` if you only want to save the figure without opening a Matplotlib window.
