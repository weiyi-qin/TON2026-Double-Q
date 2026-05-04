from pathlib import Path
import random
import time

import cvxpy as cp
import numpy as np
import pandas as pd


SEED = 200
X_DIM = 100
ROUND_NUM = 1
ALGORITHM_LABEL = "PSGD"
OUTPUT_FILE = "baseline_24_results.npz"

SCRIPT_DIR = Path(__file__).resolve().parent
COST_DATA_FILE = SCRIPT_DIR / "filtered_file.csv"
ARRIVAL_DATA_FILE = SCRIPT_DIR / "experiment_data_100vms.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def require_data_files():
    missing = [str(path.name) for path in (COST_DATA_FILE, ARRIVAL_DATA_FILE) if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required data file(s): {missing_text}. "
            "Place them in the JobScheduling folder before running this script."
        )


def load_experiment_data():
    require_data_files()

    cost_data = pd.read_csv(COST_DATA_FILE, header=None)
    cost_col = cost_data.iloc[:, 3].values
    cost_steps = int(len(cost_col) / 10)

    ct = []
    for i in range(cost_steps):
        group_data = cost_col[i * 10 : (i + 1) * 10]
        data_r = np.repeat(group_data, 10)
        ct.append(np.abs(data_r) * 0.005)

    arrival_data = pd.read_csv(ARRIVAL_DATA_FILE)
    lambda_series = arrival_data.groupby("timestamp", sort=True)["avg_cpu"].sum().values

    target_mean = 600
    target_std = 23.0
    series_mean = lambda_series.mean()
    series_std = lambda_series.std()

    job_arrival_number = np.maximum(
        target_mean + target_std * (lambda_series - series_mean) / (series_std + 1e-8),
        0,
    )

    total_steps = min(len(ct), len(job_arrival_number))
    return ct[:total_steps], job_arrival_number[:total_steps], total_steps


def service_rate(x):
    x_vec = np.array(x, dtype=float).reshape(-1)
    x_vec = np.maximum(x_vec, 0.0)
    return float(np.sum(4.0 * np.log(1.0 + 4.0 * x_vec)))


def service_grad(x):
    x_vec = np.array(x, dtype=float).reshape(-1, 1)
    x_vec = np.maximum(x_vec, 0.0)
    return 16.0 / (1.0 + 4.0 * x_vec)


def project_to_box(x):
    return np.clip(np.array(x, dtype=float), 0.0, 100.0)


MAX_SERVICE = service_rate(np.full((X_DIM, 1), 100.0))


def cap_lambda(lambda_target):
    return float(np.clip(lambda_target, 0.0, MAX_SERVICE - 1e-6))


def project_onto_capacity(x_point, lambda_target):
    lambda_safe = cap_lambda(lambda_target)
    y_var = cp.Variable(shape=(X_DIM, 1))
    constraints = [
        y_var >= 0,
        y_var <= 100,
        cp.sum(4 * cp.log(1 + 4 * y_var)) >= lambda_safe,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y_var - x_point)), constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=4000, verbose=False)
    except Exception:
        return project_to_box(x_point)
    if y_var.value is None:
        return project_to_box(x_point)
    return y_var.value


def distance_and_gradient_to_capacity(x_point, lambda_target, d_lip):
    projection = project_onto_capacity(x_point, lambda_target)
    diff = np.array(x_point) - np.array(projection)
    dist_norm = float(np.linalg.norm(diff))

    if dist_norm < 1e-10:
        return 0.0, np.zeros_like(x_point)

    return d_lip * dist_norm, d_lip * diff / dist_norm


def evaluate_trajectory(choice_history, delay_history, ct, job_arrival_number, total_steps):
    accumulated_loss = []
    accumulated_violation = []
    total_loss = 0.0
    total_violation = 0.0

    if len(delay_history) > 0:
        delay_history[0] = 0.0

    for step in range(total_steps):
        x_value = choice_history[step]
        loss = float((ct[step] @ x_value).item())
        total_loss += loss
        accumulated_loss.append(total_loss)

        if step == 0:
            violation = 100.0
        else:
            violation = max(
                0.0,
                float(job_arrival_number[step] + delay_history[step] - service_rate(x_value)),
            )
        total_violation += violation
        accumulated_violation.append(total_violation)

    return np.array(accumulated_loss), np.array(accumulated_violation)


def naive_surrogate_gd_centralized(
    step,
    choice_history,
    gradient_norms,
    lambda_delay,
    ct,
    job_arrival_number,
):
    if step == 1:
        lambda_estimate = job_arrival_number[0]
        return np.zeros((X_DIM, 1)), 0.0, lambda_estimate

    x_prev = choice_history[-1]
    ct_estimate = np.array(ct[step - 2]).reshape(-1, 1)
    lambda_estimate = job_arrival_number[step - 2] + lambda_delay

    grad_loss = ct_estimate
    d_lip = 10*np.sqrt(10)
    _, grad_dist = distance_and_gradient_to_capacity(x_prev, lambda_estimate, d_lip)

    g_value = lambda_estimate - service_rate(x_prev)
    if g_value > 0:
        grad_hinge = -service_grad(x_prev)
    else:
        grad_hinge = np.zeros_like(x_prev)

    gradient = grad_loss + grad_dist + grad_hinge

    gradient_norm = float(np.linalg.norm(gradient) ** 2)
    gradient_norms_arr = np.array(gradient_norms)

    eta = 1 / np.sqrt(step)
    if gradient_norms_arr.size > 0:
        eta = min(eta, 50.0 / np.sqrt(gradient_norms_arr.sum() + gradient_norm + 1e-8))

    x_new = project_to_box(x_prev - eta * gradient)

    lambda_estimate_new = job_arrival_number[step - 1] + lambda_delay
    delay_new = max(0.0, lambda_estimate_new - service_rate(x_new))
    return x_new, gradient_norm, delay_new


def run_experiment():
    set_seed(SEED)
    ct, job_arrival_number, total_steps = load_experiment_data()

    print("Start running PSGD algorithm...")
    averaged_loss = np.zeros(total_steps)
    averaged_violation = np.zeros(total_steps)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        gradient_norms = []
        lambda_current = job_arrival_number[0]
        delay_history = []

        for step in range(1, total_steps + 1):
            x_value, gradient_store, lambda_current = naive_surrogate_gd_centralized(
                step,
                choice_history,
                gradient_norms,
                lambda_current,
                ct,
                job_arrival_number,
            )
            gradient_norms.append(gradient_store)
            choice_history.append(x_value)
            delay_history.append(lambda_current)

            if step % 200 == 0:
                print(
                    "PSGD (Centralized): Run",
                    total_run + 1,
                    "Step",
                    step,
                    "finished.",
                )

        loss, violation = evaluate_trajectory(
            choice_history,
            delay_history,
            ct,
            job_arrival_number,
            total_steps,
        )
        averaged_loss += loss
        averaged_violation += violation

        end_time = time.time()
        print(f"PSGD Run {total_run + 1} elapsed time: {end_time - start_time:.2f} seconds.")

    averaged_loss /= ROUND_NUM
    averaged_violation /= ROUND_NUM

    output_path = SCRIPT_DIR / OUTPUT_FILE
    np.savez(
        output_path,
        loss=averaged_loss,
        violation=averaged_violation,
        total_steps=total_steps,
        algorithm_label=ALGORITHM_LABEL,
    )
    print(f"PSGD algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
