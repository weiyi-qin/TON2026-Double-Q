from pathlib import Path
import random
import time

import cvxpy as cp
import numpy as np
import pandas as pd


SEED = 200
X_DIM = 100
CONSTRAINT_DIM = 1
ROUND_NUM = 1
ALGORITHM_LABEL = "Double-DQ"
OUTPUT_FILE = "double_dq_results.npz"

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


def double_dq_centralized(
    step,
    choice_history,
    zq,
    zq_hat,
    gradient_norms,
    lambda_delay,
    ct,
    job_arrival_number,
    total_steps,
):
    epsilon = 0.01
    radius = 10 * np.sqrt(10)
    d_lip = 100
    beta_dq = 1 / (2 * d_lip * radius)
    lam = np.sqrt(total_steps)
    u_dq = 1.0
    v_min = 0.1 * epsilon * total_steps
    eps_dq = 1 / total_steps

    if step == 1:
        lambda_estimate = job_arrival_number[0]
        for i in range(CONSTRAINT_DIM):
            zq[i] = 0.0
            zq_hat[i] = v_min
        return np.zeros((X_DIM, 1)), zq, zq_hat, 0.0, lambda_estimate

    x_prev = choice_history[-1]
    ct_estimate = np.array(ct[step - 2]).reshape(-1, 1)
    lambda_estimate = job_arrival_number[step - 2] + lambda_delay

    _, grad_d_prev = distance_and_gradient_to_capacity(x_prev, lambda_estimate, d_lip)

    for i in range(CONSTRAINT_DIM):
        d_prev, _ = distance_and_gradient_to_capacity(x_prev, lambda_estimate, d_lip)
        zq[i] = zq[i] + beta_dq * d_prev

    psi_prime = np.array([zq[i] + lam for i in range(CONSTRAINT_DIM)])

    loss_grad = ct_estimate
    l_grad = u_dq * (loss_grad + grad_d_prev)

    cons_grad_terms = np.zeros((X_DIM, 1))
    for i in range(CONSTRAINT_DIM):
        cons_grad_terms += beta_dq * psi_prime[i] * grad_d_prev

    gradient = l_grad + cons_grad_terms

    gradient_norm = float(np.linalg.norm(gradient) ** 2)
    gradient_norms_arr = np.array(gradient_norms)
    alpha = np.sqrt(2) * radius / (
        2 * np.sqrt(gradient_norms_arr.sum() + gradient_norm + 1e-8)
    )

    x_var = cp.Variable(shape=(X_DIM, 1))
    y_var = cp.Variable(shape=(X_DIM, 1))
    lambda_safe = cap_lambda(lambda_estimate)

    constraints = [
        x_var >= 0,
        x_var <= 100,
        y_var >= 0,
        y_var <= 100,
        cp.sum(4 * cp.log(1 + 4 * y_var)) >= lambda_safe,
    ]

    dist_penalty = zq_hat[0] * d_lip * cp.norm(x_var - y_var, 2)
    gradient_row = gradient.reshape(1, -1)
    objective = cp.Minimize(
        gradient_row @ (x_var - x_prev)
        + dist_penalty
        + (1 / (2 * alpha)) * cp.sum_squares(x_var - x_prev)
    )

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=5000, verbose=False)
    except Exception:
        x_new = x_prev.copy()
    else:
        x_new = x_var.value if x_var.value is not None else x_prev.copy()

    lambda_estimate_new = job_arrival_number[step - 1] + lambda_delay
    d_new, _ = distance_and_gradient_to_capacity(x_new, lambda_estimate_new, d_lip)
    zq_hat[0] = max(v_min, (1 - eps_dq) * zq_hat[0] + d_new)

    delay_new = max(0.0, lambda_estimate_new - service_rate(x_new))
    return x_new, zq, zq_hat, gradient_norm, delay_new


def run_experiment():
    set_seed(SEED)
    ct, job_arrival_number, total_steps = load_experiment_data()

    print("Start running DoubleDQ algorithm...")
    averaged_loss = np.zeros(total_steps)
    averaged_violation = np.zeros(total_steps)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        zq = np.zeros(CONSTRAINT_DIM)
        zq_hat = np.zeros(CONSTRAINT_DIM)
        gradient_norms = []
        lambda_current = job_arrival_number[0]
        delay_history = []

        for step in range(1, total_steps + 1):
            x_value, zq, zq_hat, gradient_store, lambda_current = double_dq_centralized(
                step,
                choice_history,
                zq,
                zq_hat,
                gradient_norms,
                lambda_current,
                ct,
                job_arrival_number,
                total_steps,
            )
            gradient_norms.append(gradient_store)
            choice_history.append(x_value)
            delay_history.append(lambda_current)

            if step % 200 == 0:
                print(
                    "DoubleDQ (Centralized): Run",
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
        print(f"DoubleDQ Run {total_run + 1} elapsed time: {end_time - start_time:.2f} seconds.")

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
    print(f"DoubleDQ algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
