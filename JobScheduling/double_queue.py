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
ALGORITHM_LABEL = "Double-Q"
OUTPUT_FILE = "double_queue_results.npz"

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


def double_queue_centralized(
    step,
    choice_history,
    vq,
    zq,
    gradient_norms,
    lambda_delay,
    ct,
    job_arrival_number,
    total_steps,
):
    epsilon = 1

    if step == 1:
        lambda_estimate = job_arrival_number[0]
        for i in range(CONSTRAINT_DIM):
            vq[i] = epsilon * total_steps
            zq[i] = 0
        return np.zeros((X_DIM, 1)), vq, zq, 0.0, lambda_estimate

    ct_estimate = ct[step - 2]
    lambda_estimate = job_arrival_number[step - 2] + lambda_delay

    v_weight = 1
    lamda = 1 / (2 * np.sqrt(total_steps))
    g_const = 0.2
    r_const = 100
    d_const = 10 * np.sqrt(10)
    beta = 1 / (4 * d_const * r_const * np.exp(g_const)+1)

    for i in range(CONSTRAINT_DIM):
        zq[i] = zq[i] + beta * lambda_delay

    cons_gradient = []
    for i in range(CONSTRAINT_DIM):
        if lambda_estimate - np.sum(4 * np.log(1 + 4 * choice_history[-1])) > 0:
            grad_term = -16 / (1 + 4 * choice_history[-1])
            cons_gradient.append(grad_term.flatten())
        else:
            cons_gradient.append(np.zeros(X_DIM))

    ct_estimate = np.array(ct_estimate).flatten()
    gradient = v_weight * ct_estimate + lamda * np.exp(lamda * zq[0]) * beta * cons_gradient[0]

    gradient_norm = np.linalg.norm(gradient) ** 2
    gradient_norms_arr = np.array(gradient_norms)
    alpha = (
        np.sqrt(2)
        * r_const
        / (2 * np.sqrt(gradient_norms_arr.sum() + gradient_norm))
        * 0.1
    )

    x_var = cp.Variable(shape=(X_DIM, 1))
    constraints = [x_var >= 0, x_var <= 100]

    objective = cp.Minimize(
        gradient @ x_var
        + vq[0] * cp.maximum(0, lambda_estimate - cp.sum(4 * cp.log(1 + 4 * x_var)))
        + (1 / (2 * alpha)) * cp.sum_squares(x_var - choice_history[-1])
    )
    prob = cp.Problem(objective, constraints)
    prob.solve()
    x_new = x_var.value
    if x_new is None:
        x_new = choice_history[-1].copy()

    eta = 1 / total_steps
    gamma = epsilon * total_steps
    lambda_estimate_new = job_arrival_number[step - 1] + lambda_delay

    for i in range(CONSTRAINT_DIM):
        vq[i] = max(
            gamma,
            (1 - eta) * vq[i]
            + max(0, lambda_estimate_new - np.sum(4 * np.log(1 + 4 * x_new))),
        )

    delay_new = max(0, lambda_estimate_new - np.sum(4 * np.log(1 + 4 * x_new)))
    return x_new, vq, zq, float(gradient_norm), delay_new


def run_experiment():
    set_seed(SEED)
    ct, job_arrival_number, total_steps = load_experiment_data()

    print("Start running DoubleQueue algorithm...")
    averaged_loss = np.zeros(total_steps)
    averaged_violation = np.zeros(total_steps)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        vq = np.zeros(CONSTRAINT_DIM)
        zq = np.zeros(CONSTRAINT_DIM)
        gradient_norms = []
        lambda_current = job_arrival_number[0]
        delay_history = []

        for step in range(1, total_steps + 1):
            x_value, vq, zq, gradient_store, lambda_current = double_queue_centralized(
                step,
                choice_history,
                vq,
                zq,
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
                    "DoubleQueue (Centralized): Run",
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
        print(f"DoubleQueue Run {total_run + 1} elapsed time: {end_time - start_time:.2f} seconds.")

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
    print(f"DoubleQueue algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
