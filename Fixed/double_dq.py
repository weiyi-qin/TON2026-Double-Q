from pathlib import Path
import math
import random
import time

import cvxpy as cp
import numpy as np


SEED = 43
X_DIM = 2
CONSTRAINT_DIM = 3
A_RANGE = (10, 50)
B_RANGE = (0, 20)
TOTAL_STEPS = 1000
ROUND_NUM = 1
ALGORITHM_LABEL = "Double-DQ"
OUTPUT_FILE = "double_dq_results.npz"

SCRIPT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate_theta1(t):
    return np.random.uniform(
        low=-math.pow(t, 1 / 10),
        high=math.pow(t, 1 / 10),
        size=(X_DIM, 1),
    )


def generate_theta2(t):
    if t in range(1, 350):
        return np.random.uniform(low=-1, high=0, size=(X_DIM, 1))
    if t in range(400, 750):
        return np.random.uniform(low=-1, high=0, size=(X_DIM, 1))
    if t in range(800, 1001):
        return np.random.uniform(low=-1, high=0, size=(X_DIM, 1))
    return np.random.uniform(low=0, high=1, size=(X_DIM, 1))


def generate_theta3(t, ut):
    theta3 = np.zeros((X_DIM, 1))
    theta3[0] = math.pow(-1, ut[t - 1])
    theta3[1] = math.pow(-1, ut[t - 1])
    return theta3


def generate_a():
    return np.random.uniform(
        low=A_RANGE[0],
        high=A_RANGE[1],
        size=(CONSTRAINT_DIM, X_DIM),
    )


def generate_b():
    return np.random.uniform(
        low=B_RANGE[0],
        high=B_RANGE[1],
        size=(CONSTRAINT_DIM, 1),
    )


def generate_loss_and_constraints():
    ut = random.sample(range(1, TOTAL_STEPS + 1), TOTAL_STEPS)
    theta_list = []

    for step_index in range(TOTAL_STEPS):
        theta1 = generate_theta1(step_index + 1)
        theta2 = generate_theta2(step_index + 1)
        theta3 = generate_theta3(step_index, ut)
        theta_list.append(theta1 + theta2 + theta3)

    return theta_list, generate_a(), generate_b()


def project_onto_xt(x_point, a_mat, b_vec):
    y_var = cp.Variable(shape=x_point.shape)
    constraints = [y_var >= -1, y_var <= 1, a_mat @ y_var <= b_vec]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y_var - x_point)), constraints)
    prob.solve(solver=cp.ECOS)
    if y_var.value is None:
        return x_point.copy()
    return y_var.value


def distance_and_gradient(x_point, a_mat, b_vec, d_lip):
    violation = np.maximum(0, a_mat @ x_point - b_vec)
    if (
        np.all(violation <= 1e-8)
        and np.all(x_point >= -1 - 1e-8)
        and np.all(x_point <= 1 + 1e-8)
    ):
        return 0.0, np.zeros_like(x_point)

    projection = project_onto_xt(x_point, a_mat, b_vec)
    diff = x_point - projection
    dist_norm = np.linalg.norm(diff)
    if dist_norm < 1e-10:
        return 0.0, np.zeros_like(x_point)

    return d_lip * dist_norm, d_lip * diff / dist_norm


def double_dq_centralized(
    step,
    choice_history,
    zq,
    zq_hat,
    gradient_norms,
    theta_list,
    a_mat,
    b_vec,
):
    epsilon = 0.01
    radius = 2 * np.sqrt(2)
    d_lip = 20.0
    beta_dq = 1 / (2 * d_lip * radius)
    lam = np.sqrt(TOTAL_STEPS)
    u_dq = 1.0
    v_min = 0.1 * epsilon * TOTAL_STEPS
    eps_dq = 1 / TOTAL_STEPS

    if step == 1:
        for i in range(CONSTRAINT_DIM):
            zq[i] = 0.0
            zq_hat[i] = v_min
        return np.zeros((X_DIM, 1)), zq, zq_hat, 0.0

    theta_estimate = theta_list[step - 2]
    x_prev = choice_history[-1]

    _, grad_d_prev = distance_and_gradient(x_prev, a_mat, b_vec, d_lip)

    for i in range(CONSTRAINT_DIM):
        d_i, _ = distance_and_gradient(x_prev, a_mat[i : i + 1], b_vec[i : i + 1], d_lip)
        zq[i] = zq[i] + beta_dq * d_i

    psi_prime = np.array([zq[i] + lam for i in range(CONSTRAINT_DIM)])

    loss_grad = (2 * (x_prev - theta_estimate) + 20 * theta_estimate).T
    l_grad = u_dq * (loss_grad + grad_d_prev.T)

    cons_grad_terms = np.zeros((1, X_DIM))
    for i in range(CONSTRAINT_DIM):
        _, grad_d_i = distance_and_gradient(x_prev, a_mat[i : i + 1], b_vec[i : i + 1], d_lip)
        cons_grad_terms += beta_dq * psi_prime[i] * grad_d_i.T

    gradient = l_grad + cons_grad_terms

    gradient_norm = np.linalg.norm(gradient) ** 2
    gradient_norms_arr = np.array(gradient_norms)
    alpha = (
        np.sqrt(2)
        * radius
        / (2 * np.sqrt(gradient_norms_arr.sum() + gradient_norm))
    )

    x_var = cp.Variable(shape=(X_DIM, 1))
    constraints = [x_var >= -1, x_var <= 1]

    dist_penalty = 0
    for i in range(CONSTRAINT_DIM):
        y_i = cp.Variable(shape=(X_DIM, 1))
        constraints += [
            y_i >= -1,
            y_i <= 1,
            a_mat[i : i + 1] @ y_i <= b_vec[i : i + 1],
        ]
        dist_penalty += zq_hat[i] * d_lip * cp.norm(x_var - y_i, 2)

    objective = cp.Minimize(
        gradient @ (x_var - x_prev)
        + dist_penalty
        + (1 / alpha) * cp.sum_squares(x_var - x_prev)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=5000)
    x_new = x_var.value
    if x_new is None:
        x_new = x_prev.copy()

    for i in range(CONSTRAINT_DIM):
        d_i_new, _ = distance_and_gradient(x_new, a_mat[i : i + 1], b_vec[i : i + 1], d_lip)
        zq_hat[i] = max(v_min, (1 - eps_dq) * zq_hat[i] + d_i_new)

    return x_new, zq, zq_hat, float(gradient_norm)


def evaluate_choices(choice_history, theta_list, a_mat, b_vec):
    accumulated_loss = []
    total_loss = 0
    for step in range(TOTAL_STEPS):
        x_value = choice_history[step]
        theta = theta_list[step]
        loss = (x_value - theta).T @ (x_value - theta) + 20 * theta.T @ x_value
        total_loss += loss[0, 0]
        accumulated_loss.append(total_loss)

    accumulated_violation = []
    total_violation = 0
    for step in range(TOTAL_STEPS):
        x_value = choice_history[step]
        violation = np.linalg.norm(np.maximum(0, np.matmul(a_mat, x_value) - b_vec), 2)
        total_violation += violation
        accumulated_violation.append(total_violation)

    return np.array(accumulated_loss), np.array(accumulated_violation)


def run_experiment():
    set_seed(SEED)
    theta_list, a_mat, b_vec = generate_loss_and_constraints()

    print("Start running DoubleDQ algorithm...")
    averaged_loss = np.zeros(TOTAL_STEPS)
    averaged_violation = np.zeros(TOTAL_STEPS)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        zq = np.zeros(CONSTRAINT_DIM)
        zq_hat = np.zeros(CONSTRAINT_DIM)
        gradient_norms = []

        for step in range(1, TOTAL_STEPS + 1):
            x_value, zq, zq_hat, gradient_store = double_dq_centralized(
                step,
                choice_history,
                zq,
                zq_hat,
                gradient_norms,
                theta_list,
                a_mat,
                b_vec,
            )
            gradient_norms.append(gradient_store)
            choice_history.append(x_value)

            if step % 200 == 0:
                print(
                    "DoubleDQ (Centralized): Run",
                    total_run + 1,
                    "Step",
                    step,
                    "finished.",
                )

        loss, violation = evaluate_choices(choice_history, theta_list, a_mat, b_vec)
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
        total_steps=TOTAL_STEPS,
        seed=SEED,
        algorithm_label=ALGORITHM_LABEL,
    )
    print(f"DoubleDQ algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
