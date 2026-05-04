from pathlib import Path
import math
import random
import time

import cvxpy as cp
import numpy as np


SEED = 52
X_DIM = 2
CONSTRAINT_DIM = 3
A_RANGE = (10, 50)
B_RANGE = (0, 10)
TOTAL_STEPS = 1000
X_UPPER = 6
X_LOWER = -6
NUMBER_OF_THETA = 40
ROUND_NUM = 20
ALGORITHM_LABEL = "Double-Q"
OUTPUT_FILE = "double_queue_results.npz"

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
    if t in range(800, 1000):
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
    a_list = []
    b_list = []

    for step_index in range(TOTAL_STEPS):
        theta1 = generate_theta1(step_index)
        theta2 = generate_theta2(step_index)
        theta3 = generate_theta3(step_index, ut)
        theta_list.append(theta1 + theta2 + theta3)
        a_list.append(generate_a())
        b_list.append(generate_b())

    return theta_list, a_list, b_list


def positive_constraint_value(a_row, x_value, b_value):
    return max(0.0, (a_row @ x_value - b_value).item())


def double_queue_centralized(
    step,
    choice_history,
    vq,
    zq,
    gradient_norms,
    theta_list,
    a_list,
    b_list,
):
    epsilon = 0.01
    if step == 1:
        for i in range(CONSTRAINT_DIM):
            vq[i] = epsilon / np.power(step, 2)
            zq[i] = 0
        return np.zeros((X_DIM, 1)), vq, zq, 0.0

    theta_estimate = theta_list[step - 2]
    a_estimate = a_list[step - 2]
    b_estimate = b_list[step - 2]
    x_prev = choice_history[-1]

    v_weight = 5
    lamda = 1 / (2 * np.sqrt(TOTAL_STEPS))
    g_const = 2
    r_const = (X_UPPER - X_LOWER) * np.sqrt(2)
    d_const = 30
    beta = 1 / (4 * d_const * r_const * np.exp(g_const)+1)

    for i in range(CONSTRAINT_DIM):
        zq[i] = zq[i] + beta * positive_constraint_value(a_estimate[i], x_prev, b_estimate[i])

    cons_gradient = []
    for i in range(CONSTRAINT_DIM):
        if positive_constraint_value(a_estimate[i], x_prev, b_estimate[i]) > 0:
            cons_gradient.append(a_estimate[i])
        else:
            cons_gradient.append(np.zeros(X_DIM))

    gradient = v_weight * (2 * (x_prev - theta_estimate) + NUMBER_OF_THETA * theta_estimate).T
    for i in range(CONSTRAINT_DIM):
        gradient += lamda * np.exp(lamda * zq[i]) * beta * cons_gradient[i]

    gradient_norm = np.linalg.norm(gradient) ** 2
    gradient_norms_arr = np.array(gradient_norms)
    alpha = (
        np.sqrt(2)
        * r_const
        / np.sqrt(gradient_norms_arr.sum() + gradient_norm)
    )

    x_var = cp.Variable(shape=(X_DIM, 1))
    constraints = [x_var >= X_LOWER, x_var <= X_UPPER]
    constraint_penalty = 0
    for i in range(CONSTRAINT_DIM):
        constraint_penalty += vq[i] * (
            cp.sum(
                cp.abs(a_estimate[i] @ x_var - b_estimate[i])
                + a_estimate[i] @ x_var
                - b_estimate[i]
            )
            / 2
        )

    objective = cp.Minimize(
        gradient @ (x_var - x_prev)
        + constraint_penalty
        + (1 / (2 * alpha)) * cp.sum_squares(x_var - x_prev)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    x_new = x_var.value
    if x_new is None:
        x_new = x_prev.copy()

    a_current = a_list[step - 1]
    b_current = b_list[step - 1]
    eta = 1 / TOTAL_STEPS
    gamma = epsilon * TOTAL_STEPS

    for i in range(CONSTRAINT_DIM):
        vq[i] = max(
            gamma,
            (1 - eta) * vq[i] + positive_constraint_value(a_current[i], x_new, b_current[i]),
        )

    return x_new, vq, zq, float(gradient_norm)


def evaluate_choices(choice_history, theta_list, a_list, b_list):
    accumulated_loss = []
    total_loss = 0
    for step in range(TOTAL_STEPS):
        x_value = choice_history[step]
        theta = theta_list[step]
        loss = (x_value - theta).T @ (x_value - theta) + NUMBER_OF_THETA * theta.T @ x_value
        total_loss += loss[0, 0]
        accumulated_loss.append(total_loss)

    accumulated_violation = []
    total_violation = 0
    for step in range(TOTAL_STEPS):
        x_value = choice_history[step]
        violation = np.linalg.norm(
            np.maximum(0, np.matmul(a_list[step], x_value) - b_list[step]),
            2,
        )
        total_violation += violation
        accumulated_violation.append(total_violation)

    return np.array(accumulated_loss), np.array(accumulated_violation)


def run_experiment():
    set_seed(SEED)
    theta_list, a_list, b_list = generate_loss_and_constraints()

    print("Start running DoubleQueue algorithm...")
    averaged_loss = np.zeros(TOTAL_STEPS)
    averaged_violation = np.zeros(TOTAL_STEPS)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        vq = np.zeros(CONSTRAINT_DIM)
        zq = np.zeros(CONSTRAINT_DIM)
        gradient_norms = []

        for step in range(1, TOTAL_STEPS + 1):
            x_value, vq, zq, gradient_store = double_queue_centralized(
                step,
                choice_history,
                vq,
                zq,
                gradient_norms,
                theta_list,
                a_list,
                b_list,
            )
            gradient_norms.append(gradient_store)
            choice_history.append(x_value)

            if step % 200 == 0:
                print(
                    "DoubleQueue (Centralized): Run",
                    total_run + 1,
                    "Step",
                    step,
                    "finished.",
                )

        loss, violation = evaluate_choices(choice_history, theta_list, a_list, b_list)
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
        total_steps=TOTAL_STEPS,
        algorithm_label=ALGORITHM_LABEL,
    )
    print(f"DoubleQueue algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
