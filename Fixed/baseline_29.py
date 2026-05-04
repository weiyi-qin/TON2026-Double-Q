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
ALGORITHM_LABEL = "[29]"
OUTPUT_FILE = "baseline_29_results.npz"

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


def project_onto_st(x_point, a_set, b_set):
    y_var = cp.Variable(shape=x_point.shape)
    constraints = [y_var >= -1, y_var <= 1]
    if a_set is not None and b_set is not None and a_set.size > 0:
        constraints.append(a_set @ y_var <= b_set)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(y_var - x_point)), constraints)
    prob.solve(solver=cp.ECOS)
    if y_var.value is None:
        return x_point.copy()
    return y_var.value


def coco_centralized(
    step,
    choice_history,
    a_history,
    b_history,
    theta_list,
    a_mat,
    b_vec,
):
    a_tm1 = a_mat
    b_tm1 = b_vec

    if step == 1:
        x_init = np.zeros((X_DIM, 1))
        x_init = np.clip(x_init, -1, 1)
        return x_init, a_history, b_history

    x_t = choice_history[-1]
    theta_t = theta_list[step - 2]
    gradient = 0.1 * (x_t - theta_t) + 20 * theta_t

    d_coco = 2 * np.sqrt(2)
    g_coco = 20.0
    eta_t = 2 * d_coco / (g_coco * np.sqrt(step))
    v_t = x_t - eta_t * gradient

    if len(a_history) > 0:
        a_tminus2 = np.vstack(a_history)
        b_tminus2 = np.vstack(b_history)
        y_t = project_onto_st(v_t, a_tminus2, b_tminus2)
    else:
        y_t = project_onto_st(v_t, None, None)

    a_history.append(a_tm1.copy())
    b_history.append(b_tm1.copy())

    a_tminus1 = np.vstack(a_history)
    b_tminus1 = np.vstack(b_history)
    x_next = project_onto_st(y_t, a_tminus1, b_tminus1)
    return x_next, a_history, b_history


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

    print("Start running COCO algorithm...")
    averaged_loss = np.zeros(TOTAL_STEPS)
    averaged_violation = np.zeros(TOTAL_STEPS)

    for total_run in range(ROUND_NUM):
        start_time = time.time()
        choice_history = []
        a_history = []
        b_history = []

        for step in range(1, TOTAL_STEPS + 1):
            x_value, a_history, b_history = coco_centralized(
                step,
                choice_history,
                a_history,
                b_history,
                theta_list,
                a_mat,
                b_vec,
            )
            choice_history.append(x_value)

            if step % 200 == 0:
                print(
                    "COCO (Centralized): Run",
                    total_run + 1,
                    "Step",
                    step,
                    "finished.",
                )

        loss, violation = evaluate_choices(choice_history, theta_list, a_mat, b_vec)
        averaged_loss += loss
        averaged_violation += violation

        end_time = time.time()
        print(f"COCO Run {total_run + 1} elapsed time: {end_time - start_time:.2f} seconds.")

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
    print(f"COCO algorithm completed. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
