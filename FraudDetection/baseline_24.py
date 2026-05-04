from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


SEED = 53
BATCH_SIZE = 50
DEVICE = torch.device("cpu")
ALGORITHM_LABEL = "[24]"
OUTPUT_FILE = "baseline_24_results.npz"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "creditcard.csv"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing required data file: creditcard.csv. "
            "Place it in the FraudDetection folder before running this script."
        )

    raw_data = pd.read_csv(DATA_PATH)
    required_columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing_columns = [column for column in required_columns if column not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"creditcard.csv is missing columns: {missing_columns}")
    return raw_data


class SigmoidModel(nn.Module):
    def __init__(self, input_size=29, hidden_size=10, output_size=1):
        super(SigmoidModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


loss_fn = torch.nn.BCELoss()


def zero_tensor():
    return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)


def positive_part(value):
    return torch.clamp(value, min=0.0)


def soft_threshold(vector, threshold):
    return torch.sign(vector) * torch.clamp(torch.abs(vector) - threshold, min=0.0)


def prepare_experiment_state(raw_data):
    set_seed(SEED)
    data = raw_data.sample(frac=1)
    z = np.array(data.iloc[:, 1:30])
    y = np.array(data["Class"])

    scaler = StandardScaler()
    z = scaler.fit_transform(z)

    sample_length = len(z)
    step = (sample_length - 1) // BATCH_SIZE

    model = SigmoidModel(input_size=29, hidden_size=10, output_size=1)
    model.to(DEVICE)

    constraint_upper = (torch.rand(step) * 10 + 10).to(DEVICE)
    return z, y, step, model, constraint_upper


def make_batch(z, y, t):
    start_idx = t * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch_z = torch.tensor(z[start_idx:end_idx, :], dtype=torch.float32).to(DEVICE)
    batch_y = torch.tensor(y[start_idx:end_idx], dtype=torch.float32).to(DEVICE)
    return batch_z, batch_y


def compute_prediction_loss(model, z, y, t):
    batch_z, batch_y = make_batch(z, y, t)
    hat_y = model(batch_z)
    hat_y = torch.clamp(hat_y, min=1e-7, max=1 - 1e-7)
    return loss_fn(torch.squeeze(hat_y), batch_y)


def flatten_parameters(model):
    return torch.cat([
        param.detach().reshape(-1)
        for param in model.parameters()
        if param.requires_grad
    ])


def flatten_gradients(model):
    gradients = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            gradients.append(torch.zeros_like(param).reshape(-1))
        else:
            gradients.append(param.grad.detach().reshape(-1))
    return torch.cat(gradients)


def assign_flat_parameters(model, vector):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            count = param.numel()
            param.copy_(vector[offset : offset + count].reshape_as(param))
            offset += count


def loss_value_and_gradient(model, z, y, t):
    model.zero_grad(set_to_none=True)
    loss = compute_prediction_loss(model, z, y, t)
    loss.backward()
    return loss.item(), flatten_gradients(model)


def parameter_constraint_value(vector, bound):
    bound = torch.as_tensor(bound, dtype=vector.dtype, device=vector.device)
    return torch.norm(vector, p=1) + 0.5 * torch.sum(vector * vector) - bound


def hard_violation(vector, bound):
    return positive_part(parameter_constraint_value(vector, bound))


def hard_constraint_gradient(vector, bound):
    if parameter_constraint_value(vector, bound) > 0:
        return torch.sign(vector) + vector
    return torch.zeros_like(vector)


def project_to_constraint(vector, bound, max_iter=80):
    bound = torch.as_tensor(bound, dtype=vector.dtype, device=vector.device)
    if parameter_constraint_value(vector, bound) <= 0:
        return vector.clone()

    low = torch.tensor(0.0, dtype=vector.dtype, device=vector.device)
    high = torch.tensor(1.0, dtype=vector.dtype, device=vector.device)

    def projected(mu):
        return soft_threshold(vector, mu) / (1.0 + mu)

    while parameter_constraint_value(projected(high), bound) > 0:
        high = high * 2.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        if parameter_constraint_value(projected(mid), bound) > 0:
            low = mid
        else:
            high = mid

    return projected(high)


def distance_and_gradient_to_constraint(vector, bound, scale=1.0):
    projection = project_to_constraint(vector, bound)
    diff = vector - projection
    distance = torch.norm(diff)
    if distance < 1e-10:
        return zero_tensor(), torch.zeros_like(vector)
    return scale * distance, scale * diff / distance


def append_metrics(losses, violations, time_loss, time_violation, loss_item, model, constraint_upper, t):
    time_loss += loss_item
    vector = flatten_parameters(model)
    time_violation += hard_violation(vector, constraint_upper[t]).item()
    losses.append(time_loss)
    violations.append(time_violation)
    return time_loss, time_violation


def run_native_sgd(raw_data):
    z, y, step, model, constraint_upper = prepare_experiment_state(raw_data)

    d_lip = 20.0
    radius = 1.0
    gradient_norms = []
    averaged_loss = []
    averaged_violation = []
    time_loss = 0.0
    time_violation = 0.0

    for t in range(1, step):
        model.train()
        loss_item, loss_grad = loss_value_and_gradient(model, z, y, t)
        vector = flatten_parameters(model)

        _, distance_grad = distance_and_gradient_to_constraint(
            vector,
            constraint_upper[t - 1],
            scale=d_lip,
        )
        constraint_grad = hard_constraint_gradient(vector, constraint_upper[t - 1])
        constraint_weight = 2.0
        gradient = loss_grad + distance_grad + constraint_weight * constraint_grad

        gradient_norm = torch.sum(gradient * gradient)
        gradient_norms.append(gradient_norm)
        eta = 2 * np.sqrt(2) * radius / torch.sqrt(torch.stack(gradient_norms).sum() + 1e-8)
        new_vector = vector - eta * gradient

        assign_flat_parameters(model, new_vector)
        time_loss, time_violation = append_metrics(
            averaged_loss,
            averaged_violation,
            time_loss,
            time_violation,
            loss_item,
            model,
            constraint_upper,
            t,
        )

        if (t + 1) % 1000 == 0:
            print(f"NativeSGD {t + 1}")

    return {"loss": averaged_loss, "violation": averaged_violation}


def run_experiment():
    raw_data = load_raw_data()
    print(f"Loaded {DATA_PATH.name}: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
    print("Start running NativeSGD algorithm...")
    started_at = time.time()
    result = run_native_sgd(raw_data)
    elapsed = time.time() - started_at

    loss = np.array(result["loss"])
    violation = np.array(result["violation"])
    output_path = SCRIPT_DIR / OUTPUT_FILE
    np.savez(
        output_path,
        loss=loss,
        violation=violation,
        total_steps=len(loss),
        algorithm_label=ALGORITHM_LABEL,
    )
    print(f"NativeSGD algorithm completed in {elapsed:.2f} seconds. Results saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
