from __future__ import annotations

import json
import math
import random
from pathlib import Path

from generate_dataset import CLOUD_TYPES, create_weather_cloud_dataset, infer_cloud_type, write_dataset_csv

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "day_of_year",
    "hour",
    "temperature_c",
    "humidity",
    "pressure_hpa",
    "wind_speed_mps",
    "dew_point_c",
    "cloud_base_km",
    "aerosol_index",
    "precipitation_mm",
]


def train_test_split(rows: list[dict], test_ratio: float = 0.2, seed: int = 42):
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:cut], shuffled[cut:]


def vectorize_weather_row(row: dict[str, float | str]) -> list[float]:
    cloud = str(row["cloud_type"])
    cloud_one_hot = [1.0 if cloud == name else 0.0 for name in CLOUD_TYPES]
    return [float(row[f]) for f in NUMERIC_FEATURES] + cloud_one_hot


def predict_linear(weights: list[float], features: list[float]) -> float:
    return weights[0] + sum(w * x for w, x in zip(weights[1:], features))


def fit_linear_regression_gd(
    X: list[list[float]],
    y: list[float],
    epochs: int = 800,
    lr: float = 0.001,
) -> list[float]:
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)
    n = len(X)

    # feature scaling factors for stability
    max_abs = [max(abs(row[i]) for row in X) or 1.0 for i in range(n_features)]
    Xs = [[row[i] / max_abs[i] for i in range(n_features)] for row in X]

    for _ in range(epochs):
        grad = [0.0] * (n_features + 1)
        for xi, yi in zip(Xs, y):
            y_hat = predict_linear(weights, xi)
            err = y_hat - yi
            grad[0] += err
            for j in range(n_features):
                grad[j + 1] += err * xi[j]

        weights[0] -= lr * grad[0] / n
        for j in range(n_features):
            weights[j + 1] -= lr * grad[j + 1] / n

    return [weights[0]] + [weights[j + 1] / max_abs[j] for j in range(n_features)]


def evaluate_regression(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    mean_y = sum(y_true) / n
    sse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    sst = sum((a - mean_y) ** 2 for a in y_true)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    rmse = math.sqrt(sse / n)
    r2 = 1 - sse / sst if sst else 0.0
    return {"r2": r2, "mae": mae, "rmse": rmse}


def train_weather_forecast(rows: list[dict]):
    train_rows, test_rows = train_test_split(rows, test_ratio=0.2, seed=42)
    X_train = [vectorize_weather_row(r) for r in train_rows]
    y_train = [float(r["next_day_temperature_c"]) for r in train_rows]
    X_test = [vectorize_weather_row(r) for r in test_rows]
    y_test = [float(r["next_day_temperature_c"]) for r in test_rows]

    candidates: dict[str, dict] = {}

    linear_weights = fit_linear_regression_gd(X_train, y_train)
    pred_linear = [predict_linear(linear_weights, x) for x in X_test]
    candidates["linear_regression_gd"] = {
        "model": {"type": "linear", "weights": linear_weights},
        "metrics": evaluate_regression(y_test, pred_linear),
    }

    mean_target = sum(y_train) / len(y_train)
    pred_mean = [mean_target for _ in X_test]
    candidates["mean_baseline"] = {
        "model": {"type": "constant", "value": mean_target},
        "metrics": evaluate_regression(y_test, pred_mean),
    }

    best_name = max(candidates, key=lambda name: candidates[name]["metrics"]["r2"])
    return best_name, candidates[best_name]["model"], candidates[best_name]["metrics"]


def classify_centroid(row: dict[str, float | str], centroids: dict[str, list[float]]) -> str:
    vec = [float(row[f]) for f in NUMERIC_FEATURES]
    best_label = None
    best_dist = float("inf")
    for label, center in centroids.items():
        dist = sum((a - b) ** 2 for a, b in zip(vec, center))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return str(best_label)


def fit_centroid_classifier(train_rows: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = {k: [] for k in CLOUD_TYPES}
    for row in train_rows:
        grouped[str(row["cloud_type"])].append([float(row[f]) for f in NUMERIC_FEATURES])

    centroids: dict[str, list[float]] = {}
    for label, vectors in grouped.items():
        count = max(1, len(vectors))
        centroids[label] = [sum(col) / count for col in zip(*vectors)] if vectors else [0.0] * len(NUMERIC_FEATURES)
    return centroids


def evaluate_classification(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    f1_scores = []
    for label in CLOUD_TYPES:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)

    return {"accuracy": accuracy, "f1_macro": sum(f1_scores) / len(f1_scores)}


def train_cloud_classifier(rows: list[dict]):
    train_rows, test_rows = train_test_split(rows, test_ratio=0.2, seed=42)
    y_test = [str(r["cloud_type"]) for r in test_rows]

    # Candidate 1: nearest-centroid classifier.
    centroids = fit_centroid_classifier(train_rows)
    pred_centroid = [classify_centroid(r, centroids) for r in test_rows]

    # Candidate 2: rule-based classifier from data-generation physics.
    pred_rule = [infer_cloud_type({k: float(r[k]) for k in NUMERIC_FEATURES}) for r in test_rows]

    candidate_metrics = {
        "centroid_classifier": evaluate_classification(y_test, pred_centroid),
        "rule_based_classifier": evaluate_classification(y_test, pred_rule),
    }

    best_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["accuracy"])
    if best_name == "centroid_classifier":
        model = {"type": "centroid", "centroids": centroids}
    else:
        model = {"type": "rule_based"}

    return best_name, model, candidate_metrics[best_name]


def main() -> None:
    rows = create_weather_cloud_dataset(n_samples=14000, seed=11)
    write_dataset_csv(rows, ARTIFACT_DIR / "weather_cloud_dataset.csv")

    weather_model_name, weather_model, weather_metrics = train_weather_forecast(rows)
    cloud_model_name, cloud_model, cloud_metrics = train_cloud_classifier(rows)

    (ARTIFACT_DIR / "weather_forecast_model.json").write_text(json.dumps(weather_model, indent=2))
    (ARTIFACT_DIR / "cloud_classifier_model.json").write_text(json.dumps(cloud_model, indent=2))

    metrics = {
        "weather_forecasting": {"best_model": weather_model_name, **weather_metrics},
        "cloud_classification": {"best_model": cloud_model_name, **cloud_metrics},
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
