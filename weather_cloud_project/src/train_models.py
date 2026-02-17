import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path("weather_cloud_project/data/weather_cloud_dataset.csv")
MODELS_DIR = Path("weather_cloud_project/models")
REPORT_PATH = MODELS_DIR / "metrics_report.json"

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "cloud_cover_pct",
    "dew_point_c",
]


class GaussianNB:
    def fit(self, x, y):
        grouped = defaultdict(list)
        for features, label in zip(x, y):
            grouped[label].append(features)

        self.classes_ = sorted(grouped)
        self.priors_ = {}
        self.stats_ = {}
        total = len(x)

        for label, rows in grouped.items():
            self.priors_[label] = len(rows) / total
            transposed = list(zip(*rows))
            self.stats_[label] = []
            for values in transposed:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
                self.stats_[label].append((mean, variance + 1e-6))

    def _log_probability(self, x, label):
        log_prob = math.log(self.priors_[label])
        for value, (mean, var) in zip(x, self.stats_[label]):
            log_prob += -0.5 * math.log(2 * math.pi * var) - ((value - mean) ** 2) / (2 * var)
        return log_prob

    def predict(self, x):
        preds = []
        for row in x:
            best = max(self.classes_, key=lambda c: self._log_probability(row, c))
            preds.append(best)
        return preds


class KNN:
    def __init__(self, k=7):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    @staticmethod
    def _distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(self, x):
        preds = []
        for row in x:
            distances = [
                (self._distance(row, train_row), label)
                for train_row, label in zip(self.x_train, self.y_train)
            ]
            distances.sort(key=lambda t: t[0])
            nearest = [label for _, label in distances[: self.k]]
            preds.append(Counter(nearest).most_common(1)[0][0])
        return preds


def load_data():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    x = [[float(r[f]) for f in FEATURES] for r in rows]
    y_cloud = [r["cloud_type"] for r in rows]
    y_rain = [r["rain_tomorrow"] for r in rows]
    return x, y_cloud, y_rain


def train_test_split_stratified(x, y, test_size=0.2, seed=42):
    random.seed(seed)
    by_class = defaultdict(list)
    for features, label in zip(x, y):
        by_class[label].append((features, label))

    train, test = [], []
    for label_rows in by_class.values():
        random.shuffle(label_rows)
        split_idx = int(len(label_rows) * (1 - test_size))
        train.extend(label_rows[:split_idx])
        test.extend(label_rows[split_idx:])

    random.shuffle(train)
    random.shuffle(test)

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return list(x_train), list(x_test), list(y_train), list(y_test)


def accuracy_score(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def evaluate_task(name, x, y):
    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)

    candidates = {
        "gaussian_nb": GaussianNB(),
        "knn_k7": KNN(k=7),
        "knn_k11": KNN(k=11),
    }

    results = {}
    best_name = None
    best_acc = -1

    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        results[model_name] = round(acc, 4)
        if acc > best_acc:
            best_acc = acc
            best_name = model_name

    return {
        "task": name,
        "best_model": best_name,
        "best_accuracy": round(best_acc, 4),
        "candidate_accuracies": results,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    x, y_cloud, y_rain = load_data()

    cloud_result = evaluate_task("cloud_classification", x, y_cloud)
    rain_result = evaluate_task("weather_forecast", x, y_rain)

    report = {
        "cloud_classification": cloud_result,
        "weather_forecast": rain_result,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"cloud_classification best={cloud_result['best_model']} accuracy={cloud_result['best_accuracy']}"
    )
    print(f"weather_forecast best={rain_result['best_model']} accuracy={rain_result['best_accuracy']}")
    print(f"Saved metrics report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
