# Weather Forecast & Cloud Classification Project

This project builds two machine-learning models from a weather dataset:

1. **Cloud classification** (Clear, Cirrus, Cumulus, Cumulonimbus, Stratus, Nimbus)
2. **Weather forecast** (Rain / No Rain for tomorrow)

It trains multiple classifiers and automatically keeps the **highest accuracy** model for each task.

## Project Structure

- `data/weather_cloud_dataset.csv` – generated dataset used for training
- `src/generate_dataset.py` – creates reproducible weather + cloud labels
- `src/train_models.py` – trains candidate models, selects the best by accuracy
- `models/metrics_report.json` – final model accuracy report

## Quick Start

```bash
python weather_cloud_project/src/generate_dataset.py
python weather_cloud_project/src/train_models.py
```

## Current Training Result

From the latest training run:

- Cloud classification best model: `knn_k7` with **0.8870** accuracy
- Weather forecast best model: `knn_k11` with **0.8869** accuracy

See complete per-model scores in `weather_cloud_project/models/metrics_report.json`.

## Notes

- The dataset is synthetic but weather-informed and includes controlled label noise.
- Candidate models compared:
  - Gaussian Naive Bayes
  - KNN (k=7)
  - KNN (k=11)
