# Weather Forecast & Cloud Classification Project

This project trains two machine-learning pipelines on a generated meteorological dataset:

1. **Weather Forecasting (Regression)**: predicts the next-day temperature.
2. **Cloud Classification (Multiclass Classification)**: predicts cloud type (`cirrus`, `stratus`, `cumulus`, `cumulonimbus`).

The training script runs multiple candidate models for each task and selects the one with the best validation score.

## Project structure

- `generate_dataset.py` – creates a realistic synthetic weather + cloud dataset.
- `train.py` – trains, evaluates, and saves best-performing models.
- `artifacts/` – generated outputs (dataset, metrics, and model files).

## Quick start

```bash
cd weather-cloud-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Output

After running training, the script writes:

- `artifacts/weather_cloud_dataset.csv`
- `artifacts/weather_forecast_model.json`
- `artifacts/cloud_classifier_model.json`
- `artifacts/metrics.json`

`metrics.json` contains:
- Best weather model and test metrics (`r2`, `mae`, `rmse`)
- Best cloud model and test metrics (`accuracy`, `f1_macro`)
