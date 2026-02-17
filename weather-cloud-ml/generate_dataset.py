from __future__ import annotations

import csv
import math
import random
from pathlib import Path

CLOUD_TYPES = ["cirrus", "stratus", "cumulus", "cumulonimbus"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def infer_cloud_type(row: dict[str, float]) -> str:
    humidity = row["humidity"]
    pressure_hpa = row["pressure_hpa"]
    wind_speed_mps = row["wind_speed_mps"]
    cloud_base_km = row["cloud_base_km"]

    if cloud_base_km > 6.5 and humidity < 60:
        return "cirrus"
    if cloud_base_km < 2.0 and humidity > 70 and wind_speed_mps < 7:
        return "stratus"
    if humidity > 78 and wind_speed_mps > 9 and pressure_hpa < 1008:
        return "cumulonimbus"
    return "cumulus"


def create_weather_cloud_dataset(n_samples: int = 12000, seed: int = 7) -> list[dict[str, float | str]]:
    random.seed(seed)
    rows: list[dict[str, float | str]] = []

    for _ in range(n_samples):
        day_of_year = random.randint(1, 365)
        hour = random.randint(0, 23)

        seasonal_temp = 15 + 10 * math.sin(2 * math.pi * day_of_year / 365)
        daily_temp_cycle = 4 * math.sin(2 * math.pi * hour / 24)
        temperature_c = seasonal_temp + daily_temp_cycle + random.gauss(0, 1.8)

        humidity = _clamp(65 - 0.9 * (temperature_c - 15) + random.gauss(0, 8), 20, 100)
        pressure_hpa = 1014 + 5 * math.cos(2 * math.pi * day_of_year / 365) + random.gauss(0, 4)
        wind_speed_mps = _clamp(random.gammavariate(2.3, 2.0), 0, 20)
        dew_point_c = temperature_c - (100 - humidity) / 5 + random.gauss(0, 0.6)
        cloud_base_km = _clamp(0.4 + (temperature_c - dew_point_c) / 5 + random.gauss(0, 0.5), 0.2, 10)
        aerosol_index = _clamp(random.gauss(0.6 + humidity / 180, 0.15), 0.05, 1.8)

        precipitation_mm = _clamp(
            0.15 * (humidity - 68)
            + 0.18 * (10 - cloud_base_km)
            + 0.2 * (wind_speed_mps - 5)
            + random.gauss(0, 1.2),
            0,
            50,
        )

        base_row = {
            "day_of_year": float(day_of_year),
            "hour": float(hour),
            "temperature_c": temperature_c,
            "humidity": humidity,
            "pressure_hpa": pressure_hpa,
            "wind_speed_mps": wind_speed_mps,
            "dew_point_c": dew_point_c,
            "cloud_base_km": cloud_base_km,
            "aerosol_index": aerosol_index,
            "precipitation_mm": precipitation_mm,
        }

        cloud_type = infer_cloud_type(base_row)
        if random.random() < 0.05:
            cloud_type = random.choice(CLOUD_TYPES)

        next_day_temperature_c = (
            temperature_c
            + 0.65 * math.sin(2 * math.pi * (day_of_year + 1) / 365)
            - 0.07 * precipitation_mm
            + random.gauss(0, 1.1)
        )

        row: dict[str, float | str] = {
            **base_row,
            "cloud_type": cloud_type,
            "next_day_temperature_c": next_day_temperature_c,
        }
        rows.append(row)

    return rows


def write_dataset_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    samples = create_weather_cloud_dataset()
    output = Path(__file__).parent / "artifacts" / "weather_cloud_dataset.csv"
    output.parent.mkdir(exist_ok=True)
    write_dataset_csv(samples, output)
    print(f"Wrote {len(samples)} rows to {output}")
