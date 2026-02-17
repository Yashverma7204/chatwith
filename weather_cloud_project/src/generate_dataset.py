import csv
import random
from pathlib import Path

random.seed(42)
N_SAMPLES = 3000
OUTPUT = Path("weather_cloud_project/data/weather_cloud_dataset.csv")

FIELDS = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "cloud_cover_pct",
    "dew_point_c",
    "cloud_type",
    "rain_tomorrow",
]

CLOUD_CLASSES = ["Clear", "Cirrus", "Cumulus", "Cumulonimbus", "Stratus", "Nimbus"]


def clamp(value, low, high):
    return max(low, min(high, value))


def cloud_label(cloud_cover, humidity, temp, pressure, wind_speed):
    if cloud_cover < 15 and humidity < 45:
        return "Clear"
    if cloud_cover > 80 and humidity > 80 and pressure < 1008:
        return "Nimbus"
    if cloud_cover > 60 and temp < 12:
        return "Stratus"
    if cloud_cover > 50 and wind_speed > 25:
        return "Cumulonimbus"
    if cloud_cover > 35:
        return "Cumulus"
    return "Cirrus"


def rain_tomorrow(cloud_cover, humidity, pressure, wind_speed, dew_point, temp):
    score = 0
    score += 2 if cloud_cover > 65 else 0
    score += 2 if humidity > 75 else 0
    score += 2 if pressure < 1009 else 0
    score += 1 if wind_speed > 22 else 0
    score += 1 if dew_point > 14 else 0
    score += 1 if temp < 10 else 0
    return "Rain" if score >= 4 else "No Rain"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(N_SAMPLES):
        temp = clamp(random.gauss(18, 9), -8, 42)
        humidity = clamp(random.gauss(65, 20), 10, 100)
        pressure = clamp(random.gauss(1012, 8), 985, 1035)
        wind_speed = clamp(random.gammavariate(3.2, 4.0), 0, 60)
        cloud_cover = clamp(random.gauss(55, 28), 0, 100)
        dew_point = clamp(temp - (100 - humidity) / 5 + random.gauss(0, 1.2), -15, 28)

        cloud = cloud_label(cloud_cover, humidity, temp, pressure, wind_speed)
        rain = rain_tomorrow(cloud_cover, humidity, pressure, wind_speed, dew_point, temp)

        if random.random() < 0.05:
            cloud = random.choice(CLOUD_CLASSES)
        if random.random() < 0.05:
            rain = "Rain" if rain == "No Rain" else "No Rain"

        rows.append(
            [
                round(temp, 2),
                round(humidity, 2),
                round(pressure, 2),
                round(wind_speed, 2),
                round(cloud_cover, 2),
                round(dew_point, 2),
                cloud,
                rain,
            ]
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)

    print(f"Dataset generated: {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
