from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "WindSpeedKmph",
    "CloudCoverOktas",
    "SunshineHours",
    "Rainfall_mm",
    "RainToday",
    "RainTomorrow",
    "RainProbabilityPercent",
    "Year",
    "Month",
    "Day",
]

app = Flask(__name__)
app.secret_key = "weather-prediction-local"


def find_model(model_number: int) -> Path:
    candidates = [
        BASE_DIR / f"random_forest_model{model_number}.pkl",
        BASE_DIR / f"D__WEATHER_random_forest_model{model_number}.pkl",
    ]
    candidates.extend(BASE_DIR.glob(f"*random_forest_model{model_number}.pkl"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find random_forest_model{model_number}.pkl in {BASE_DIR}"
    )


def load_model(model_number: int) -> Any:
    with find_model(model_number).open("rb") as model_file:
        return pickle.load(model_file)


def model_columns(model: Any) -> list[str]:
    trained_columns = getattr(model, "feature_names_in_", None)
    if trained_columns is not None:
        return [str(column) for column in trained_columns]
    return FEATURE_COLUMNS


def to_number(field_name: str, label: str) -> float:
    raw_value = request.form.get(field_name, "").strip()
    if raw_value == "":
        raise ValueError(f"{label} is required.")
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error


def build_features() -> pd.DataFrame:
    selected_date = request.form.get("date", "").strip()
    parsed_date = date.fromisoformat(selected_date) if selected_date else date.today()

    values = {
        "MinTemp": to_number("min_temp", "Minimum temperature"),
        "MaxTemp": to_number("max_temp", "Maximum temperature"),
        "Humidity9am": to_number("humidity_9am", "9am humidity"),
        "Humidity3pm": to_number("humidity_3pm", "3pm humidity"),
        "Pressure9am": to_number("pressure_9am", "9am pressure"),
        "Pressure3pm": to_number("pressure_3pm", "3pm pressure"),
        "WindSpeedKmph": to_number("wind_speed", "Wind speed"),
        "CloudCoverOktas": to_number("cloud_cover", "Cloud cover"),
        "SunshineHours": to_number("sunshine", "Sunshine hours"),
        "Rainfall_mm": to_number("rainfall", "Rainfall"),
        "RainToday": int(request.form.get("rain_today", "0")),
        "RainTomorrow": 0,
        "RainProbabilityPercent": to_number(
            "rain_probability", "Current rain probability"
        ),
        "Year": parsed_date.year,
        "Month": parsed_date.month,
        "Day": parsed_date.day,
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def classification_result(model: Any, features: pd.DataFrame) -> tuple[bool, float]:
    columns = model_columns(model)
    model_input = features.reindex(columns=columns)
    prediction = int(model.predict(model_input)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        rain_index = classes.index(1) if 1 in classes else -1
        rain_probability = float(probabilities[rain_index]) if rain_index >= 0 else float(prediction)
    else:
        rain_probability = float(prediction)
    return bool(prediction), max(0.0, min(100.0, rain_probability * 100))


def percentage_result(model: Any, features: pd.DataFrame) -> float:
    columns = model_columns(model)
    model_input = features.reindex(columns=columns)
    if hasattr(model, "predict_proba") and getattr(model, "classes_", None) is not None:
        probabilities = model.predict_proba(model_input)[0]
        classes = list(model.classes_)
        rain_index = classes.index(1) if 1 in classes else -1
        value = float(probabilities[rain_index]) if rain_index >= 0 else float(probabilities[-1])
    else:
        value = float(model.predict(model_input)[0])
    if 0 <= value <= 1:
        value *= 100
    return max(0.0, min(100.0, value))


@app.get("/")
def index():
    return render_template("index.html", today=date.today().isoformat())


@app.post("/predict")
def predict():
    try:
        features = build_features()
        classification_model = load_model(1)
        percentage_model = load_model(2)
        will_rain, class_probability = classification_result(classification_model, features)
        predicted_percentage = percentage_result(percentage_model, features)
        combined_probability = round((class_probability + predicted_percentage) / 2, 1)
        result = {
            "will_rain": will_rain,
            "label": "Rain expected" if will_rain else "Dry tomorrow",
            "percentage": round(predicted_percentage, 1),
            "confidence": combined_probability,
        }
        return render_template("index.html", today=date.today().isoformat(), result=result)
    except (FileNotFoundError, ValueError, OSError, pickle.UnpicklingError) as error:
        flash(str(error), "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
