# Skycast weather predictor

A Flask web app that predicts tomorrow's rain classification and rain percentage using the supplied random forest models.

## Project files

- `app.py` - Flask routes, input validation, model loading, and prediction logic.
- `templates/index.html` - Responsive weather input and result screen.
- `static/styles.css` - Visual styling.
- `random_forest_model1.pkl` - Rain/no-rain classifier.
- `random_forest_model2.pkl` - Rain percentage model.

The app also recognizes the current exported filenames `D__WEATHER_random_forest_model1.pkl` and `D__WEATHER_random_forest_model2.pkl`.

## Run on Windows

1. Open PowerShell in `D:\WEATHER`.
2. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies and start Flask:

```powershell
python -m pip install -r requirements.txt
python app.py
```

4. Open https://weather-n4kr.onrender.com in a browser.

The form uses the 16 columns shown in the supplied sample. If a loaded scikit-learn model contains `feature_names_in_`, the app automatically follows that trained feature order.
