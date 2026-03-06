from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.pyfunc

# ================= MLFLOW CONFIG =================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_NAME = "AQI_Best_Model"

print("Loading best model from MLflow Registry...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
    print("Best model loaded successfully")
except Exception as e:
    raise RuntimeError("Failed to load model from MLflow Registry") from e


# ================= PATHS =================
BASE_DIR = r"C:\Users\Ill-Us-Ion\Desktop\aqiproject\src\components"
DATASET_PATH = r"C:\Users\Ill-Us-Ion\Desktop\aqiproject\notebook\aqi_cleaned_processed.csv"

FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "feature_columns.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")


REQUIRED_FILES = [FEATURE_PATH, SCALER_PATH, DATASET_PATH]
for f in REQUIRED_FILES:
    if not os.path.exists(f):
        raise RuntimeError(f"Missing required file: {f}")



# ================= LOAD DATA =================
df = pd.read_csv(DATASET_PATH)
df = pd.read_csv(DATASET_PATH, low_memory=False)

# -------- CLEAN DATE --------
df["date"] = df["date"].replace(
    [999999, "999999", "", "na", "null", "unknown"],
    np.nan
)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -------- CLEAN HOUR --------
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

# keep valid hours only
df.loc[(df["hour"] < 0) | (df["hour"] > 23), "hour"] = np.nan

# convert hour to integer safely
df["hour"] = df["hour"].fillna(df["hour"].mode()[0]).astype(int)

# -------- SAFE DATETIME CREATION --------
df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

# drop invalid rows
df = df.dropna(subset=["datetime"])


feature_columns = joblib.load(FEATURE_PATH)
scaler = joblib.load(SCALER_PATH)



# ================= APP =================
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "AQI Prediction API is running"})



def detect_spikes_series(values, window=3, z_thresh=2.0):
    values = np.array(values)

    spikes = [0] * len(values)

    for i in range(window, len(values)):
        window_vals = values[i-window:i]
        mean = window_vals.mean()
        std = window_vals.std() + 1e-6

        z_score = abs(values[i] - mean) / std

        if z_score > z_thresh:
            spikes[i] = 1

    return spikes

# ================= FEATURE EXTRACTION =================
def extract_features(city, dt):
    city_df = df[df["city"] == city]

    hour_avg = city_df[city_df["hour"] == dt.hour].mean(numeric_only=True)
    if hour_avg.isnull().any():
        hour_avg = city_df.mean(numeric_only=True)

    dow = dt.dayofweek
    month = dt.month

    model_features = {
        "pm2_5": hour_avg["pm2_5"],
        "pm10": hour_avg["pm10"],
        "no2": hour_avg["no2"],
        "so2": hour_avg["so2"],
        "co": hour_avg["co"],
        "o3": hour_avg["o3"],
        "temperature": hour_avg["temperature"],
        "humidity": hour_avg["humidity"],
        "wind_speed": hour_avg["wind_speed"],
        "rainfall": hour_avg["rainfall"],
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    display_features = {
        "pm2_5": round(float(hour_avg["pm2_5"]), 2),
        "pm10": round(float(hour_avg["pm10"]), 2),
        "no2": round(float(hour_avg["no2"]), 2),
        "temperature": round(float(hour_avg["temperature"]), 2),
        "humidity": round(float(hour_avg["humidity"]), 2),
    }

    return model_features, display_features



def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    else:
        return "Very Poor"


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    city = data["city"]
    start_dt = pd.to_datetime(data["start_date"])
    hours = int(data["hours_ahead"])

    future_dates = pd.date_range(start_dt, periods=hours, freq="h")

    result = []
    prev_aqi = None

    for idx, dt in enumerate(future_dates, start=1):

        feats, disp = extract_features(city, dt)

        # lag influence (your existing logic)
        if prev_aqi is not None:
            feats["pm2_5"] *= np.clip(prev_aqi / 100, 0.7, 1.3)

        X = pd.DataFrame([feats]).reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(X)

        aqi_pred = model.predict(
            pd.DataFrame(X_scaled, columns=feature_columns)
        )[0]

        prev_aqi = float(aqi_pred)

        result.append({
            "hour": idx,
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "city": city,
            "predicted_aqi": round(prev_aqi, 2),
            "category": get_aqi_category(prev_aqi),

            # optional table columns
            "pm2_5": disp["pm2_5"],
            "pm10": disp["pm10"],
            "no2": disp["no2"],
            "temperature": disp["temperature"],
            "humidity": disp["humidity"]
        })

    aqi_values = [r["predicted_aqi"] for r in result]

    spikes = detect_spikes_series(aqi_values)

    for i, s in enumerate(spikes):
        result[i]["spike"] = s

    return jsonify({
    "city": city,
    "hours": hours,
    "predictions": result
    })




def detect_spikes(df, column="pm2_5", window=3, threshold=1.5):
    """
    Simple spike detection using rolling mean deviation
    """
    values = df[column]

    rolling_mean = values.rolling(window=window, center=True).mean()

    spikes = (values - rolling_mean).abs() > threshold * rolling_mean.std()

    return spikes.astype(int)


@app.route("/api/pollution-spikes", methods=["GET"])
def get_pollution_spikes():

    city = request.args.get("city")
    start_time = request.args.get("start")
    end_time = request.args.get("end")

    if not all([city, start_time, end_time]):
        return jsonify({"error": "city, start, end are required"}), 400

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    city_df = df[df["city"] == city]

    filtered = city_df[
        (city_df["datetime"] >= start_time) &
        (city_df["datetime"] <= end_time)
    ]

    filtered = filtered.sort_values("datetime")

    filtered["spike"] = detect_spikes(filtered, column="pm2_5")

    spikes_df = filtered[filtered["spike"] == 1]


    return jsonify({
        "timestamps": spikes_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "pm2_5": spikes_df["pm2_5"].tolist(),
        "pm10": spikes_df["pm10"].tolist()
    })


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
