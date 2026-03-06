import streamlit as st
import os
import joblib
import json
import pandas as pd
import numpy as np
import requests
import datetime
import time
from dotenv import load_dotenv
from google import genai

# ==========================================
# 1. API KEYS & CONFIGURATION
# ==========================================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

if not OPENWEATHER_API_KEY:
    st.error("OPENWEATHER_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. LOAD ML ARTIFACTS
# ==========================================

@st.cache_resource
def load_ml_assets():
    BASE_DIR = r"C:\Users\Ill-Us-Ion\Desktop\aqiproject\src\components\artifacts"
    try:
        model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        features = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

        meta_path = os.path.join(BASE_DIR, "model_meta.json")
        model_type = "unknown"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                model_type = json.load(f).get("model_type", "unknown")

        return model, scaler, features, model_type

    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None


best_model, scaler, features_list, model_type = load_ml_assets()

if best_model is None:
    st.stop()

# ==========================================
# 3. AQI TOOL FUNCTIONS
# ==========================================

def get_aqi_prediction(city: str) -> dict:
    try:
        # Geocoding
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_data = requests.get(geo_url, timeout=10).json()

        if not geo_data:
            return {"error": f"City '{city}' not found"}

        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

        # APIs
        pol_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        met_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

        pol_res = requests.get(pol_url, timeout=10).json()["list"][0]["components"]
        met_res = requests.get(met_url, timeout=10).json()

        now = datetime.datetime.now()

        data = {
            "pm2_5": pol_res["pm2_5"],
            "pm10": pol_res["pm10"],
            "no2": pol_res["no2"],
            "so2": pol_res["so2"],
            "co": pol_res["co"],
            "o3": pol_res["o3"],
            "temperature": met_res["main"]["temp"],
            "humidity": met_res["main"]["humidity"],
            "wind_speed": met_res["wind"]["speed"],
            "rainfall": met_res.get("rain", {}).get("1h", 0),
            "hour_sin": np.sin(2 * np.pi * now.hour / 24),
            "hour_cos": np.cos(2 * np.pi * now.hour / 24),
            "dow_sin": np.sin(2 * np.pi * now.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * now.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * now.month / 12),
            "month_cos": np.cos(2 * np.pi * now.month / 12),
        }

        input_df = pd.DataFrame([data])[features_list]
        input_scaled = scaler.transform(input_df)
        pred = float(best_model.predict(input_scaled)[0])

        return {
            "city": city,
            "predicted_aqi": round(pred, 2),
            "pollutants": pol_res,
            "temperature_c": data["temperature"],
            "condition": met_res["weather"][0]["description"]
        }

    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=900)
def get_aqi_prediction_cached(city: str) -> dict:
    return get_aqi_prediction(city)

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="AQI Forecast AI", page_icon="🌬️")
st.title("🌬️ AQI Forecasting AI Dashboard")

MODEL_ID = "gemini-2.5-flash"

st.caption(f"2026 Edition | Using {MODEL_ID}")

instruction = (
    "You are an AQI expert. Use the get_aqi_prediction_cached tool to fetch real-time data "
    "and predict AQI for a city in India. Analyze pollutants and provide health advice "
    "based on CPCB India AQI categories."
)

# ==========================================
# 5. CHAT INITIALIZATION (NEW SDK)
# ==========================================

if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model=MODEL_ID,
        config={
            "system_instruction": instruction,
            "tools": [get_aqi_prediction_cached]
        }
    )
    st.session_state.messages = []
    st.success(f"Using model: {MODEL_ID}")

# ==========================================
# 6. CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 7. SAFE SEND WITH RETRY
# ==========================================

def safe_send(chat, prompt):
    for _ in range(3):
        try:
            return chat.send_message(prompt)
        except Exception as e:
            if "429" in str(e):
                time.sleep(2)
            else:
                raise

# ==========================================
# 8. USER INPUT
# ==========================================

if prompt := st.chat_input("What is the air quality in Mumbai?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = safe_send(st.session_state.chat, prompt)
            reply = response.text
            st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"Chat Error: {e}")
