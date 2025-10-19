# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from joblib import load
import pandas as pd
import numpy as np
from app.preprocessing import preprocess_new_data

# ======================================
# 🚀 FastAPI Setup
# ======================================
app = FastAPI(title="☀️ Solar Power Predictor API", version="2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 🔹 Load Model & Data
# ======================================
model = load("models/final_xgb_model.pkl")

weather_df = pd.read_pickle("app/Data/df5_clean.pkl")
weather_df = weather_df.rename(columns={"Time": "timestamp"})
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")

print(f"✅ Weather data loaded: {weather_df.shape}")

# ======================================
# 🔹 Schemas
# ======================================
class InputData(BaseModel):
    Geff_Reference_Wm2: float
    Ambient_Temp_degC: float
    Relative_Humidity: float
    Wind_Speed_ms: float
    Temperature_Test_degC: float
    Soiling_Geff: float
    Soiling_Isc: float
    GHI_Wm2: float
    POA1_Wm2: float
    hour: int
    month: int

class TimestampInput(BaseModel):
    timestamp: str

# ======================================
# 🔹 Helper
# ======================================
EXPECTED_COLS = [
    'Total Solar Irradiance on Inclined Plane POA1 (W/m2)',
    'Total Solar Irradiance on Horizontal Plane GHI (W/m2)',
    'Geff Reference (W/m2)',
    'hour',
    'Temperature Test (Deg C)',
    'Soiling Loss Index Isc (%)',
    'Soiling Loss Index Geff (%)',
    'Ambient Temp. (degree centigrade)',
    'Relative Humidity (%)',
    'Wind Speed (m/s)',
    'month'
]

def prepare_for_model(df: pd.DataFrame):
    X = df.reindex(columns=EXPECTED_COLS).fillna(0)
    return X

# ======================================
# 🟢 Manual Prediction
# ======================================
@app.post("/predict/manual/")
async def predict_manual(data: InputData):
    try:
        df = pd.DataFrame([{
            "Geff Reference (W/m2)": data.Geff_Reference_Wm2,
            "Ambient Temp. (degree centigrade)": data.Ambient_Temp_degC,
            "Relative Humidity (%)": data.Relative_Humidity,
            "Wind Speed (m/s)": data.Wind_Speed_ms,
            "Temperature Test (Deg C)": data.Temperature_Test_degC,
            "Soiling Loss Index Geff (%)": data.Soiling_Geff,
            "Soiling Loss Index Isc (%)": data.Soiling_Isc,
            "Total Solar Irradiance on Horizontal Plane GHI (W/m2)": data.GHI_Wm2,
            "Total Solar Irradiance on Inclined Plane POA1 (W/m2)": data.POA1_Wm2,
            "hour": data.hour,
            "month": data.month
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        y_pred = np.clip(model.predict(X_ready), 0, None)

        return {
            "Predicted_Power": round(float(y_pred[0]), 2),
            "Status": "✅ Success",
            "Features": df.to_dict(orient="records")[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# 🟣 Predict by Timestamp
# ======================================
@app.post("/predict/by_time/")
async def predict_by_time(data: TimestampInput):
    try:
        ts = pd.to_datetime(data.timestamp, errors="coerce")
        if pd.isna(ts):
            raise HTTPException(status_code=422, detail="Invalid timestamp format")

        match_df = weather_df[
            (weather_df["timestamp"].dt.month == ts.month) &
            (weather_df["timestamp"].dt.day == ts.day) &
            (weather_df["timestamp"].dt.hour == ts.hour)
        ]

        if match_df.empty:
            match_df = weather_df[
                (weather_df["timestamp"].dt.month == ts.month) &
                (weather_df["timestamp"].dt.hour == ts.hour)
            ]

        if match_df.empty:
            raise HTTPException(status_code=404, detail="No record found for that timestamp.")

        row = match_df.iloc[0]

        df = pd.DataFrame([{
            "Geff Reference (W/m2)": row["Geff Reference (W/m2)"],
            "Ambient Temp. (degree centigrade)": row["Ambient Temp. (degree centigrade)"],
            "Relative Humidity (%)": row["Relative Humidity (%)"],
            "Wind Speed (m/s)": row["Wind Speed (m/s)"],
            "Temperature Test (Deg C)": row["Temperature Test (Deg C)"],
            "Soiling Loss Index Geff (%)": row["Soiling Loss Index Geff (%)"],
            "Soiling Loss Index Isc (%)": row["Soiling Loss Index Isc (%)"],
            "Total Solar Irradiance on Horizontal Plane GHI (W/m2)": row["Total Solar Irradiance on Horizontal Plane GHI (W/m2)"],
            "Total Solar Irradiance on Inclined Plane POA1 (W/m2)": row["Total Solar Irradiance on Inclined Plane POA1 (W/m2)"],
            "hour": ts.hour,
            "month": ts.month
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        y_pred = np.clip(model.predict(X_ready), 0, None)

        return {
            "Predicted_Power": round(float(y_pred[0]), 2),
            "Status": "✅ Success",
            "Features": df.to_dict(orient="records")[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# 📊 Data Endpoints
# ======================================
@app.get("/data/monthly_summary/")
async def monthly_summary():
    """Return average predicted power per day using model."""
    try:
        df = weather_df.copy()
        df["day"] = df["timestamp"].dt.day
        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        df["Predicted_Power"] = np.clip(model.predict(X_ready), 0, None)
        summary = df.groupby("day")["Predicted_Power"].mean().reset_index()
        return {"days": summary["day"].tolist(), "power": summary["Predicted_Power"].round(2).tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/current_stats/")
async def current_stats():
    """Return current dashboard values from CSV and model."""
    try:
        now = pd.Timestamp.now()
        match = weather_df[
            (weather_df["timestamp"].dt.month == now.month) &
            (weather_df["timestamp"].dt.day == now.day) &
            (weather_df["timestamp"].dt.hour == now.hour)
        ]
        if match.empty:
            match = weather_df.tail(1)

        row = match.iloc[0]
        df = pd.DataFrame([{
            "Geff Reference (W/m2)": row["Geff Reference (W/m2)"],
            "Ambient Temp. (degree centigrade)": row["Ambient Temp. (degree centigrade)"],
            "Relative Humidity (%)": row["Relative Humidity (%)"],
            "Wind Speed (m/s)": row["Wind Speed (m/s)"],
            "Temperature Test (Deg C)": row["Temperature Test (Deg C)"],
            "Soiling Loss Index Geff (%)": row["Soiling Loss Index Geff (%)"],
            "Soiling Loss Index Isc (%)": row["Soiling Loss Index Isc (%)"],
            "Total Solar Irradiance on Horizontal Plane GHI (W/m2)": row["Total Solar Irradiance on Horizontal Plane GHI (W/m2)"],
            "Total Solar Irradiance on Inclined Plane POA1 (W/m2)": row["Total Solar Irradiance on Inclined Plane POA1 (W/m2)"],
            "hour": now.hour,
            "month": now.month
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        predicted_power = float(np.clip(model.predict(X_ready)[0], 0, None))
        temperature = float(row["Ambient Temp. (degree centigrade)"])
        irradiance = float(row["Geff Reference (W/m2)"])
        efficiency = round(predicted_power / 70000 * 100, 2)

        return {
            "timestamp": str(now),
            "Predicted_Power": round(predicted_power, 2),
            "Temperature": round(temperature, 2),
            "Irradiance": round(irradiance, 2),
            "Efficiency": efficiency,
            "Status": "✅ Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================
# 🌐 Serve Frontend
# ======================================
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
