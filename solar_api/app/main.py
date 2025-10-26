# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing import preprocess_new_data


app = FastAPI(title="☀️ Solar Power Predictor API", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_CAPACITY_W = int(os.getenv("SYSTEM_CAPACITY_W", 303709))

model = load("models/final_xgb_model_V2.pkl")

weather_df = pd.read_pickle("app/Data/test_data.pkl")
weather_df = weather_df.rename(columns={"Time": "timestamp"})
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")

print(f"✅ Weather data loaded: {weather_df.shape}")


class InputData(BaseModel):
    Geff_Reference_Wm2: float
    Geff_Test_Wm2: float
    hour: int
    Module_Surface_Temperature2_degC: float
    Module_Surface_Temperature1_degC: float
    Temperature_Reference_Cell_degC: float
    Temperature_Test_degC: float
    Soiling_Loss_Index_Isc_percent: float
    Soiling_Loss_Index_Geff_percent: float
    Ambient_Temp_degC: float


class TimestampInput(BaseModel):
    timestamp: str


EXPECTED_COLS = [
    'Geff Reference (W/m2)',
    'Geff Test (W/m2)',
    'hour',
    'Module Surface Temperature2 (degree centigrade)',
    'Module Surface Temperature1 (degree centigrade)',
    'Temperature Reference Cell (Deg C)',
    'Temperature Test (Deg C)',
    'Soiling Loss Index Isc (%)',
    'Soiling Loss Index Geff (%)',
    'Ambient Temp. (degree centigrade)'
]


def prepare_for_model(df: pd.DataFrame):
    X = df.reindex(columns=EXPECTED_COLS).fillna(0)
    return X


@app.post("/predict/manual/")
async def predict_manual(data: InputData):
    try:
        df = pd.DataFrame([{
            "Geff Reference (W/m2)": data.Geff_Reference_Wm2,
            "Geff Test (W/m2)": data.Geff_Test_Wm2,
            "hour": data.hour,
            "Module Surface Temperature2 (degree centigrade)": data.Module_Surface_Temperature2_degC,
            "Module Surface Temperature1 (degree centigrade)": data.Module_Surface_Temperature1_degC,
            "Temperature Reference Cell (Deg C)": data.Temperature_Reference_Cell_degC,
            "Temperature Test (Deg C)": data.Temperature_Test_degC,
            "Soiling Loss Index Isc (%)": data.Soiling_Loss_Index_Isc_percent,
            "Soiling Loss Index Geff (%)": data.Soiling_Loss_Index_Geff_percent,
            "Ambient Temp. (degree centigrade)": data.Ambient_Temp_degC
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        if "hour" in X_ready.columns:
            X_ready = X_ready.drop(columns=["hour"])
        y_pred = np.clip(model.predict(X_ready), 0, None)
        predicted_power = float(y_pred[0])
        efficiency = max(0.0, min((predicted_power / SYSTEM_CAPACITY_W) * 100, 100))

        return {
            "Predicted_Power": round(predicted_power, 2),
            "Efficiency": round(efficiency, 2),
            "Capacity_W": SYSTEM_CAPACITY_W,
            "Status": "✅ Success",
            "Features": df.to_dict(orient="records")[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "Geff Test (W/m2)": row["Geff Test (W/m2)"],
            "hour": ts.hour,
            "Module Surface Temperature2 (degree centigrade)": row["Module Surface Temperature2 (degree centigrade)"],
            "Module Surface Temperature1 (degree centigrade)": row["Module Surface Temperature1 (degree centigrade)"],
            "Temperature Reference Cell (Deg C)": row["Temperature Reference Cell (Deg C)"],
            "Temperature Test (Deg C)": row["Temperature Test (Deg C)"],
            "Soiling Loss Index Isc (%)": row["Soiling Loss Index Isc (%)"],
            "Soiling Loss Index Geff (%)": row["Soiling Loss Index Geff (%)"],
            "Ambient Temp. (degree centigrade)": row["Ambient Temp. (degree centigrade)"]
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        if "hour" in X_ready.columns:
            X_ready = X_ready.drop(columns=["hour"])
        y_pred = np.clip(model.predict(X_ready), 0, None)
        predicted_power = float(y_pred[0])
        efficiency = max(0.0, min((predicted_power / SYSTEM_CAPACITY_W) * 100, 100))

        return {
            "Predicted_Power": round(predicted_power, 2),
            "Efficiency": round(efficiency, 2),
            "Capacity_W": SYSTEM_CAPACITY_W,
            "Status": "✅ Success",
            "Features": df.to_dict(orient="records")[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/monthly_summary/")
async def monthly_summary():
    try:
        df = weather_df.copy()
        df["day"] = df["timestamp"].dt.day
        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        if "hour" in X_ready.columns:
            X_ready = X_ready.drop(columns=["hour"])
        df["Predicted_Power"] = np.clip(model.predict(X_ready), 0, None)
        summary = df.groupby("day")["Predicted_Power"].mean().reset_index()
        return {
            "days": summary["day"].tolist(),
            "power": summary["Predicted_Power"].round(2).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/current_stats/")
async def current_stats():
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
            "Geff Test (W/m2)": row["Geff Test (W/m2)"],
            "hour": now.hour,
            "Module Surface Temperature2 (degree centigrade)": row["Module Surface Temperature2 (degree centigrade)"],
            "Module Surface Temperature1 (degree centigrade)": row["Module Surface Temperature1 (degree centigrade)"],
            "Temperature Reference Cell (Deg C)": row["Temperature Reference Cell (Deg C)"],
            "Temperature Test (Deg C)": row["Temperature Test (Deg C)"],
            "Soiling Loss Index Isc (%)": row["Soiling Loss Index Isc (%)"],
            "Soiling Loss Index Geff (%)": row["Soiling Loss Index Geff (%)"],
            "Ambient Temp. (degree centigrade)": row["Ambient Temp. (degree centigrade)"]
        }])

        X_ready = preprocess_new_data(df)
        X_ready = prepare_for_model(X_ready)
        if "hour" in X_ready.columns:
            X_ready = X_ready.drop(columns=["hour"])
        predicted_power = float(np.clip(model.predict(X_ready)[0], 0, None))
        temperature = float(row["Ambient Temp. (degree centigrade)"])
        irradiance = float(row["Geff Reference (W/m2)"])
        efficiency = max(0.0, min((predicted_power / SYSTEM_CAPACITY_W) * 100, 100))

        return {
            "timestamp": str(now),
            "Predicted_Power": round(predicted_power, 2),
            "Temperature": round(temperature, 2),
            "Irradiance": round(irradiance, 2),
            "Efficiency": round(efficiency, 2),
            "Capacity_W": SYSTEM_CAPACITY_W,
            "Status": "✅ Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
