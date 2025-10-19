
# ☀️ Solar Power Predictor AI

An **AI-powered system** that predicts solar energy generation using a trained **XGBoost model**.  
Built with **FastAPI** for the backend and a **modern interactive dashboard (HTML + JS + Chart.js)** for visualization.  
Users can predict power output by selecting a date/time or entering weather values manually.

---

## 🚀 Features

- 🧠 Accurate solar power predictions using XGBoost  
- 🌐 RESTful API built with FastAPI  
- 💻 Interactive front-end dashboard with live charts  
- 📅 Predict by timestamp or manual input  
- 📊 Visualization of daily and monthly generation trends  
- 🧹 Clean preprocessing and error handling  

---

## 📂 Project Structure

```

solar_api/
│
├── app/
│   ├── main.py                # FastAPI main application
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── static/                # Front-end files (HTML, JS, CSS)
│   │   └── index.html
│   └── Data/
│       └── df5_clean.pkl      # Historical weather data
│
├── models/
│   └── final_xgb_model.pkl    # Trained XGBoost model
│
├── requirements.txt           # Project dependencies
└── README.md

````

---

## ⚙️ Installation & Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Run FastAPI server

```bash
uvicorn app.main:app --reload
```

Then open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔍 Example Requests

### Manual Input

```bash
POST /predict/manual/
Content-Type: application/json

{
  "Geff_Reference_Wm2": 203.1,
  "Ambient_Temp_degC": 6.0,
  "Relative_Humidity": 66.3,
  "Wind_Speed_ms": 0.9,
  "Temperature_Test_degC": 7.7,
  "Soiling_Geff": 0,
  "Soiling_Isc": 0,
  "GHI_Wm2": 98.0,
  "POA1_Wm2": 127.6,
  "hour": 8,
  "month": 1
}
```

### Predict by Timestamp

```bash
POST /predict/by_time/
Content-Type: application/json

{
  "timestamp": "2025-01-01 08:00:00"
}
```

---

## 📊 Example Response

```json
{
  "Predicted_Power": 51493.57,
  "Status": "✅ Success",
  "Message": "Prediction generated successfully."
}
```

---

## 🧩 Tech Stack

| Component                | Description             |
| ------------------------ | ----------------------- |
| **Python (FastAPI)**     | Backend REST API        |
| **XGBoost**              | Machine learning model  |
| **Pandas / NumPy**       | Data processing         |
| **HTML + JS + Chart.js** | Front-end visualization |
| **Uvicorn**              | Local server runtime    |

---

