# fastapi_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# 🎯 FastAPIアプリ
app = FastAPI()

# 🌐 CORS（Hugging Faceからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 モデルの読み込み
model = joblib.load("models/model_cat.pkl")  # ← Render上のmodelsフォルダに配置

# 📥 入力データ形式
class InputData(BaseModel):
    sex: int
    height: float
    waist: float
    weight_age1: float
    measure: int
    daysold: int

# 📤 推論エンドポイント
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    gain_pred = model.predict(df)[0]
    return {
        "gain_pred": round(gain_pred, 2),
        "status": "success"
    }
