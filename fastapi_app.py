# fastapi_app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import uvicorn
from catboost import CatBoostRegressor

# 🎯 FastAPIアプリ
app = FastAPI()

# 🌐 CORS（Hugging Faceからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://huggingface.co/spaces/drosshopper/horse-weight-predictor2"],  # 本番環境では制限推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 モデルの読み込み
model = joblib.load("models/model_cat.pkl")  # ← Render上のmodelsフォルダに配置
explainer = shap.Explainer(model)

# 📥 入力データ形式
class InputData(BaseModel):
    sex: int
    height: float
    waist: float
    weight_age1: float
    measure: int
    daysold: int




# SHAP値と変化量を返すエンドポイント
@app.post("/predict")
async def predict_with_shap(data: InputData, request: Request):
    referer = request.headers.get("referer", "")
    allowed = "huggingface.co/spaces/drosshopper/horse-weight-predictor2"
    if not referer or allowed not in referer:
        raise HTTPException(status_code=403, detail="外部からのアクセスは許可されていません")
    input_df = pd.DataFrame([data.dict()])
    shap_values = explainer(input_df)


    
    # SHAPの出力から値を取り出し
    base_value = shap_values.base_values[0]
    shap_contributions = shap_values.values[0].tolist()
    feature_names = shap_values.feature_names
    gain_pred = model.predict(input_df)[0]

    return {
        "gain_pred": gain_pred,
        "base_value": base_value,
        "contributions": shap_contributions,
        "features": feature_names
    }
