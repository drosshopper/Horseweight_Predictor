# fastapi_app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np
from scipy.spatial import distance
import uvicorn
from catboost import CatBoostRegressor


# 🎯 FastAPIアプリ
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

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
    leg: float         # ← この1行を追加
    weight_age1: float
    measure: int
    daysold: int

# ▶️ 類似度計算に使う重み（10kgに相当する単位変換）
weights = np.array([1.0, 0.667, 2.0, 0.125])  # height, waist, leg, result

# ▶️ 重み付きユークリッド距離関数
def scaled_euclidean(x, y, weights):
    diff = (x - y) * weights
    return np.sqrt(np.sum(diff**2))


# SHAP値と変化量を返すエンドポイント
@app.post("/predict")
async def predict_with_shap(data: InputData, request: Request):
    input_df = pd.DataFrame([data.dict()])
    input_df_model = input_df.drop(columns=["leg"]) 
    shap_values = explainer(input_df_model)

    
    
    # SHAPの出力から値を取り出し
    base_value = shap_values.base_values[0]
    shap_contributions = shap_values.values[0].tolist()
    feature_names = shap_values.feature_names
    gain_pred = model.predict(input_df_model)[0]
    pred_weight = data.weight_age1 + gain_pred

    # ✅ 類似度評価（重み付きユークリッド距離）
    input_vec = np.array([
        data.height,
        data.waist,
        data.leg,
        pred_weight
    ])

    # 📦 参照データ読み込み
    df_all = pd.read_csv("models/WeightSuggestall.csv")
    features = ["height", "waist", "leg", "result"]
    ref_df = df_all.dropna(subset=features).copy()
    X = ref_df[features].values

    # ✅ 距離計算（scaled Euclidean）
    ref_df["euclidean_distance"] = [
        scaled_euclidean(row, input_vec, weights) for row in X
    ]

    # ✅ 重賞馬に限定して類似TOP3を取得
    graded_matches = (
        ref_df[ref_df["graded_winner"] == 1]
        .sort_values("euclidean_distance")
        .head(3)
    )

    top_matches = []
    for _, row in graded_matches.iterrows():
        top_matches.append({
            "name": row.get("name", "不明"),
            "distance": round(row["euclidean_distance"], 3),
            "features": {
                "height": row["height"],
                "waist": row["waist"],
                "leg": row["leg"],
                "result": row["result"]
            }
        })






    return {
        "gain_pred": gain_pred,
        "base_value": base_value,
        "contributions": shap_contributions,
        "features": feature_names,
        "top_matches": top_matches  # 👈 名馬類似度TOP3を追加

    }
