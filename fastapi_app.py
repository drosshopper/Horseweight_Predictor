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

# 🌐 CORS（Hugging Face 2スペースからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://huggingface.co/spaces/drosshopper/horse-weight-predictor2",
        "https://huggingface.co/spaces/drosshopper/horse-weight-predictor-test"
    ],
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

# ▶️ マハラノビス距離関数（今回使用）
def mahalanobis_dist(x, y, VI):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff.T, VI), diff))

# 🧠 管囲判定関数（比率に基づく）
def judge_leg_ratio_extended(ratio: float) -> str:
    if ratio <= 0.928:
        return "極めて細い"
    elif ratio <= 0.949:
        return "非常に細い"
    elif ratio <= 0.968:
        return "細い"
    elif ratio <= 0.979:
        return "やや細い"
    elif ratio <= 1.020:
        return "標準"
    elif ratio <= 1.032:
        return "やや太い"
    elif ratio <= 1.049:
        return "太い"
    elif ratio <= 1.069:
        return "非常に太い"
    else:
        return "極めて太い"

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


    # 🦵 標準管囲の計算・診断
    leg_pred = 6.1529196 + 0.0118197 * pred_weight + 0.0326342 * data.height + 0.0191873 * data.waist
    leg_ratio = data.leg / leg_pred
    leg_judge = judge_leg_ratio_extended(leg_ratio)

    
    # ✅ 類似度評価（マハラノビス距離 + 指数スコア）
    input_vec = np.array([
        data.height,
        data.waist,
        pred_weight  # ← 推論後の予測体重
    ])
    
    # 📦 参照データ読み込み
    df_all = pd.read_csv("models/WeightSuggestall.csv")
    features = ["height", "waist", "result"]
    ref_df = df_all.dropna(subset=features).copy()
    
    # ✅ 共分散行列は全馬で構築
    X = ref_df[features].values
    cov_matrix = np.cov(X.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # ✅ 入力ベクトル（予測体重）
    input_vec = np.array([data.height, data.waist, pred_weight])
    
    # ✅ 全馬に距離・スコアを付与
    ref_df["distance"] = [mahalanobis_dist(input_vec, row, inv_cov_matrix) for row in X]
    beta = 0.4
    ref_df["score"] = 100 * np.exp(-beta * ref_df["distance"])
    
    # ✅ 重賞馬のみ抽出して上位3頭を返す
    graded_df = ref_df[ref_df["graded_winner"] == 1].copy()
    
    top_matches = []
    for _, row in graded_df.sort_values("score", ascending=False).head(3).iterrows():
        top_matches.append({
            "name": row.get("name", "不明"),
            "distance": round(row["distance"], 3),
            "score": round(row["score"], 1),
            "features": {
                "height": row["height"],
                "waist": row["waist"],
                "result": row["result"]
            },
            "graded_titles": [
                win for win in [row.get("win1"), row.get("win2"), row.get("win3")] if pd.notna(win)
            ]
        })








    return {
        "gain_pred": gain_pred,
        "base_value": base_value,
        "contributions": shap_contributions,
        "features": feature_names,
        "top_matches": top_matches,
        "leg_pred": leg_pred,
        "leg_ratio": leg_ratio,
        "leg_judge": leg_judge
    }
