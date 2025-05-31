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

    # ✅ 名馬との類似度評価（マハラノビス距離）
    # ✅ 入力ベクトル
    input_vec = np.array([
        data.height,
        data.waist,
        data.leg,
        pred_weight
    ])

    # ✅ データ読み込み
    df_all = pd.read_csv("models/WeightSuggestall.csv")
    features = ["height", "waist", "leg", "result"]
    ref_df = df_all.dropna(subset=features).copy()  # 全馬対象に変更
    X = ref_df[features].values
    
    # ✅ 重みの定義（自由に調整可能）
    weights = np.array([20, 20, 1, 50])
    
    # ✅ 共分散行列は全体で計算
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    
    # ✅ 重み付きマハラノビス距離関数
    def weighted_mahalanobis(x, y, inv_cov, weights):
        diff = (x - y) * weights
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    
    # ✅ 距離を全馬に対して計算
    ref_df["mahalanobis"] = [
        weighted_mahalanobis(row, input_vec, inv_cov, weights) for row in X
    ]
    
    # ✅ 重賞勝ち馬だけから最も近い馬を抽出（TOP3）
    graded_matches = (
        ref_df[ref_df["graded_winner"] == 1]
        .sort_values("mahalanobis")
        .head(3)
    )
    
    top_matches = []
    for _, row in graded_matches.iterrows():
        top_matches.append({
            "name": row.get("name", "不明"),
            "distance": round(row["mahalanobis"], 3),
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
