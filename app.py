import streamlit as st
import pandas as pd
import numpy as np
import joblib
import calendar
import cloudpickle as cp
from datetime import datetime

st.markdown("""
<style>
body, .stTextInput, .stNumberInput, .stSelectbox {
    font-family: 'Segoe UI', sans-serif;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stExpander"] div[role="button"] {
    border: 3px solid #003f8c;
    border-radius: 6px;
    padding: 6px;
    background-color: #f8faff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🐎 デビュー時馬体重予測システム")
st.markdown("---")


def load_models():
    model_dir = 'models'
    with open(f"{model_dir}/model_direct.pkl", "rb") as f:
        md = cp.load(f)
    with open(f"{model_dir}/correction_direct.pkl", "rb") as f:
        cd = cp.load(f)
    with open(f"{model_dir}/correction_gain.pkl", "rb") as f:  # correction_growth → correction_gain に修正
        cg = cp.load(f)
    with open(f"{model_dir}/imputer_cloud.pkl", "rb") as f:  # imputer.pkl → imputer_cloud.pkl に修正
        imp = cp.load(f)
    return md, cd, cg, imp

def adjust_invalid_date(year, month, day):
    dim = calendar.monthrange(year, month)[1]
    if day > dim:
        day = dim
        month += 1
        if month > 12:
            month = 1
    return datetime(year, month, day)

with st.expander("基本情報", expanded=True):
    col1, col2 = st.columns([2,1])
    with col1:
        weight_age1 = st.number_input("測尺時体重 (kg)", min_value=300, max_value=600, value=430, step=1, format="%d")
    with col2:
        sex = st.selectbox("性別", ["牡", "牝"])
    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.number_input("体高 (cm)", min_value=130.0, max_value=200.0, value=150.0, step=1.0, format="%.1f")
    with c2:
        waist_input = st.text_input("胸囲 (cm)（任意）", value="")
        waist = float(waist_input) if waist_input.strip() != "" else np.nan
    with c3:
        leg_input = st.text_input("管囲 (cm)（任意）", value="")
        leg = float(leg_input) if leg_input.strip() != "" else np.nan

with st.expander("日付情報", expanded=True):
    lbl1, lbl2 = st.columns(2)
    with lbl1:
        st.markdown("**生まれた月日**")
    with lbl2:
        st.markdown("**測尺公開月日**")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        birth_month = st.selectbox("生まれた月", list(range(1,13)), key="bm")
    with d2:
        birth_day   = st.selectbox("生まれた日", list(range(1,32)), key="bd")
    with d3:
        measure_month = st.selectbox("測尺公開月", list(range(1,13)), index=5, key="mm")
    with d4:
        measure_day   = st.selectbox("測尺公開日", list(range(1,32)), index=0, key="md")

st.markdown("---")

now = datetime.now()
bd = adjust_invalid_date(now.year-1, birth_month, birth_day)
md = adjust_invalid_date(now.year,   measure_month, measure_day)
daysold = (md - bd).days
ref = datetime(bd.year+2, 6, 1).date()
measure_days = (ref - md.date()).days

if st.button("予測実行"):
    model_direct, correction_direct, correction_growth, imputer = load_models()

    df = pd.DataFrame({
        'sex':         [1 if sex == "牡" else 2],
        'daysold':     [daysold],
        'height':      [height],
        'waist':       [waist],
        'leg':         [leg],
        'weight_age1': [weight_age1],
        'measure':     [measure_days]
    })

    df_filled = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # 補完された値がある列にマークを付ける
    warning_cols = []
    for col in ['waist', 'leg']:
        if pd.isna(df[col].values[0]):
            warning_cols.append(col)

    pred_direct = model_direct.predict(df_filled)
    residual_linear = correction_direct.predict(df_filled)
    corrected_pred = pred_direct + residual_linear

    df_corr = pd.DataFrame({
        'predicted': corrected_pred,
        'weight_age1': df_filled['weight_age1']
    })
    residual_nonlinear = correction_growth.predict(df_corr)
    final_pred = corrected_pred + residual_nonlinear
    pred_rounded = int(round(final_pred[0]))

    ci_90 = 23
    ci_80 = 19
    ci_50 = 11

    lower_90, upper_90 = int(round(pred_rounded + ci_90[0])), int(round(pred_rounded + ci_90[1]))
    lower_80, upper_80 = int(round(pred_rounded + ci_80[0])), int(round(pred_rounded + ci_80[1]))
    lower_50, upper_50 = int(round(pred_rounded + ci_50[0])), int(round(pred_rounded + ci_50[1]))

    st.markdown(f"<div style='text-align: center; margin-top: 1rem; margin-bottom: 1rem;'>"
                f"<p style='font-size: 1.2rem; font-weight: bold; color: #003f8c; margin-bottom: 0.5rem;'>🦄 予測デビュー体重</p>"
                f"<p style='font-size: 2.5rem; font-weight: bold; color: black;'>{pred_rounded} kg</p>"
                f"<p style='font-size: 1rem; color: gray;'>90%信頼区間: {lower_90} kg ～ {upper_90} kg</p>"
                f"<p style='font-size: 0.9rem; color: gray;'>80%信頼区間: {lower_80} kg ～ {upper_80} kg</p>"
                f"<p style='font-size: 0.8rem; color: gray;'>50%信頼区間: {lower_50} kg ～ {upper_50} kg</p>"
                f"</div>", unsafe_allow_html=True)

    if warning_cols:
        st.markdown(f'<p style="font-size: 0.8rem; color: gray;">以下の項目は平均値で補完されています: {', '.join(warning_cols)}</p>', unsafe_allow_html=True)
