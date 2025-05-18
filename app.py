
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import calendar
from datetime import datetime

# --- モデル読み込み関数 ---
def load_models():
    model_dir = "models"
    with open(f"{model_dir}/model_direct_cloud.pkl", "rb") as f:
        md = cloudpickle.load(f)
    with open(f"{model_dir}/model_gain_cloud.pkl", "rb") as f:
        mg = cloudpickle.load(f)
    with open(f"{model_dir}/correction_direct_cloud.pkl", "rb") as f:
        cd = cloudpickle.load(f)
    with open(f"{model_dir}/correction_gain_cloud.pkl", "rb") as f:
        cg = cloudpickle.load(f)
    with open(f"{model_dir}/imputer_cloud.pkl", "rb") as f:
        imp = cloudpickle.load(f)
    with open(f"{model_dir}/hybrid_alpha.txt", "r") as f:
        alpha = float(f.read())
    return md, mg, cd, cg, imp, alpha

# --- 無効日修正 ---
def adjust_invalid_date(year, month, day):
    dim = calendar.monthrange(year, month)[1]
    if day > dim:
        day = dim
        month += 1
        if month > 12:
            month = 1
    return datetime(year, month, day)

# --- UI ---
st.title("🐴 デビュー時馬体重予測システム")

with st.expander("基本情報", expanded=True):
    col1, col2 = st.columns([2,1])
    with col1:
        weight_age1 = st.number_input("測尺時体重 (kg)", min_value=300, max_value=600, value=430, step=1)
    with col2:
        sex = st.selectbox("性別", ["牡", "牝"])
    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.number_input("体高 (cm)", 130.0, 200.0, 150.0, step=1.0)
    with c2:
        waist_input = st.text_input("胸囲 (cm)（任意）", "")
        waist = float(waist_input) if waist_input.strip() else np.nan
    with c3:
        leg_input = st.text_input("管囲 (cm)（任意）", "")
        leg = float(leg_input) if leg_input.strip() else np.nan

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
        birth_day = st.selectbox("生まれた日", list(range(1,32)), key="bd")
    with d3:
        measure_month = st.selectbox("測尺月", list(range(1,13)), index=5, key="mm")
    with d4:
        measure_day = st.selectbox("測尺日", list(range(1,32)), index=0, key="md")

st.markdown("---")

now = datetime.now()
bd = adjust_invalid_date(now.year-1, birth_month, birth_day)
md = adjust_invalid_date(now.year, measure_month, measure_day)
daysold = (md - bd).days
ref = datetime(bd.year+2, 6, 1).date()
measure_days = (ref - md.date()).days

if st.button("予測実行"):
    model_direct, model_gain, correction_direct, correction_gain, imputer, alpha = load_models()

    df = pd.DataFrame({
        'sex': [1 if sex == "牡" else 2],
        'daysold': [daysold],
        'height': [height],
        'waist': [waist],
        'leg': [leg],
        'weight_age1': [weight_age1],
        'measure': [measure_days]
    })

    df_filled = pd.DataFrame(imputer.transform(df), columns=df.columns)
    pred_direct = model_direct.predict(df_filled) + correction_direct.predict(df_filled)
    pred_gain = model_gain.predict(df_filled) + df_filled['weight_age1'] + correction_gain.predict(df_filled)
    pred_hybrid = alpha * pred_gain + (1 - alpha) * pred_direct
    pred_rounded = int(round(pred_hybrid[0]))

    st.markdown(f"### 🦄 予測デビュー体重：**{pred_rounded} kg**")

    ci50_lower = pred_rounded - 15
    ci75_lower = pred_rounded - 27
    ci90_lower = pred_rounded - 35
    ci50_upper = pred_rounded + 15
    ci75_upper = pred_rounded + 27
    ci90_upper = pred_rounded + 35

    st.markdown(f"""
<ul>
<li>{ci50_lower}kg ～ {ci50_upper}kg（±15kg）</li>
<li>{ci75_lower}kg ～ {ci75_upper}kg（±27kg）</li>
<li>{ci90_lower}kg ～ {ci90_upper}kg（±35kg）</li>
</ul>
""", unsafe_allow_html=True)

    if np.isnan(waist) or np.isnan(leg):
        st.markdown('<p style="font-size: 0.8rem; color: gray;">未入力の項目は平均値で補完されています。</p>', unsafe_allow_html=True)

    # 標準値計算
    x = df_filled['weight_age1'].iloc[0]
    std_height = 0.6435 * x - 0.001235 * x**2 + 0.000000873 * x**3 + 33.322
    std_waist = 1.1700 * x - 0.002394 * x**2 + 0.00000172 * x**3 - 27.097
    std_leg = -0.4447 * x + 0.000958 * x**2 - 0.000000657 * x**3 + 85.696

    height_ratio = height / std_height * 100
    waist_ratio = waist / std_waist * 100 if not np.isnan(waist) else np.nan
    leg_ratio = leg / std_leg * 100 if not np.isnan(leg) else np.nan

    def classify_with_percent(r, thresholds, labels):
        if r >= thresholds["上位10%"]: return f"非常に{labels['上']}（上位10％）"
        elif r >= thresholds["上位30%"]: return f"{labels['上']}（上位30％）"
        elif r <= thresholds["下位10%"]: return f"非常に{labels['下']}（下位10％）"
        elif r <= thresholds["下位30%"]: return f"{labels['下']}（下位30％）"
        else: return "標準"

    height_eval = classify_with_percent(height_ratio, {
        "上位10%": 102.50, "上位30%": 101.02, "下位30%": 98.96, "下位10%": 97.53
    }, {"上": "高い", "下": "低い"})

    waist_eval = classify_with_percent(waist_ratio, {
        "上位10%": 103.16, "上位30%": 101.30, "下位30%": 98.70, "下位10%": 96.65
    }, {"上": "太い", "下": "細い"}) if not np.isnan(waist_ratio) else "不明"

    leg_eval = classify_with_percent(leg_ratio, {
        "上位10%": 103.64, "上位30%": 101.57, "下位30%": 98.64, "下位10%": 96.28
    }, {"上": "太い", "下": "細い"}) if not np.isnan(leg_ratio) else "不明"

    # 表形式出力
    result_table = pd.DataFrame({
        "項目": ["体高（cm）", "胸囲（cm）", "管囲（cm）"],
        "実測値": [height, waist, leg],
        "標準値": [std_height, std_waist, std_leg],
        "差": [
            height - std_height,
            waist - std_waist if not np.isnan(waist) else np.nan,
            leg - std_leg if not np.isnan(leg) else np.nan
        ],
        "比率（%）": [height_ratio, waist_ratio, leg_ratio],
        "評価": [height_eval, waist_eval, leg_eval]
    })

    result_table["差"] = result_table["差"].apply(lambda x: f"+{x:.1f} cm" if x > 0 else f"{x:.1f} cm" if pd.notna(x) else "―")
    result_table["実測値"] = result_table["実測値"].apply(lambda x: f"{x:.1f} cm" if pd.notna(x) else "―")
    result_table["標準値"] = result_table["標準値"].apply(lambda x: f"{x:.1f} cm" if pd.notna(x) else "―")
    result_table["比率（%）"] = result_table["比率（%）"].apply(lambda x: f"{x:.1f} %" if pd.notna(x) else "―")

    st.markdown("### 🏇 馬体評価")
    st.table(result_table)
