import streamlit as st
import pandas as pd
import numpy as np
import joblib
import calendar
from datetime import datetime

st.markdown("""
<style>
body, .stTextInput, .stNumberInput, .stSelectbox {
    font-family: 'Segoe UI', sans-serif;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# --- カスタム CSS（Expander の青枠＆metric 強調） ---
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

# --- ヘッダー ---
st.markdown("## 🐴 デビュー時馬体重予測システム")
st.markdown("---")

# --- モデル読み込み ---
def load_models():
    model_dir = 'models'
    md  = joblib.load(f"{model_dir}/model_direct.pkl")
    mg  = joblib.load(f"{model_dir}/model_gain.pkl")
    cd  = joblib.load(f"{model_dir}/correction_direct.pkl")
    cg  = joblib.load(f"{model_dir}/correction_gain.pkl")
    imp = joblib.load(f"{model_dir}/imputer.pkl")
    with open(f"{model_dir}/hybrid_alpha.txt", "r") as f:
        alpha = float(f.read())
    return md, mg, cd, cg, imp, alpha

def adjust_invalid_date(year, month, day):
    dim = calendar.monthrange(year, month)[1]
    if day > dim:
        day = dim
        month += 1
        if month > 12:
            month = 1
    return datetime(year, month, day)

# --- 入力セクション：基本情報 ---
with st.expander("基本情報", expanded=True):
    col1, col2 = st.columns([2,1])
    with col1:
        weight_age1 = st.number_input(
            "測尺時体重 (kg)", 
            min_value=300, max_value=600, 
            value=430, step=1, format="%d"
        )
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

# --- 入力セクション：日付情報 ---
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

# --- 日数・予測処理 ---
now = datetime.now()
bd = adjust_invalid_date(now.year-1, birth_month, birth_day)
md = adjust_invalid_date(now.year,   measure_month, measure_day)
daysold      = (md - bd).days
ref          = datetime(bd.year+2, 6, 1).date()
measure_days = (ref - md.date()).days

if st.button("予測実行"):
    model_direct, model_gain, correction_direct, correction_gain, imputer, alpha = load_models()

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

    pred_direct = model_direct.predict(df_filled) + correction_direct.predict(df_filled)
    pred_gain = model_gain.predict(df_filled) + df_filled['weight_age1'] + correction_gain.predict(df_filled)
    pred_hybrid = alpha * pred_gain + (1 - alpha) * pred_direct
    pred_rounded = int(round(pred_hybrid[0]))

    # 信頼区間（固定幅）
    ci50_lower = pred_rounded - 15
    ci75_lower = pred_rounded - 27
    ci90_lower = pred_rounded - 35
    ci50_upper = pred_rounded + 15
    ci75_upper = pred_rounded + 27
    ci90_upper = pred_rounded + 35

    # 表示
    # 🔽 予測体重のカスタム表示（大きめフォント）
    st.markdown(f"""
<div style="text-align: center; margin-top: 1rem; margin-bottom: 1rem;">
    <p style="font-size: 1.2rem; font-weight: bold; color: #003f8c; margin-bottom: 0.5rem;">🦄 予測デビュー体重</p>
    <p style="font-size: 2.5rem; font-weight: bold; color: black;">{pred_rounded} kg</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
ul.compact-list li {{
    margin-bottom: 0.1rem;
}}
</style>
<ul class="compact-list">
<br>最小馬体重</br>
<li>{0} kg以上になる確率75％未満</li>
<li>{1} kg以上になる確率90％未満</li>
<li>{2} kg以上になる確率95％未満</li>
<br>最大馬体重</br>
<li>{3} kg以下になる確率75％未満</li>
<li>{4} kg以下になる確率90％未満</li>
<li>{5} kg以下になる確率95％未満</li>
</ul>
""".format(ci50_lower, ci75_lower, ci90_lower, ci50_upper, ci75_upper, ci90_upper), unsafe_allow_html=True)

    # 🔽 未入力時の補完メッセージ（小さな文字で）
    if np.isnan(waist) or np.isnan(leg):
        st.markdown(
            '<p style="font-size: 0.8rem; color: gray;">未入力の項目は過去のデータから算定した平均値と仮定して計算しました。</p>',
            unsafe_allow_html=True
        )

# 追加コードは既存の "if st.button("予測実行"):" の末尾に追記してください

    # --- 標準値の算出（3次式） ---
    x = df_filled['weight_age1'].iloc[0]
    std_height = 0.6435 * x - 0.001235 * x**2 + 0.000000873 * x**3 + 33.322
    std_waist  = 1.1700 * x - 0.002394 * x**2 + 0.00000172 * x**3 - 27.097
    std_leg    = -0.4447 * x + 0.000958 * x**2 - 0.000000657 * x**3 + 85.696

    # --- 比率（％） ---
    height_ratio = height / std_height * 100
    waist_ratio = waist / std_waist * 100 if not np.isnan(waist) else np.nan
    leg_ratio   = leg / std_leg * 100 if not np.isnan(leg) else np.nan

    # --- 評価関数（非常に〜付き） ---
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

    # --- 表形式出力 ---
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

    # 実測値・標準値：小数第1位＋cm
    for col in ["実測値", "標準値"]:
        result_table[col] = result_table[col].apply(lambda x: f"{x:.1f} cm" if pd.notna(x) else "―")

    # 差：小数第1位＋cm、かつ正の数値には + を明示
    def format_diff(x):
        if pd.isna(x):
            return "―"
        return f"+{x:.1f} cm" if x > 0 else f"{x:.1f} cm"
    result_table["差"] = result_table["差"].apply(format_diff)

    # 比率：%表記
    result_table["比率（%）"] = result_table["比率（%）"].apply(lambda x: f"{x:.1f} %" if pd.notna(x) else "―")


    # 表形式出力 ---（整形済み）
    st.markdown("### 🏇 馬体診断")

    # インデックスを消す（ここで実行！）
    result_table = result_table.reset_index(drop=True)

    # テーブル表示
    st.table(result_table)

    st.markdown(
        '<p style="font-size: 0.8rem; color: gray;">'
        '※同程度の予測馬体重の馬同士で比べたときの評価であり、全体の中でのサイズを評価するものではありません'
        '</p>',
        unsafe_allow_html=True
    )

    # --- 象限に応じたコメント出力 ---
    def merge_class(h_ratio, w_ratio):
        if h_ratio >= 101.02: h_zone = "高体高"
        elif h_ratio <= 98.96: h_zone = "低体高"
        else: h_zone = "標準体高"

        if w_ratio >= 101.30: w_zone = "太胸囲"
        elif w_ratio <= 98.70: w_zone = "細胸囲"
        else: w_zone = "標準胸囲"

        return f"{h_zone} × {w_zone}"

    zone_label = merge_class(height_ratio, waist_ratio)

    zone_comments = {
        "低体高 × 細胸囲": "代表馬：シャフリヤール、シュネルマイスター。適性：芝ダ比率は標準だが、短距離傾向がややある。",
        "低体高 × 標準胸囲": "代表馬：ラウダシオン、オーソリティ、アスコリピチェーノ。適性：芝ダは標準だが、短距離傾向が非常に強い。",
        "低体高 × 太胸囲": "代表馬はサリオス、ピースオブエイト。全体比率は4％(平均11％)とかなり珍しいタイプ。芝ダの比率は標準だが、短距離の傾向がやや強め。",
        "標準体高 × 細胸囲": "代表馬はレガレイラ、ジラルディーナ。芝ダ比率は標準だが、長距離傾向がややある。",
        "標準体高 × 標準胸囲": "代表馬はリバティアイランド、ソングライン、ドゥレッツァ。芝ダ比率は標準だが、やや長距離傾向がある。",
        "標準体高 × 太胸囲": "代表馬はレシステンシア、サートゥルナーリア。芝ダ比率は標準だが、短距離傾向がやや強め。",
        "高体高 × 細胸囲": "代表馬はクイーンズウォーク。全体比率は3％(平均11％)とかなり珍しいタイプ。芝の長距離傾向が非常に強い。",
        "高体高 × 標準胸囲": "代表馬はタスティエーラ、エフフォーリア、レイパパレ。芝ダ比率は標準だが、長距離傾向が強い。",
        "高体高 × 太胸囲": "代表馬はイクイノックス、ナミュール、ヴェラアズール。芝ダ比率も距離傾向も標準。"
    }
    st.markdown(f"#### 🧭 {zone_label}")
    st.markdown(f"<p style='font-size: 0.95rem'>{zone_comments.get(zone_label, '該当するコメントはありません。')}</p>", unsafe_allow_html=True)


st.markdown("""
---
<p style="text-align: center; font-size: 0.8rem; color: gray;">
© 2025 drosshopper AI. All rights reserved.
</p>
""", unsafe_allow_html=True)
