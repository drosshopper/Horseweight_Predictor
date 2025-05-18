# predictor.py
"""
Prediction module for debut weight (デビュー時馬体重) using hybrid of direct & gain models.
Place this file next to app.py and a `models/` folder containing:
  - model_direct.pkl
  - model_gain.pkl
  - correction_direct.pkl
  - correction_gain.pkl
  - imputer.pkl
  - hybrid_alpha.txt (e.g. just "0.73")
"""

from pathlib import Path
import pandas as pd
import joblib

# --------------------------------------------------
