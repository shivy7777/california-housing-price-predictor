import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0f1117;
        color: #e8e8e8;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
        color: #f0c96b;
    }

    .context-box {
        background: linear-gradient(135deg, #1a1f2e, #1f2a3a);
        border-left: 4px solid #f0c96b;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2a2f3e;
    }

    .metric-card h2 {
        font-size: 2rem;
        margin: 0;
        color: #f0c96b;
    }

    .metric-card p {
        margin: 0;
        color: #aaa;
        font-size: 0.85rem;
    }

    .prediction-box {
        background: linear-gradient(135deg, #1f3a2a, #1a2f1f);
        border: 2px solid #4caf7d;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .prediction-box h1 {
        font-size: 3rem;
        color: #4caf7d !important;
        margin: 0;
    }

    .prediction-box p {
        color: #aaa;
        font-size: 0.9rem;
    }

    .stSlider > div > div > div {
        background: #f0c96b;
    }

    section[data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #2a2f3e;
    }

    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        font-family: 'DM Sans', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        color: #f0c96b !important;
        border-bottom-color: #f0c96b !important;
    }

    .badge {
        display: inline-block;
        background: #f0c96b22;
        color: #f0c96b;
        border: 1px solid #f0c96b55;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_engineer():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    # Feature engineering — 2 new meaningful features
    df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, 1)
    df["PopulationDensity"] = df["Population"] / df["AveOccup"].replace(0, 1)
    return df, housing.target_names, housing.feature_names

@st.cache_resource
def train_models(df):
    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
        "RoomsPerHousehold", "BedroomsPerRoom", "PopulationDensity"
    ]
    X = df[features]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    return {
        "rf": rf, "lr": lr, "scaler": scaler,
        "features": features, "X_test": X_test, "y_test": y_test,
        "rf_rmse": rf_rmse, "rf_r2": rf_r2,
        "lr_rmse": lr_rmse, "lr_r2": lr_r2,
        "rf_preds": rf_preds, "lr_preds": lr_preds
    }

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df, target_names, feature_names = load_and_engineer()
models = train_models(df)

# ─────────────────────────────────────────────
# CONTEXT / ABOUT SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="context-box">
    <h3>🏠 About This Project</h3>
    <p>
        This app predicts California housing prices using the classic <strong>California Housing Dataset</strong>
        from 
