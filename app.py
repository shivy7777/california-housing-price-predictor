import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SF Housing Price Predictor",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD & CLEAN SF DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_sf_data():
    with open("sf_homes_750.json", "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    # Clean column names
    df = df.rename(columns={"bedrooms": "beds", "bathrooms": "baths"})

    # Drop bad rows
    df = df.dropna(subset=["price", "sqft", "beds", "baths", "yearBuilt"])
    df = df[df["sqft"] > 300]
    df = df[df["price"] >= 300000]
    df = df[df["price"] <= 15000000]
    df = df[df["yearBuilt"] > 0]

    # Feature engineering
    df["houseAge"] = 2025 - df["yearBuilt"]
    df["pricePerSqftCalc"] = df["price"] / df["sqft"]
    df["bedsPerBath"] = df["beds"] / (df["baths"] + 0.1)
    df["totalRooms"] = df["beds"] + df["baths"]

    # Encode neighborhood
    le = LabelEncoder()
    df["neighborhoodCode"] = le.fit_transform(df["neighborhood"])

    return df, le


@st.cache_resource
def train_models(df):
    features = [
        "beds", "baths", "sqft", "lotSize", "houseAge",
        "daysOnMarket", "neighborhoodCode", "bedsPerBath", "totalRooms"
    ]

    X = df[features]
    y = df["price"]

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
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
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


df, le = load_sf_data()
models = train_models(df)
neighborhoods = sorted(df["neighborhood"].unique().tolist())

# ─────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────

st.title("🌉 San Francisco Housing Price Predictor")

with st.expander("📖 About This Project", expanded=True):
    st.markdown(f"""
This app predicts San Francisco home prices using real listing data.

Dataset: **{len(df)} listings** across {len(neighborhoods)} neighborhoods.

Models:
- Random Forest
- Linear Regression

Features include sqft, beds, baths, age, lot size, and neighborhood.
""")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.header("🎛️ Property Details")

neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    neighborhoods,
    index=neighborhoods.index("Noe Valley") if "Noe Valley" in neighborhoods else 0
)

beds = st.sidebar.slider("Bedrooms", 1, 8, 3)
baths = st.sidebar.slider("Bathrooms", 1.0, 6.0, 2.0, 0.5)
sqft = st.sidebar.slider("Square Footage", 400, 8000, 1800, 50)
lot_size = st.sidebar.slider("Lot Size", 0, 10000, 2500, 100)
year_built = st.sidebar.slider("Year Built", 1880, 2025, 1930)
dom = st.sidebar.slider("Days on Market", 1, 120, 14)

house_age = 2025 - year_built
beds_per_bath = beds / (baths + 0.1)
total_rooms = beds + baths
neighborhood_code = le.transform([neighborhood])[0] if neighborhood in le.classes_ else 0

user_input = np.array([[
    beds, baths, sqft, lot_size, house_age,
    dom, neighborhood_code, beds_per_bath, total_rooms
]])

rf_pred = models["rf"].predict(user_input)[0]
lr_pred = models["lr"].predict(models["scaler"].transform(user_input))[0]

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

st.header("Price Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric("🌲 Random Forest", f"${rf_pred:,.0f}")

with col2:
    st.metric("📈 Linear Regression", f"${lr_pred:,.0f}")
