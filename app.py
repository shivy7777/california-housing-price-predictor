import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
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

sns.set_style("whitegrid")

# ─────────────────────────────────────────────
# LOAD & CLEAN SF DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_sf_data():
    with open("sfhousedataset.json", "r") as f:
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
This app predicts SF home prices using real listing data.

Dataset: **{len(df)} listings** across {len(neighborhoods)} neighborhoods.

Models:
- Random Forest
- Linear Regression
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
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction", "🏘️ Neighborhood Analysis", "📊 EDA", "🤖 Model Comparison"
])

# TAB 1
with tab1:
    st.header("Price Prediction")
    st.caption(f"{beds}bd/{baths}ba • {sqft:,} sqft • {neighborhood}")

    col1, col2 = st.columns(2)
    col1.metric("🌲 Random Forest", f"${rf_pred:,.0f}")
    col2.metric("📈 Linear Regression", f"${lr_pred:,.0f}")

# TAB 2
with tab2:
    st.header("Neighborhood Analysis")

    hood_stats = df.groupby("neighborhood").agg(
        median_price=("price", "median"),
        listings=("price", "count")
    ).reset_index().sort_values("median_price", ascending=False)

    top20 = hood_stats.head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top20,
        x="neighborhood",
        y="median_price",
        hue="median_price",
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Median Price ($)")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    st.pyplot(fig)

# TAB 3
with tab3:
    st.header("EDA")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["price"], bins=50, ax=ax)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)

# TAB 4
with tab4:
    st.header("Model Comparison")

    idx = np.random.choice(len(models["y_test"]), min(200, len(models["y_test"])), replace=False)

    y = models["y_test"].values[idx]
    rf = models["rf_preds"][idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=y, y=rf, ax=ax)
    lims = [min(y.min(), rf.min()), max(y.max(), rf.max())]
    ax.plot(lims, lims, linestyle="--", color="gray")  # reference line: perfect predictions
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    fig.tight_layout()
    st.pyplot(fig)
