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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

.stApp, .stApp > div {
background-color: #0f1117 !important;
}

.stMarkdown p, .stMarkdown li, .stMarkdown span,
.stText, label, .stSelectbox label {
color: #ffffff !important;
font-family: 'DM Sans', sans-serif !important;
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
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_and_engineer():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, 1)
    df["PopulationDensity"] = df["Population"] / df["AveOccup"].replace(0, 1)

    return df


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

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    lr_preds = lr.predict(X_test_s)

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    return {
        "rf": rf,
        "lr": lr,
        "scaler": scaler,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "rf_preds": rf_preds,
        "lr_preds": lr_preds
    }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df = load_and_engineer()
models = train_models(df)

# ─────────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────────

st.sidebar.header("Input Features")

med_inc = st.sidebar.slider("Median Income", 0.5, 15.0, 5.0)
house_age = st.sidebar.slider("House Age", 1, 52, 20)
ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.sidebar.slider("Population", 3, 5000, 1000)
ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 2.5)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 37.0)
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -120.0)

rooms_per_hh = ave_rooms / max(house_age, 1)
bedrms_per_room = ave_bedrms / max(ave_rooms, 1)
pop_density = population / max(ave_occup, 1)

user_input = np.array([[

    med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude,
    rooms_per_hh, bedrms_per_room, pop_density

]])

rf_prediction = models["rf"].predict(user_input)[0] * 100000
lr_prediction = models["lr"].predict(
    models["scaler"].transform(user_input)
)[0] * 100000

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs(["Prediction", "Exploration"])

# ─────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────

with tab1:

    st.title("California Housing Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Random Forest Prediction", f"${rf_prediction:,.0f}")

    with col2:
        st.metric("Linear Regression Prediction", f"${lr_prediction:,.0f}")

# ─────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────

with tab2:

    st.title("Exploratory Data Analysis")

    fig = px.histogram(
        df,
        x="MedHouseVal",
        nbins=50,
        color_discrete_sequence=["#f0c96b"]
    )

    st.plotly_chart(fig, use_container_width=True)

    corr = df.corr()

    heat = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis"
        )
    )

    st.plotly_chart(heat, use_container_width=True)
