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

html, body, [class*="css"], p, li, span, div, label, .stMarkdown, .stText {
    font-family: 'DM Sans', sans-serif;
    color: #ffffff !important;
}

.context-box p, .context-box strong {
    color: #ffffff !important;
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
}

.stSlider > div > div > div {
    background: #f0c96b;
}

section[data-testid="stSidebar"] {
    background-color: #13161f;
}

.badge {
    display: inline-block;
    background: #f0c96b22;
    color: #f0c96b;
    border: 1px solid #f0c96b55;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
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

    df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"].replace(0, 1)
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, 1)
    df["PopulationDensity"] = df["Population"] / df["AveOccup"].replace(0, 1)

    return df, housing.target_names, housing.feature_names


@st.cache_resource
def train_models(df):

    features = [
        "MedInc","HouseAge","AveRooms","AveBedrms",
        "Population","AveOccup","Latitude","Longitude",
        "RoomsPerHousehold","BedroomsPerRoom","PopulationDensity"
    ]

    X = df[features]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s,y_train)

    lr_preds = lr.predict(X_test_s)

    lr_rmse = np.sqrt(mean_squared_error(y_test,lr_preds))
    lr_r2 = r2_score(y_test,lr_preds)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train,y_train)

    rf_preds = rf.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test,rf_preds))
    rf_r2 = r2_score(y_test,rf_preds)

    return {
        "rf":rf,
        "lr":lr,
        "scaler":scaler,
        "features":features,
        "X_test":X_test,
        "y_test":y_test,
        "rf_rmse":rf_rmse,
        "rf_r2":rf_r2,
        "lr_rmse":lr_rmse,
        "lr_r2":lr_r2,
        "rf_preds":rf_preds,
        "lr_preds":lr_preds
    }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df, target_names, feature_names = load_and_engineer()
models = train_models(df)

# ─────────────────────────────────────────────
# PROJECT DESCRIPTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="context-box">

<h3>About This Project</h3>

This app predicts California housing prices using the
<strong>California Housing Dataset</strong>.

Two machine learning models are compared:

• Linear Regression  
• Random Forest

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
st.sidebar.markdown("## Input Features")

med_inc = st.sidebar.slider("Median Income",0.5,15.0,5.0)
house_age = st.sidebar.slider("House Age",1,52,20)
ave_rooms = st.sidebar.slider("Avg Rooms",1.0,10.0,5.0)
ave_bedrms = st.sidebar.slider("Avg Bedrooms",0.5,5.0,1.0)
population = st.sidebar.slider("Population",3,5000,1000)
ave_occup = st.sidebar.slider("Avg Occupants",1.0,6.0,2.5)
latitude = st.sidebar.slider("Latitude",32.5,42.0,37.0)
longitude = st.sidebar.slider("Longitude",-124.5,-114.0,-120.0)

rooms_per_hh = ave_rooms/max(house_age,1)
bedrms_per_room = ave_bedrms/max(ave_rooms,1)
pop_density = population/max(ave_occup,1)

user_input = np.array([[
    med_inc,house_age,ave_rooms,ave_bedrms,
    population,ave_occup,latitude,longitude,
    rooms_per_hh,bedrms_per_room,pop_density
]])

rf_prediction = models["rf"].predict(user_input)[0]*100000

lr_scaled = models["scaler"].transform(user_input)
lr_prediction = models["lr"].predict(lr_scaled)[0]*100000

# confidence interval
tree_preds = np.array([tree.predict(user_input)[0] for tree in models["rf"].estimators_])
confidence_low = np.percentile(tree_preds,10)*100000
confidence_high = np.percentile(tree_preds,90)*100000

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3 = st.tabs(["Prediction","EDA","Model Comparison"])

# ─────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────
with tab1:

    st.markdown("## Price Prediction")

    col1,col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="prediction-box">
        <p>Random Forest Estimate</p>
        <h1>${rf_prediction:,.0f}</h1>
        <p>Range: ${confidence_low:,.0f} – ${confidence_high:,.0f}</p>
        </div>
        """,unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-box">
        <p>Linear Regression Estimate</p>
        <h1>${lr_prediction:,.0f}</h1>
        </div>
        """,unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────
with tab2:

    st.markdown("## Exploratory Data Analysis")

    fig = px.histogram(
        df,
        x="MedHouseVal",
        nbins=60,
        color_discrete_sequence=["#f0c96b"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────
with tab3:

    st.markdown("## Model Comparison")

    sample_idx = np.random.choice(len(models["y_test"]),300,replace=False)

    y_sample = models["y_test"].values[sample_idx]*100000
    rf_sample = models["rf_preds"][sample_idx]*100000
    lr_sample = models["lr_preds"][sample_idx]*100000

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_sample,
        y=rf_sample,
        mode="markers",
        name="Random Forest"
    ))

    fig.add_trace(go.Scatter(
        x=y_sample,
        y=lr_sample,
        mode="markers",
        name="Linear Regression"
    ))

    st.plotly_chart(fig,use_container_width=True)
