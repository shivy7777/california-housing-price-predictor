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

st.set_page_config(
    page_title="CA Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


df = load_and_engineer()
models = train_models(df)

st.title("🏠 California Housing Price Predictor")

with st.expander("📖 About This Project", expanded=True):
    st.markdown("""
This app predicts California housing prices using the **California Housing Dataset** from the 1990 census.  
Two machine learning models, **Random Forest** and **Linear Regression**, are trained and compared side-by-side.  
Three new features are engineered beyond the raw dataset to improve predictive power.

**Why this matters:** Understanding what drives property values helps buyers make smarter decisions,
helps cities identify undervalued neighborhoods, and is a foundational application of regression modeling in data science.

**Tech stack:** `scikit-learn` · `Plotly` · `Pandas` · `NumPy` · `Streamlit`
""")


# Sidebar Inputs
st.sidebar.header("🎛️ Input Features")
st.sidebar.caption("Adjust values to generate a price prediction.")

med_inc = st.sidebar.slider("Median Income (tens of thousands)", 0.5, 15.0, 5.0, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 10.0, 5.0, 0.1)
ave_bedrms = st.sidebar.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.0, 0.1)
population = st.sidebar.slider("Block Population", 3, 5000, 1000)
ave_occup = st.sidebar.slider("Avg Occupants per Household", 1.0, 6.0, 2.5, 0.1)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 37.0, 0.1)
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -120.0, 0.1)

# Feature Engineering for input
rooms_per_hh = ave_rooms / max(house_age, 1)
bedrms_per_room = ave_bedrms / max(ave_rooms, 1)
pop_density = population / max(ave_occup, 1)

user_input = np.array([[
    med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude,
    rooms_per_hh, bedrms_per_room, pop_density
]])

# Predictions
rf_prediction = models["rf"].predict(user_input)[0] * 100000
lr_input_scaled = models["scaler"].transform(user_input)
lr_prediction = models["lr"].predict(lr_input_scaled)[0] * 100000

tree_preds = np.array([t.predict(user_input)[0] for t in models["rf"].estimators_])
confidence_low = np.percentile(tree_preds, 10) * 100000
confidence_high = np.percentile(tree_preds, 90) * 100000

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Exploratory Analysis", "🤖 Model Comparison"])

# TAB 1
with tab1:
    st.header("Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🌲 Random Forest Estimate",
            value=f"${rf_prediction:,.0f}",
            delta=f"90% range: ${confidence_low:,.0f} – ${confidence_high:,.0f}"
        )

    with col2:
        st.metric(
            label="📈 Linear Regression Estimate",
            value=f"${lr_prediction:,.0f}"
        )

    st.divider()
    st.subheader("Model Performance on Test Set")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random Forest R²", f"{models['rf_r2']:.3f}")
    c2.metric("Random Forest RMSE", f"${models['rf_rmse']*100000:,.0f}")
    c3.metric("Linear Regression R²", f"{models['lr_r2']:.3f}")
    c4.metric("Linear Regression RMSE", f"${models['lr_rmse']*100000:,.0f}")

    st.divider()
    st.subheader("🔍 What Drives the Price?")

    importances = pd.Series(
        models["rf"].feature_importances_,
        index=models["features"]
    ).sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        marker=dict(color=importances.values, colorscale="Viridis")
    ))

    fig.update_layout(
        xaxis_title="Importance Score",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


# TAB 2
with tab2:
    st.header("Exploratory Data Analysis")

    st.subheader("Price Distribution")
    fig1 = px.histogram(
        df,
        x="MedHouseVal",
        nbins=60,
        labels={"MedHouseVal": "Median House Value ($100k)"},
        color_discrete_sequence=["#636EFA"]
    )

    fig1.update_layout(bargap=0.05)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Correlation Heatmap")

    numeric_cols = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "MedHouseVal",
        "RoomsPerHousehold", "BedroomsPerRoom"
    ]

    corr = df[numeric_cols].corr()

    fig2 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        text=np.round(corr.values, 2),
        texttemplate="%{text}"
    ))

    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🗺️ Geographic Price Map")

    sample = df.sample(3000, random_state=42)

    fig3 = px.scatter(
        sample,
        x="Longitude",
        y="Latitude",
        color="MedHouseVal",
        size="MedInc",
        color_continuous_scale="Plasma",
        opacity=0.7
    )

    fig3.update_layout(height=480)
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Each dot is a census block. Size = median income. Color = house price.")


# TAB 3
with tab3:
    st.header("Model Comparison")

    sample_idx = np.random.choice(len(models["y_test"]), 300, replace=False)

    y_sample = models["y_test"].values[sample_idx] * 100000
    rf_sample = models["rf_preds"][sample_idx] * 100000
    lr_sample = models["lr_preds"][sample_idx] * 100000

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=y_sample,
        y=rf_sample,
        mode='markers',
        name='Random Forest',
        marker=dict(opacity=0.6, size=5)
    ))

    fig4.add_trace(go.Scatter(
        x=y_sample,
        y=lr_sample,
        mode='markers',
        name='Linear Regression',
        marker=dict(opacity=0.6, size=5)
    ))

    max_val = max(y_sample.max(), rf_sample.max())

    fig4.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', width=1.5)
    ))

    fig4.update_layout(
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        height=480
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.info("Random Forest is significantly more accurate than Linear Regression, especially for high prices.")
