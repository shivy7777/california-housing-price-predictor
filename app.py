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
page_icon=" ",
layout="wide",
initial_sidebar_state="expanded"
)
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
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
<h3> About This Project</h3>
<p>
This app predicts California housing prices using the classic <strong>California Housing Dataset</strong>
from the 1990 census. Two machine learning models — <strong>Linear Regression</strong> and
<strong>Random Forest</strong> — are trained and compared side-by-side.
The app also engineers three new features beyond the raw dataset to improve predictive power.
</p>
<br>
<p>
<strong>Why housing price prediction matters:</strong> Understanding what drives property values helps
buyers make smarter decisions, helps cities identify undervalued neighborhoods, and is a core
application of regression modeling in data science.
</p>
<br>
<span class="badge">sklearn</span>
<span class="badge">Random Forest</span>
<span class="badge">Linear Regression</span>
<span class="badge">Plotly</span>
<span class="badge">Feature Engineering</span>
<span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────
st.sidebar.markdown("## Input Features")
st.sidebar.markdown("Adjust values to generate a price prediction.")
med_inc = st.sidebar.slider("Median Income (tens of thousands)", 0.5, 15.0, 5.0, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 10.0, 5.0, 0.1)
ave_bedrms = st.sidebar.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.0, 0.1)
population = st.sidebar.slider("Block Population", 3, 5000, 1000)
ave_occup = st.sidebar.slider("Avg Occupants per Household", 1.0, 6.0, 2.5, 0.1)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 37.0, 0.1)
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -120.0, 0.1)
# Derived features from inputs

rooms_per_hh = ave_rooms / max(house_age, 1)
bedrms_per_room = ave_bedrms / max(ave_rooms, 1)
pop_density = population / max(ave_occup, 1)
user_input = np.array([[
med_inc, house_age, ave_rooms, ave_bedrms,
population, ave_occup, latitude, longitude,
rooms_per_hh, bedrms_per_room, pop_density
]])
# Predict with both models
rf_prediction = models["rf"].predict(user_input)[0] * 100000
lr_input_scaled = models["scaler"].transform(user_input)
lr_prediction = models["lr"].predict(lr_input_scaled)[0] * 100000
# Confidence range (RF std across trees)
tree_preds = np.array([tree.predict(user_input)[0] for tree in models["rf"].estimators_])
confidence_low = np.percentile(tree_preds, 10) * 100000
confidence_high = np.percentile(tree_preds, 90) * 100000

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Prediction", " Exploratory Analysis", " Model Comparison"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
st.markdown("## Price Prediction")
col1, col2 = st.columns(2)
with col1:
st.markdown(f"""
<div class="prediction-box">
<p>Random Forest Estimate</p>
<h1>${rf_prediction:,.0f}</h1>
<p>90% confidence range: ${confidence_low:,.0f} – ${confidence_high:,.0f}</p>
</div>
""", unsafe_allow_html=True)
with col2:
st.markdown(f"""
<div class="prediction-box" style="border-color: #6b9ff0; background: linear-gradient(135deg, #1f2a3a, #1a2035);">

<p>Linear Regression Estimate</p>
<h1 style="color: #6b9ff0 !important;">${lr_prediction:,.0f}</h1>
<p>Simpler model, higher bias</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# Model metrics side by side
st.markdown("### Model Performance on Test Set")
c1, c2, c3, c4 = st.columns(4)
with c1:
st.markdown(f'<div class="metric-card"><h2>{models["rf_r2"]:.3f}</h2><p>Random Forest R2</p></div>', unsafe_allow_html=True)
with c2:
st.markdown(f'<div class="metric-card"><h2>${models["rf_rmse"]*100000:,.0f}</h2><p>Random Forest RMSE</p></div>', unsafe_allow_html=True)
with c3:
st.markdown(f'<div class="metric-card"><h2>{models["lr_r2"]:.3f}</h2><p>Linear Regression R2</p></div>', unsafe_allow_html=True)
with c4:
st.markdown(f'<div class="metric-card"><h2>${models["lr_rmse"]*100000:,.0f}</h2><p>Linear Regression RMSE</p></div>', unsafe_allow_html=True)
# Feature importance
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### What Drives the Price?")
importances = pd.Series(
models["rf"].feature_importances_,
index=models["features"]
).sort_values(ascending=True)
fig = go.Figure(go.Bar(
x=importances.values,
y=importances.index,
orientation='h',
marker=dict(
color=importances.values,
colorscale=[[0, '#1a1f2e'], [1, '#f0c96b']],
showscale=False
)
))
fig.update_layout(
paper_bgcolor='#0f1117',
plot_bgcolor='#0f1117',
font=dict(color='#e8e8e8', family='DM Sans'),
xaxis=dict(title='Importance Score', gridcolor='#2a2f3e'),
yaxis=dict(gridcolor='#2a2f3e'),
margin=dict(l=0, r=0, t=10, b=0),
height=380

)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tab2:
st.markdown("## Exploratory Data Analysis")
# Visualization 1 — Price Distribution
st.markdown("### Price Distribution")
fig1 = px.histogram(
df, x="MedHouseVal", nbins=60,
color_discrete_sequence=["#f0c96b"],
labels={"MedHouseVal": "Median House Value ($100k)"}
)
fig1.update_layout(
paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
font=dict(color='#e8e8e8', family='DM Sans'),
bargap=0.05,
xaxis=dict(gridcolor='#2a2f3e'),
yaxis=dict(gridcolor='#2a2f3e')
)
st.plotly_chart(fig1, use_container_width=True)
# Visualization 2 — Correlation Heatmap
st.markdown("### Correlation Heatmap")
numeric_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
"Population", "AveOccup", "MedHouseVal",
"RoomsPerHousehold", "BedroomsPerRoom"]
corr = df[numeric_cols].corr()
fig2 = go.Figure(go.Heatmap(
z=corr.values,
x=corr.columns,
y=corr.columns,
colorscale=[[0, '#1a2f3a'], [0.5, '#2a2f3e'], [1, '#f0c96b']],
text=np.round(corr.values, 2),
texttemplate="%{text}",
showscale=True
))
fig2.update_layout(
paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
font=dict(color='#e8e8e8', family='DM Sans'),
height=450,
margin=dict(l=0, r=0, t=10, b=0)

)
st.plotly_chart(fig2, use_container_width=True)
# Visualization 3 — Geographic Scatter
st.markdown("### Geographic Price Map")
sample = df.sample(3000, random_state=42)
fig3 = px.scatter(
sample, x="Longitude", y="Latitude",
color="MedHouseVal",
size="MedInc",
color_continuous_scale=["#1a1f2e", "#f0c96b", "#ff6b6b"],
labels={"MedHouseVal": "Price ($100k)", "MedInc": "Income"},
opacity=0.7
)
fig3.update_layout(
paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
font=dict(color='#e8e8e8', family='DM Sans'),
xaxis=dict(title="Longitude", gridcolor='#2a2f3e'),
yaxis=dict(title="Latitude", gridcolor='#2a2f3e'),
height=480,
margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig3, use_container_width=True)
st.caption("Each dot is a census block. Size = median income. Color = house price. Coastal areas clearly command higher prices.")

# ══════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
st.markdown("## Model Comparison")
st.markdown("Random Forest vs Linear Regression — predicted vs actual on the test set.")
sample_idx = np.random.choice(len(models["y_test"]), 300, replace=False)
y_sample = models["y_test"].values[sample_idx] * 100000
rf_sample = models["rf_preds"][sample_idx] * 100000
lr_sample = models["lr_preds"][sample_idx] * 100000
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
x=y_sample, y=rf_sample, mode='markers',
name='Random Forest',
marker=dict(color='#f0c96b', opacity=0.6, size=5)
))
fig4.add_trace(go.Scatter(
x=y_sample, y=lr_sample, mode='markers',
name='Linear Regression',

marker=dict(color='#6b9ff0', opacity=0.6, size=5)
))
# Perfect prediction line
max_val = max(y_sample.max(), rf_sample.max())
fig4.add_trace(go.Scatter(
x=[0, max_val], y=[0, max_val],
mode='lines', name='Perfect Prediction',
line=dict(color='#ff6b6b', dash='dash', width=1.5)
))
fig4.update_layout(
paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
font=dict(color='#e8e8e8', family='DM Sans'),
xaxis=dict(title='Actual Price ($)', gridcolor='#2a2f3e'),
yaxis=dict(title='Predicted Price ($)', gridcolor='#2a2f3e'),
legend=dict(bgcolor='#1a1f2e', bordercolor='#2a2f3e'),
height=480
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
**Key Takeaway:** Random Forest (gold) clusters much tighter around the perfect prediction line
compared to Linear Regression (blue), especially at higher price points where linear models
tend to underestimate. This is because housing prices have non-linear relationships with features
like location and income that tree-based models capture more naturally.
""")
