## Before you read, this is a work in progress, so be aware it may not work at times.




[README.md](https://github.com/user-attachments/files/26198737/README.md)
# California Housing Price Predictor

A machine learning web app that predicts California housing prices using Random Forest and Linear Regression, with interactive visualizations and feature engineering.

## Features
- **Two models compared:** Random Forest vs Linear Regression
- **Feature engineering:** 3 new derived features beyond the raw dataset
- **Interactive prediction:** Adjust sliders to get real-time price estimates with confidence range
- **EDA Dashboard:** Price distribution, correlation heatmap, geographic price map
- **Model comparison:** Predicted vs actual scatter plot

## Tech Stack
- Python, Streamlit, scikit-learn, Plotly, Pandas, NumPy

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Free)
1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Deploy — done!

## Dataset
California Housing Dataset (1990 Census) via `sklearn.datasets.fetch_california_housing`
No API keys required.
