# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Page Configuration ---
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔧 Forecast Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])
uploaded_file = st.sidebar.file_uploader("Upload Stock CSV", type="csv")

# --- Theme Styling ---
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: white; }
        .css-1v0mbdj p, h1, h2, h3 { color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Load and Clean Data ---
def load_data(df):
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df = df.sort_index()
    return df[['close']]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Stock_data.csv")  # Replace this with your actual default data file

df = load_data(df)

# --- Forecasting Functions ---
def forecast_arima(df):
    series = df['close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    forecast_index = pd.date_range(start=series.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=forecast_index)

def forecast_prophet(df):
    data = df.reset_index()[['date', 'close']]
    data.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].tail(30)

def forecast_lstm(df):
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    forecast_input = scaled_data[-60:].reshape(1, 60, 1)
    forecast = []
    for _ in range(30):
        pred = model.predict(forecast_input)[0][0]
        forecast.append(pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_index = pd.date_range(start=df.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=forecast_index)

# --- UI Layout ---
st.title("📈 Stock Market Forecasting Dashboard")

tab1, tab2 = st.tabs(["📊 Data Overview", "🔮 Forecast Results"])

with tab1:
    st.subheader("Stock Closing Prices")
    st.line_chart(df['close'])

    st.subheader("Recent Records")
    st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    st.subheader(f"{model_choice} Forecast")

    with st.expander(f"What is {model_choice}?", expanded=False):
        if model_choice == "ARIMA":
            st.write("ARIMA is a traditional model that predicts future values based on past trends using autoregression and moving averages.")
        elif model_choice == "Prophet":
            st.write("Prophet is a model by Facebook designed to handle seasonality and trend changes in time series forecasting.")
        elif model_choice == "LSTM":
            st.write("LSTM (Long Short-Term Memory) is a deep learning model that learns long-term dependencies in time series data.")

    if model_choice == "ARIMA":
        forecast = forecast_arima(df)
    elif model_choice == "Prophet":
        forecast = forecast_prophet(df)
    elif model_choice == "LSTM":
        forecast = forecast_lstm(df)

    st.line_chart(pd.concat([df['close'].iloc[-30:], forecast], axis=0))
    st.subheader("Forecast Table")
    st.dataframe(forecast.to_frame(name="Predicted Close Price"), use_container_width=True)

# --- Conclusion ---
st.markdown("---")
st.markdown("## ✅ Conclusion")
st.write("""
This dashboard demonstrates how different time series models (ARIMA, Prophet, and LSTM) can be applied to real stock market data
to make short-term predictions. Users can upload their own data, select models, and visualize forecast results interactively.

**Key Takeaways:**
- ARIMA is simple and effective for stable patterns.
- Prophet handles seasonality well with interpretable output.
- LSTM can capture complex patterns using deep learning.

This tool offers a great starting point for financial analysts, students, or businesses exploring data-driven forecasting.
""")

st.markdown("📘 **Developed by [Your Name]** · Data Analytics Project · Powered by Streamlit")