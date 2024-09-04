
import streamlit as st
import pandas as pd
import numpy as np
import logging

from model import fetch_data, calculate_indicators, calculate_support_resistance, predict_future_prices
from visualizations import (
    plot_stock_price, plot_predictions, plot_technical_indicators, plot_risk_levels,
    plot_feature_importance, plot_candlestick, plot_volume, plot_moving_averages,
    plot_feature_correlations
)
from sklearn.metrics import ConfusionMatrixDisplay

from logger import get_logger

logger = get_logger(__name__)

st.title("Stock Analysis and Prediction")

# User inputs
ticker = st.text_input("Enter Stock Ticker Symbol:", value='SBILIFE.NS')
start_date = st.date_input("Start Date", value=pd.to_datetime('2021-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2024-09-01'))
algorithm = st.selectbox("Select Prediction Algorithm", ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost', 'LSTM', 'ARIMA', 'SARIMA'])

# Function to fetch and prepare data
def get_data():
    try:
        data = fetch_data(ticker, start_date, end_date)
        if data is not None:
            data = calculate_indicators(data)
            return data
        else:
            logger.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
            st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
            return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")
        return None

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Analysis", "Visualization"])

with tab1:
    if st.button("Analyze"):
        data = get_data()
        if data is not None:
            try:
                # Calculate support and resistance
                support_price, resistance_price = calculate_support_resistance(data)

                # Predict future price
                future_prices, mae, r2, accuracy, conf_matrix = predict_future_prices(data, algorithm)

                if future_prices is not None:
                    # Display technical indicators with Hinglish explanations
                    st.write("### Technical Indicators")
                    st.write(f"SMA_50 (50-day Simple Moving Average): {data['SMA_50'].iloc[-1]:.2f}")
                    st.write("**SMA_50**: Yeh 50 din ka average hai jo bataata hai stock ka long-term trend. Agar yeh price line se upar hai, toh stock ka trend upward hai.")
                    st.write(f"EMA_50 (50-day Exponential Moving Average): {data['EMA_50'].iloc[-1]:.2f}")
                    st.write("**EMA_50**: Yeh bhi ek average hai lekin recent prices ko zyada weightage deta hai. Stock ka short-term trend dikhata hai.")
                    st.write(f"RSI (Relative Strength Index): {data['RSI'].iloc[-1]:.2f}")
                    st.write("**RSI**: Yeh indicator stock ke overbought ya oversold condition ko dikhata hai. 70 se zyada overbought, aur 30 se kam oversold hai.")
                    st.write(f"MACD: {data['MACD'].iloc[-1]:.2f}")
                    st.write("**MACD**: Yeh indicator short-term aur long-term moving averages ke beech ka difference dikhata hai.")
                    st.write(f"MACD Signal: {data['MACD_Signal'].iloc[-1]:.2f}")
                    st.write("**MACD Signal**: Yeh line MACD ke signal ko dikhata hai. Jab MACD line isse cross karti hai, toh trend change hota hai.")
                    st.write(f"Bollinger High: {data['Bollinger_High'].iloc[-1]:.2f}")
                    st.write("**Bollinger High**: Yeh line stock price ki upper boundary dikhati hai. Agar price isse upar hai, toh stock overbought ho sakta hai.")
                    st.write(f"Bollinger Low: {data['Bollinger_Low'].iloc[-1]:.2f}")
                    st.write("**Bollinger Low**: Yeh line stock price ki lower boundary dikhati hai. Agar price isse neeche hai, toh stock oversold ho sakta hai.")
                    st.write(f"ATR (Average True Range): {data['ATR'].iloc[-1]:.2f}")
                    st.write("**ATR**: Yeh indicator stock ki volatility dikhata hai. Zyada ATR matlab zyada price fluctuations.")
                    st.write(f"OBV (On-Balance Volume): {data['OBV'].iloc[-1]:.2f}")
                    st.write("**OBV**: Yeh volume aur price ke relationship ko dikhata hai. Jab OBV badh raha hai, toh stock ka demand badh raha hai.")
                    
                    st.write("### Support and Resistance Levels")
                    st.write(f"Support Price: {support_price:.2f}")
                    st.write(f"Resistance Price: {resistance_price:.2f}")
                    
                    st.write("### Future Price Predictions")
                    for i, price in enumerate(future_prices):
                        st.write(f"Day {i+1}: {price:.2f}")
                    
                    if accuracy is not None and conf_matrix is not None:
                        st.write(f"**Model Accuracy:** {accuracy:.2f}")
                        st.write("**Accuracy**: Yeh metric dikhata hai ki model ne sahi predictions kitne percentage bar kiye hain.")
                        st.write("**Confusion Matrix:**")
                        st.pyplot(ConfusionMatrixDisplay(conf_matrix).plot())
                        
                    if mae is not None and r2 is not None:
                        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                        st.write("**MAE**: Yeh average error hai jo model ke predictions aur actual values ke beech ka difference dikhata hai.")
                        st.write(f"**R-squared (R2):** {r2:.2f}")
                        st.write("**R2**: Yeh metric dikhata hai ki model ne data ke variation ko kitna achhe se explain kiya hai. 1 ka matlab perfect fit hai.")
                else:
                    st.error("Model selection or prediction failed. Please check your inputs and try again.")
                    logger.error("Model selection or prediction failed. Please check inputs and try again.")
            except Exception as e:
                logger.error(f"An error occurred during analysis: {e}")
                st.error(f"An error occurred during analysis: {e}")
        else:
            st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
            logger.error("Failed to fetch data. Please check the stock ticker symbol and date range.")

with tab2:
    st.write("### Visualizations")
    
    # Fetch and prepare data for visualization
    data = get_data()
    if data is not None:
        indicators = {
            'SMA_50': data['SMA_50'],
            'EMA_50': data['EMA_50'],
            'RSI': data['RSI'],
            'MACD': data['MACD'],
            'MACD_Signal': data['MACD_Signal'],
            'Bollinger_High': data['Bollinger_High'],
            'Bollinger_Low': data['Bollinger_Low'],
            'ATR': data['ATR'],
            'OBV': data['OBV']
        }
        
        # Visualization choices
        choice = st.selectbox(
            "Choose a type of visualization",
            [
                "Stock Price",
                "Predictions vs Actual",
                "Technical Indicators",
                "Risk Levels",
                "Feature Importance",
                "Candlestick",
                "Volume",
                "Moving Averages",
                "Feature Correlations"
            ]
        )
        
        try:
            if choice == "Stock Price":
                plot_stock_price(data, ticker, indicators)
            elif choice == "Predictions vs Actual":
                future_prices, _, _, _, _ = predict_future_prices(data, algorithm)
                if future_prices is not None:
                    st.line_chart(pd.DataFrame({'Actual Prices': data['Close'], 'Predicted Prices': pd.Series(future_prices).values}))
                else:
                    st.error("Failed to fetch predictions.")
                    logger.error("Failed to fetch predictions.")
            elif choice == "Technical Indicators":
                plot_technical_indicators(data, indicators)
            elif choice == "Risk Levels":
                plot_risk_levels(data)
            elif choice == "Feature Importance":
                plot_feature_importance()
            elif choice == "Candlestick":
                plot_candlestick(data)
            elif choice == "Volume":
                plot_volume(data)
            elif choice == "Moving Averages":
                plot_moving_averages(data)
            elif choice == "Feature Correlations":
                plot_feature_correlations(data)
        except Exception as e:
            logger.error(f"An error occurred during visualization: {e}")
            st.error(f"An error occurred during visualization: {e}")
    else:
        st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
        logger.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
