import streamlit as st
import pandas as pd
from model import fetch_data, calculate_indicators, calculate_support_resistance, predict_future_prices
from visualizations import (
    plot_stock_price, plot_predictions, plot_technical_indicators, plot_risk_levels,
    plot_feature_importance, plot_candlestick, plot_volume, plot_moving_averages,
    plot_feature_correlations
)

def sidebar():
    st.sidebar.title("Stock Analysis Dashboard")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value='SBILIFE.NS')
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2021-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-09-01'))
    algorithm = st.sidebar.selectbox("Select Prediction Algorithm", ['Linear Regression', 'ARIMA','Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost', 'LSTM', 'SARIMA'])
    return ticker, start_date, end_date, algorithm

def display_analysis(data, algorithm):
    if data is not None:
        try:
            support_price, resistance_price = calculate_support_resistance(data)
            future_prices, mae, r2, accuracy, conf_matrix = predict_future_prices(data, algorithm)

            if future_prices is not None:
                st.write("### Technical Indicators")
                indicators = {
                    'SMA_50': data['SMA_50'].iloc[-1],
                    'EMA_50': data['EMA_50'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'MACD_Signal': data['MACD_Signal'].iloc[-1],
                    'Bollinger_High': data['Bollinger_High'].iloc[-1],
                    'Bollinger_Low': data['Bollinger_Low'].iloc[-1],
                    'ATR': data['ATR'].iloc[-1],
                    'OBV': data['OBV'].iloc[-1]
                }
                for key, value in indicators.items():
                    with st.expander(f"{key} Description"):
                        st.write(f"{key}: {value:.2f}")
                        st.write(get_indicator_description(key))

                st.write("### Support and Resistance Levels")
                st.write(f"Support Price: {support_price:.2f}")
                st.write(f"Resistance Price: {resistance_price:.2f}")

                st.write("### Future Price Predictions")
                st.write(pd.DataFrame({'Day': range(1, len(future_prices) + 1), 'Predicted Price': future_prices}))

                if accuracy is not None and conf_matrix is not None:
                    st.write(f"**Model Accuracy:** {accuracy:.2f}")
                    st.write("**Confusion Matrix:**")
                    st.pyplot(ConfusionMatrixDisplay(conf_matrix).plot())

                if mae is not None and r2 is not None:
                    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                    st.write(f"**R-squared (R2):** {r2:.2f}")
            else:
                st.error("Model selection or prediction failed. Please check your inputs and try again.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")

def get_indicator_description(indicator):
    descriptions = {
        'SMA_50': "SMA_50 (50-day Simple Moving Average): Yeh 50 din ka average hai jo bataata hai stock ka long-term trend. Agar yeh price line se upar hai, toh stock ka trend upward hai.",
        'EMA_50': "EMA_50 (50-day Exponential Moving Average): Yeh bhi ek average hai lekin recent prices ko zyada weightage deta hai. Stock ka short-term trend dikhata hai.",
        'RSI': "RSI (Relative Strength Index): Yeh indicator stock ke overbought ya oversold condition ko dikhata hai. 70 se zyada overbought, aur 30 se kam oversold hai.",
        'MACD': "MACD: Yeh indicator short-term aur long-term moving averages ke beech ka difference dikhata hai.",
        'MACD_Signal': "MACD Signal: Yeh line MACD ke signal ko dikhata hai. Jab MACD line isse cross karti hai, toh trend change hota hai.",
        'Bollinger_High': "Bollinger High: Yeh line stock price ki upper boundary dikhati hai. Agar price isse upar hai, toh stock overbought ho sakta hai.",
        'Bollinger_Low': "Bollinger Low: Yeh line stock price ki lower boundary dikhati hai. Agar price isse neeche hai, toh stock oversold ho sakta hai.",
        'ATR': "ATR (Average True Range): Yeh indicator stock ki volatility dikhata hai. Zyada ATR matlab zyada price fluctuations.",
        'OBV': "OBV (On-Balance Volume): Yeh volume aur price ke relationship ko dikhata hai. Jab OBV badh raha hai, toh stock ka demand badh raha hai."
    }
    return descriptions.get(indicator, "Description not available")

def display_visualizations(data, algorithm):
    if data is not None:
        choice = st.sidebar.selectbox(
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
                plot_stock_price(data)
            elif choice == "Predictions vs Actual":
                future_prices, _, _, _, _ = predict_future_prices(data, algorithm)
                if future_prices is not None:
                    st.line_chart(pd.DataFrame({'Actual Prices': data['Close'], 'Predicted Prices': pd.Series(future_prices).values}))
                else:
                    st.error("Failed to fetch predictions.")
            elif choice == "Technical Indicators":
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
            st.error(f"An error occurred during visualization: {e}")
    else:
        st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
