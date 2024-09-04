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
from ui import display_analysis
from logger import get_logger

from dashboard import display_dashboard

logger = get_logger(__name__)

st.title("Stock Analysis and Prediction")



# Sidebar for navigation
st.sidebar.title("Navigation")

# Initialize `page` to "Analytics" by default
if 'page' not in st.session_state:
    st.session_state['page'] = "Analytics"

if st.sidebar.button("Analytics"):
    st.session_state['page'] = "Analytics"
if st.sidebar.button("Dashboard"):
    st.session_state['page'] = "Dashboard"
if st.sidebar.button("Profile"):
    st.session_state['page'] = "Profile"

page = st.session_state['page']

# Function to fetch and prepare data
def get_data():
    ticker = st.session_state.get('ticker')
    start_date = st.session_state.get('start_date')
    end_date = st.session_state.get('end_date')
    
    try:
        data = fetch_data(ticker, start_date, end_date)
        if data is not None:
            data = calculate_indicators(data)
            return data
        else:
            st.error("Failed to fetch data. Please check the stock ticker symbol and date range.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Display content based on selected page
if page == "Analytics":
    st.header("Analytics")

    # Data input section
    ticker = st.text_input("Stock Ticker", "BHEL.NS")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-09-04"))
    algorithm = st.selectbox(
        "Choose an Algorithm",
        ["Linear Regression", "Random Forest", "Support Vector Machine"]
    )
    st.session_state['ticker'] = ticker
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['algorithm'] = algorithm

    # Tabs for Analyze and Visualization under Analytics
    tab1, tab2 = st.tabs(["Analyze", "Visualization"])

    # Analyze Tab
    with tab1:
        if st.button("Analyze"):
            data = get_data()
            if data is not None:
                display_analysis(data, st.session_state.get('algorithm'))

    # Visualization Tab
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
                    "Stock Price","Volume",
                    "Moving Averages",
                    "Feature Correlations",
                    "Predictions vs Actual",
                    "Technical Indicators",
                    "Risk Levels",
                    "Feature Importance",
                    "Candlestick"
                ]
            )
            
            try:
                if choice == "Stock Price":
                    plot_stock_price(data, st.session_state.get('ticker'), indicators)
                elif choice == "Predictions vs Actual":
                    future_prices, _, _, _, _ = predict_future_prices(data, st.session_state.get('algorithm'))
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

elif page == "Dashboard":
    
    
    # Display the main dashboard
    display_dashboard()

    st.write("<div style='background-color: black; color: white; padding: 10px;'>Coming Soon A lot Updates.......</div>", unsafe_allow_html=True)
    
elif page == "Profile":
    st.image("https://via.placeholder.com/150", caption="User Profile Photo")
    st.write("### User Profile")
    st.write("Name: Nandan Dutta")
    st.write("Role: Data Analyst")
    st.write("Email: n.dutta25@gmail.com")




st.markdown(
    """
    <style>
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .blinking-heart {
        animation: blink 1s infinite;
    }
    </style>
    <div style='background-color: #f1f1f1; color: #333; padding: 5px; text-align: center; border-top: 1px solid #ddd;'>
        <p>Made with <span class="blinking-heart">❤️</span> from Nandan</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Display animated running disclaimer text
st.write(
    """
    <div style='background-color: black; color: white; padding: 10px; border-radius: 5px;'>
        <marquee behavior="scroll" direction="left" scrollamount="5" style="font-size: 14px;">
            This project is for educational purposes only. The information provided here should not be used for real investment decisions. Please perform your own research and consult with a financial advisor before making any investment decisions. Use this information at your own risk.
        </marquee>
    </div>
    """,
    unsafe_allow_html=True
)
