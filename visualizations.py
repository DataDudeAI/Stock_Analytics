import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import logging
import plotly.express as px
import streamlit as st

from model import predict_future_prices

from logger import get_logger

logger = get_logger(__name__)




def plot_stock_price(data: pd.DataFrame, ticker: str, indicators: dict = None, 
                     color='blue', line_style='-', title=None):
    """
    Plot the stock price with optional indicators and customization.
    """
    required_columns = ['Date', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing columns in data for plot_stock_price: {', '.join(missing_columns)}")
        raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
    logger.info(f"Plotting stock price for {ticker}.")
    
    # Matplotlib Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color=color, linestyle=line_style)
    
    if indicators:
        for name, values in indicators.items():
            plt.plot(data['Date'], values, label=name)
    
    plt.title(title if title else f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Render the plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.line(data, x='Date', y='Close', title=title if title else f'{ticker} Stock Price')
    if indicators:
        for name, values in indicators.items():
            fig.add_scatter(x=data['Date'], y=values, mode='lines', name=name)
    
    # Render the interactive plot using Streamlit
    st.plotly_chart(fig)

def plot_predictions(data: pd.DataFrame, predictions: pd.Series, ticker: str, 
                     actual_color='blue', predicted_color='red', line_style_actual='-', line_style_predicted='--'):
    """
    Plot actual vs predicted stock prices with customization.
    """
    logger.info(f"Plotting actual vs predicted prices for {ticker}.")
    
    # Matplotlib Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Actual Prices', color=actual_color, linestyle=line_style_actual)
    plt.plot(data['Date'], predictions, label='Predicted Prices', color=predicted_color, linestyle=line_style_predicted)
    
    plt.title(f'{ticker} Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Render the plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.line(data, x='Date', y='Close', title=f'{ticker} Actual vs Predicted Prices')
    fig.add_scatter(x=data['Date'], y=predictions, mode='lines', name='Predicted Prices', line=dict(color=predicted_color))
    
    # Render the interactive plot using Streamlit
    st.plotly_chart(fig)

def generate_predictions(model, test_data):
    """
    Generate predictions using the model for the given test data.
    """
    try:
        # Extract relevant features for the model
    
        features = test_data[['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']]  # Adjust features based on your model
        predictions = model.predict(features)
        return predictions
    except KeyError as e:
        logger.error(f"Feature key error: {e}")
        st.error(f"Feature key error: {e}")
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        st.error(f"An error occurred during prediction: {e}")





def plot_technical_indicators(data: pd.DataFrame, indicators: dict, model, days=10):
    """
    Plot technical indicators along with the stock price and predictions.
    """
    logger.info("Plotting stock price with technical indicators and predictions.")
    
    # Ensure all indicators have the same length as the data
    for name, values in indicators.items():
        if len(values) != len(data):
            logger.error(f"Indicator '{name}' length {len(values)} does not match data length {len(data)}.")
            st.error(f"Indicator '{name}' length {len(values)} does not match data length {len(data)}.")
            return

    # Generate the last 30 days' dates
    end_date = data['Date'].max()
    start_date = end_date - pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter data for the last 30 days
    last_30_days_data = data[data['Date'].isin(date_range)]

    # Prepare test data for predictions
    test_data = last_30_days_data.copy()
    
    # Generate future predictions
    future_prices, _, _, _, _ = predict_future_prices(data, model, days)
    
    if future_prices is not None:
        # Generate future dates
        future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_prices
        })

        # Matplotlib Plot
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.plot(future_df['Date'], future_df['Predicted_Close'], label='Predicted Price', color='orange', linestyle='--')
        
        for name, values in indicators.items():
            plt.plot(data['Date'], values, label=name)
        
        plt.title('Stock Price with Technical Indicators and Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        # Render the plot using Streamlit
        st.pyplot(plt)
        
        # Plotly Plot (interactive)
        fig = px.line(data, x='Date', y='Close', title='Stock Price with Technical Indicators and Predictions')
        fig.add_scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted Price', line=dict(color='orange', dash='dash'))
        
        for name, values in indicators.items():
            fig.add_scatter(x=data['Date'], y=values, mode='lines', name=name)
        
        # Render the interactive plot using Streamlit
        st.plotly_chart(fig)
    else:
        st.error("No predictions available.")






def plot_risk_levels(data: pd.DataFrame, risk_levels: pd.Series, cmap='coolwarm'):
    """
    Plot risk levels with stock prices and customization.
    """
    logger.info("Plotting stock prices with risk levels.")
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.scatter(data['Date'], data['Close'], c=risk_levels, cmap=cmap, label='Risk Levels', alpha=0.7)
    
    plt.title('Stock Prices with Risk Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.colorbar(label='Risk Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    # Render Matplotlib plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.scatter(data, x='Date', y='Close', color=risk_levels, color_continuous_scale=cmap, 
                     title='Stock Prices with Risk Levels', labels={'color': 'Risk Level'})
    
    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)

def plot_feature_importance(importances: pd.Series, feature_names: list):
    """
    Plot feature importance for machine learning models.
    """
    logger.info("Plotting feature importance.")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.tight_layout()

    # Render Matplotlib plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.bar(x=importances, y=feature_names, orientation='h', 
                 title='Feature Importances', labels={'x': 'Importance', 'y': 'Feature'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)

def plot_candlestick(data: pd.DataFrame, ticker: str):
    """
    Plot candlestick chart for stock prices.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing columns in data for plot_candlestick: {', '.join(missing_columns)}")
        raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
    logger.info(f"Plotting candlestick chart for {ticker}.")
    data = data[required_columns]
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = mdates.date2num(data['Date'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    candlestick_ohlc(ax, data.values, width=0.6, colorup='green', colordown='red')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.title(f'{ticker} Candlestick Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Render Matplotlib plot using Streamlit
    st.pyplot(fig)

    # Plotly Plot (interactive)
    fig = px.line(data, x='Date', y=['Open', 'High', 'Low', 'Close'], 
                  title=f'{ticker} Candlestick Chart')
    
    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)

def plot_volume(data: pd.DataFrame):
    """
    Plot trading volume alongside stock price.
    """
    logger.info("Plotting stock price and trading volume.")
    
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.title('Stock Price and Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.bar(data['Date'], data['Volume'], color='grey', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Volume')
    
    plt.tight_layout()
    plt.xticks(rotation=45)

    # Render Matplotlib plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.bar(data, x='Date', y='Volume', title='Trading Volume',
                 labels={'Volume': 'Volume', 'Date': 'Date'})
    
    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)

def plot_moving_averages(data: pd.DataFrame, short_window: int = 20, long_window: int = 50):
    """
    Plot moving averages along with the stock price.
    """
    logger.info("Calculating and plotting moving averages.")
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], data['Short_MA'], label=f'Short {short_window}-day MA', color='orange')
    plt.plot(data['Date'], data['Long_MA'], label=f'Long {long_window}-day MA', color='purple')
    
    plt.title('Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    # Render Matplotlib plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.line(data, x='Date', y=['Close', 'Short_MA', 'Long_MA'], 
                  title='Stock Price with Moving Averages')
    
    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)

def plot_feature_correlations(data: pd.DataFrame):
    """
    Plot correlation heatmap of features.
    """
    logger.info("Plotting feature correlations heatmap.")
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    
    plt.title('Feature Correlations')
    plt.tight_layout()

    # Render Matplotlib plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.imshow(correlation_matrix, text_auto=True, 
                    title='Feature Correlations', labels={'color': 'Correlation'})
    
    # Render the interactive Plotly plot using Streamlit
    st.plotly_chart(fig)
