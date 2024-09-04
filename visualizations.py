import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import logging
import plotly.express as px
import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.dates as mdates
# from mplfinance.original_flavor import candlestick_ohlc
# import logging
# import plotly.express as px
# import streamlit as st

from logger import get_logger

logger = get_logger(__name__)


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

def plot_technical_indicators(data: pd.DataFrame, indicators: dict):
    """
    Plot technical indicators along with the stock price.
    """
    logger.info("Plotting stock price with technical indicators.")
    
    # Matplotlib Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    
    for name, values in indicators.items():
        plt.plot(data['Date'], values, label=name)
    
    plt.title('Stock Price with Technical Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Render the plot using Streamlit
    st.pyplot(plt)
    
    # Plotly Plot (interactive)
    fig = px.line(data, x='Date', y='Close', title='Stock Price with Technical Indicators')
    for name, values in indicators.items():
        fig.add_scatter(x=data['Date'], y=values, mode='lines', name=name)
    
    # Render the interactive plot using Streamlit
    st.plotly_chart(fig)


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
    plt.show()

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
    plt.show()



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
    plt.show()




# def plot_candlestick(data: pd.DataFrame, ticker: str):
#     """
#     Plot candlestick chart for stock prices.
#     """
#     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    
#     # Check if all required columns are present
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         logger.error(f"Missing columns in data for plot_candlestick: {', '.join(missing_columns)}")
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     logger.info(f"Plotting candlestick chart for {ticker}.")
#     data = data[required_columns]
#     data['Date'] = pd.to_datetime(data['Date'])
#     data['Date'] = mdates.date2num(data['Date'])
    
#     fig, ax = plt.subplots(figsize=(14, 7))
#     candlestick_ohlc(ax, data.values, width=0.6, colorup='green', colordown='red')
    
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.title(f'{ticker} Candlestick Chart')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()
