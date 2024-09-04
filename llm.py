
import streamlit as st
import requests
from model import fetch_data, calculate_indicators, calculate_support_resistance
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()

API_TOKEN = args.token
# Hugging Face API token and model URL
# API_TOKEN = os.environ['HUGGING_FACE_TOKEN']
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"

def generate_prompt(ticker, start_date, end_date):
    """Fetch data, calculate indicators, and prepare the prompt."""
    data = fetch_data(ticker, start_date, end_date)
    
    if data is None:
        return "No data available for the given ticker and date range.", None
    
    data = calculate_indicators(data)
    support, resistance = calculate_support_resistance(data)
    
    # Additional statistics
    highest_close = data['Close'].max()
    lowest_close = data['Close'].min()
    average_close = data['Close'].mean()
    average_volume = data['Volume'].mean()
    highest_volume = data['Volume'].max()
    lowest_volume = data['Volume'].min()
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std()
    
    recent_trend = "uptrend" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "downtrend" if data['Close'].iloc[-1] < data['Close'].iloc[0] else "sideways"
    
    # Summarize the key statistics
    summary = {
        'latest_close': data['Close'].iloc[-1],
        'SMA_50': data['SMA_50'].iloc[-1],
        'EMA_50': data['EMA_50'].iloc[-1],
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'MACD_Signal': data['MACD_Signal'].iloc[-1],
        'Bollinger_High': data['Bollinger_High'].iloc[-1],
        'Bollinger_Low': data['Bollinger_Low'].iloc[-1],
        'ATR': data['ATR'].iloc[-1],
        'OBV': data['OBV'].iloc[-1],
        'Support': support,
        'Resistance': resistance,
        'Highest_Close': highest_close,
        'Lowest_Close': lowest_close,
        'Average_Close': average_close,
        'Average_Volume': average_volume,
        'Highest_Volume': highest_volume,
        'Lowest_Volume': lowest_volume,
        'Volatility': volatility,
        'Recent_Trend': recent_trend,
        'Percentage_Change': (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100 if len(data) > 1 else 0
    }
    
    prompt = f"""
    Analyze the following stock data for {ticker} and provide a buy/sell recommendation:
    Latest Close Price: {summary['latest_close']}
    SMA 50: {summary['SMA_50']}
    EMA 50: {summary['EMA_50']}
    RSI: {summary['RSI']}
    MACD: {summary['MACD']}
    MACD Signal: {summary['MACD_Signal']}
    Bollinger Bands High: {summary['Bollinger_High']}
    Bollinger Bands Low: {summary['Bollinger_Low']}
    ATR: {summary['ATR']}
    OBV: {summary['OBV']}
    Support Level: {summary['Support']}
    Resistance Level: {summary['Resistance']}
    Highest Close Price: {summary['Highest_Close']}
    Lowest Close Price: {summary['Lowest_Close']}
    Average Close Price: {summary['Average_Close']}
    Average Volume: {summary['Average_Volume']}
    Highest Volume: {summary['Highest_Volume']}
    Lowest Volume: {summary['Lowest_Volume']}
    Volatility: {summary['Volatility']}
    Recent Trend: {summary['Recent_Trend']}
    Percentage Change: {summary['Percentage_Change']}%
    """
    
    return prompt, summary

def get_recommendation(prompt):
    """Get stock recommendation from Hugging Face API."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP issues
    result = response.json()
    
    return result[0]['generated_text'].strip()

def display_recommendation(ticker, start_date, end_date):
    """Fetch data, generate prompt, get recommendation, and display it in a nice format."""
    prompt, summary = generate_prompt(ticker, start_date, end_date)
    
    if summary is None:
        st.error(prompt)
        return

    try:
        recommendation = get_recommendation(prompt)
    except Exception as e:
        st.error(f"An error occurred while getting recommendation: {e}")
        return 

    # Display in a box/table format using Streamlit
    st.markdown(f"### Stock Analysis & Recommendation for {ticker}")

    st.markdown(f"""
    <div style='border:2px solid #4CAF50; padding: 15px; border-radius: 10px;'>
        <table style='width:100%; border-collapse: collapse;'>
            <tr>
                <th style='text-align: left;'>Indicator</th>
                <th style='text-align: left;'>Value</th>
            </tr>
            <tr>
                <td>Latest Close Price</td>
                <td>{summary['latest_close']}</td>
            </tr>
            <tr>
                <td>SMA 50</td>
                <td>{summary['SMA_50']}</td>
            </tr>
            <tr>
                <td>EMA 50</td>
                <td>{summary['EMA_50']}</td>
            </tr>
            <tr>
                <td>RSI</td>
                <td>{summary['RSI']}</td>
            </tr>
            <tr>
                <td>MACD</td>
                <td>{summary['MACD']}</td>
            </tr>
            <tr>
                <td>MACD Signal</td>
                <td>{summary['MACD_Signal']}</td>
            </tr>
            <tr>
                <td>Bollinger Bands High</td>
                <td>{summary['Bollinger_High']}</td>
            </tr>
            <tr>
                <td>Bollinger Bands Low</td>
                <td>{summary['Bollinger_Low']}</td>
            </tr>
            <tr>
                <td>ATR</td>
                <td>{summary['ATR']}</td>
            </tr>
            <tr>
                <td>OBV</td>
                <td>{summary['OBV']}</td>
            </tr>
            <tr>
                <td>Support Level</td>
                <td>{summary['Support']}</td>
            </tr>
            <tr>
                <td>Resistance Level</td>
                <td>{summary['Resistance']}</td>
            </tr>
            <tr>
                <td>Highest Close Price</td>
                <td>{summary['Highest_Close']}</td>
            </tr>
            <tr>
                <td>Lowest Close Price</td>
                <td>{summary['Lowest_Close']}</td>
            </tr>
            <tr>
                <td>Average Close Price</td>
                <td>{summary['Average_Close']}</td>
            </tr>
            <tr>
                <td>Average Volume</td>
                <td>{summary['Average_Volume']}</td>
            </tr>
            <tr>
                <td>Highest Volume</td>
                <td>{summary['Highest_Volume']}</td>
            </tr>
            <tr>
                <td>Lowest Volume</td>
                <td>{summary['Lowest_Volume']}</td>
            </tr>
            <tr>
                <td>Volatility</td>
                <td>{summary['Volatility']}</td>
            </tr>
            <tr>
                <td>Recent Trend</td>
                <td>{summary['Recent_Trend']}</td>
            </tr>
            <tr>
                <td>Percentage Change</td>
                <td>{summary['Percentage_Change']}%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.write('AI Recommendation:')
    st.write(recommendation)
