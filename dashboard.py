import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

def fetch_nifty50_tickers():
    return [
        "TATAMOTORS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "ITC.NS", "AXISBANK.NS", "MARUTI.NS", "TATASTEEL.NS",
        "WIPRO.NS", "SUNPHARMA.NS", "HINDALCO.NS", "HCLTECH.NS", "NTPC.NS",
        "L&T.NS", "M&M.NS", "ONGC.NS", "HDFCLIFE.NS", "ULTRACEMCO.NS",
        "ADANIGREEN.NS", "BHARTIARTL.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "DIVISLAB.NS",
        "POWERGRID.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "TCS.NS", "CIPLA.NS",
        "ASIANPAINT.NS", "GRASIM.NS", "HDFCBANK.NS", "BRITANNIA.NS", "SHREECEM.NS",
        "TECHM.NS", "INDUSINDBK.NS", "EICHERMOT.NS", "COALINDIA.NS", "GAIL.NS",
        "BOSCHLTD.NS", "M&MFIN.NS", "IDFCFIRSTB.NS", "HAVELLS.NS", "RELIANCE.NS"
    ]

def fetch_large_cap_tickers():
    return [
        "TATAMOTORS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "L&T.NS", "HDFC.NS", "ITC.NS", "AXISBANK.NS", "MARUTI.NS",
        "TATASTEEL.NS", "WIPRO.NS", "SUNPHARMA.NS", "HINDALCO.NS", "HCLTECH.NS",
        "NTPC.NS", "M&M.NS", "ONGC.NS", "HDFCLIFE.NS", "ULTRACEMCO.NS",
        "ADANIGREEN.NS", "BHARTIARTL.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "DIVISLAB.NS",
        "POWERGRID.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "TCS.NS", "CIPLA.NS",
        "ASIANPAINT.NS", "GRASIM.NS", "HDFCBANK.NS", "BRITANNIA.NS", "SHREECEM.NS",
        "TECHM.NS", "INDUSINDBK.NS", "EICHERMOT.NS", "COALINDIA.NS", "GAIL.NS",
        "BOSCHLTD.NS", "M&MFIN.NS", "IDFCFIRSTB.NS", "HAVELLS.NS", "RELIANCE.NS"
    ]

def fetch_small_cap_tickers():
    return [
        "ALOKINDS.NS", "ADANIENT.NS", "AARTIIND.NS", "AVANTIFEED.NS", "BLS.IN",
        "BHEL.NS", "BIRLACORP.NS", "CARBORUNIV.NS", "CENTRALBANK.NS", "EMAMILTD.NS",
        "FDC.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GSKCONS.NS", "HAVELLS.NS",
        "HEMIPAPER.NS", "HIL.NS", "JINDALSAW.NS", "JUBLFOOD.NS", "KOTAKMAH.NS",
        "MSTCLAS.NS", "NCC.NS", "PAGEIND.NS", "PIIND.NS", "SBI.CN",
        "SISL.NS", "SOMANYCERA.NS", "STAR.NS", "SUNDARAM.NS", "TATAINVEST.NS",
        "VSTIND.NS", "WABCOINDIA.NS", "WELCORP.NS", "ZEELEARN.NS", "ZOMATO.NS"
    ]

def get_top_movers(tickers, days=1):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                df['Ticker'] = ticker
                data[ticker] = df['Close'].pct_change().iloc[-1]  # Percentage change
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    top_gainers = sorted_data[:10]
    top_losers = sorted_data[-10:]
    
    return top_gainers, top_losers

def display_dashboard():
    st.header("Dashboard")
    
    # Fetch tickers
    nifty50_tickers = fetch_nifty50_tickers()
    large_cap_tickers = fetch_large_cap_tickers()
    small_cap_tickers = fetch_small_cap_tickers()

    # Get top gainers and losers
    top_gainers_nifty50, top_losers_nifty50 = get_top_movers(nifty50_tickers)
    top_gainers_large_cap, top_losers_large_cap = get_top_movers(large_cap_tickers)
    top_gainers_small_cap, top_losers_small_cap = get_top_movers(small_cap_tickers)
    
    # Create columns for tables
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("### Nifty 50 Top Gainers")
        if top_gainers_nifty50:
            df_gainers_nifty50 = pd.DataFrame(top_gainers_nifty50, columns=['Ticker', 'Percentage Change'])
            df_gainers_nifty50['Percentage Change'] = df_gainers_nifty50['Percentage Change'].astype(float)
            st.dataframe(df_gainers_nifty50.style.applymap(lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red'))

    with col2:
        st.write("### Nifty 50 Top Losers")
        if top_losers_nifty50:
            df_losers_nifty50 = pd.DataFrame(top_losers_nifty50, columns=['Ticker', 'Percentage Change'])
            df_losers_nifty50['Percentage Change'] = df_losers_nifty50['Percentage Change'].astype(float)
            st.dataframe(df_losers_nifty50.style.applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green'))

    with col3:
        st.write("### Large Cap Top Gainers")
        if top_gainers_large_cap:
            df_gainers_large_cap = pd.DataFrame(top_gainers_large_cap, columns=['Ticker', 'Percentage Change'])
            df_gainers_large_cap['Percentage Change'] = df_gainers_large_cap['Percentage Change'].astype(float)
            st.dataframe(df_gainers_large_cap.style.applymap(lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red'))

    with col4:
        st.write("### Large Cap Top Losers")
        if top_losers_large_cap:
            df_losers_large_cap = pd.DataFrame(top_losers_large_cap, columns=['Ticker', 'Percentage Change'])
            df_losers_large_cap['Percentage Change'] = df_losers_large_cap['Percentage Change'].astype(float)
            st.dataframe(df_losers_large_cap.style.applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green'))
    
    # Add sections for Small Cap Top Gainers and Losers in similar manner if needed
