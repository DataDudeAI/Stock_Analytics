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

# Profile
def fetch_stock_profile(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        profile = {
            "Name": info.get('longName', 'N/A'),
            "Current Price": f"₹ {info.get('currentPrice', 'N/A')}",
            "Percentage Change": f"{info.get('dayChangePercent', 'N/A')}%",
            "Last Updated": datetime.now().strftime('%d %b %H:%M %p'),
            "Website": info.get('website', 'N/A'),
            "BSE Code": info.get('symbol', 'N/A'),
            "NSE Code": info.get('symbol', 'N/A'),
            "Market Cap": f"₹ {info.get('marketCap', 'N/A') / 1e7:.2f} Cr.",
            "High / Low": f"₹ {info.get('dayHigh', 'N/A')} / ₹ {info.get('dayLow', 'N/A')}"
        }
        return profile
    except Exception as e:
        st.error(f"Error fetching profile for {ticker}: {e}")
        return {}


import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch and display stock profile
def fetch_stock_profile(ticker):
    stock = yf.Ticker(ticker)
    profile = {}
    try:
        info = stock.info
        profile['Name'] = info.get('shortName', 'N/A')
        profile['Current Price'] = info.get('currentPrice', 'N/A')
        profile['Market Cap'] = info.get('marketCap', 'N/A')
        profile['P/E Ratio'] = info.get('forwardEps', 'N/A')
        profile['Book Value'] = info.get('bookValue', 'N/A')
        profile['Dividend Yield'] = info.get('dividendYield', 'N/A')
        profile['ROCE'] = info.get('returnOnCapitalEmployed', 'N/A')
        profile['ROE'] = info.get('returnOnEquity', 'N/A')
        profile['Face Value'] = info.get('faceValue', 'N/A')
    except Exception as e:
        st.write("Error fetching stock profile:", e)
    return profile

# Function to display stock profile as a table
def display_profile(profile):
    st.subheader("Stock Profile")
    profile_df = pd.DataFrame([profile])
    st.table(profile_df)

# Function to fetch and display quarterly results
def display_quarterly_results(ticker):
    st.subheader("Quarterly Results Summary")
    stock = yf.Ticker(ticker)
    
    try:
        financials = stock.quarterly_financials.T
        if not financials.empty:
            results = {
                'Sales': financials['Total Revenue'].iloc[-1] if 'Total Revenue' in financials.columns else 'N/A',
                'Operating Profit Margin': financials['Operating Income'].iloc[-1] if 'Operating Income' in financials.columns else 'N/A',
                'Net Profit': financials['Net Income'].iloc[-1] if 'Net Income' in financials.columns else 'N/A'
            }
            results_df = pd.DataFrame([results])
            st.table(results_df)
        else:
            st.write("No quarterly results available.")
    except Exception as e:
        st.write("Error fetching quarterly results:", e)

# Function to fetch and display shareholding pattern
def display_shareholding_pattern(ticker):
    st.subheader("Shareholding Pattern")
    
    # Placeholder values, replace with actual API or data source call
    data = {
        'Category': ['Promoters', 'FIIs (Foreign Institutional Investors)', 'DIIs (Domestic Institutional Investors)', 'Public'],
        'Percentage': ['63.17%', '9.10%', '15.03%', '12.71%']
    }
    df = pd.DataFrame(data)
    st.table(df)

# Function to fetch and display financial ratios
def display_financial_ratios(ticker):
    st.subheader("Financial Ratios")
    stock = yf.Ticker(ticker)
    
    try:
        # Placeholder values, calculate actual values based on your requirements
        ratios = {
            'Debtor Days': 73,
            'Working Capital Days': 194,
            'Cash Conversion Cycle': 51
        }
        ratios_df = pd.DataFrame([ratios])
        st.table(ratios_df)
    except Exception as e:
        st.write("Error fetching financial ratios:", e)
    
    # Add sections for Small Cap Top Gainers and Losers in similar manner if needed
