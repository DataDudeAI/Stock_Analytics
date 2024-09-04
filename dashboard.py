import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Fetch Nifty 50 tickers
def fetch_nifty50_tickers():
    return [
        "TATAMOTORS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "ITC.NS", "AXISBANK.NS", "MARUTI.NS", "TATASTEEL.NS",
        "WIPRO.NS", "SUNPHARMA.NS", "HINDALCO.NS", "HCLTECH.NS", "NTPC.NS",
        "L&T.NS", "M&M.NS", "ONGC.NS", "HDFCLIFE.NS", "ULTRACEMCO.NS",
        "ADANIGREEN.NS", "BHARTIARTL.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "DIVISLAB.NS",
        "POWERGRID.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "TCS.NS", "CIPLA.NS",
        "ASIANPAINT.NS", "GRASIM.NS", "BRITANNIA.NS", "SHREECEM.NS",
        "TECHM.NS", "INDUSINDBK.NS", "EICHERMOT.NS", "COALINDIA.NS", "GAIL.NS",
        "BOSCHLTD.NS", "M&MFIN.NS", "IDFCFIRSTB.NS", "HAVELLS.NS"
    ]

# Fetch large cap tickers
def fetch_large_cap_tickers():
    return fetch_nifty50_tickers()  # Assuming large caps are the same as Nifty 50

# Fetch small cap tickers
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

# Get top movers
def get_top_movers(tickers, days=1):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty and 'Close' in df.columns:
                df['Ticker'] = ticker
                data[ticker] = df['Close'].pct_change().iloc[-1]  # Percentage change
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    top_gainers = sorted_data[:10]
    top_losers = sorted_data[-10:]
    
    return top_gainers, top_losers

# Format DataFrame with color
def format_df(df):
    if not df.empty:
        df['Percentage Change'] = pd.to_numeric(df['Percentage Change'], errors='coerce')
        return df.style.applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Percentage Change'])
    return df

# Display dashboard
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
            st.dataframe(format_df(df_gainers_nifty50))

    with col2:
        st.write("### Nifty 50 Top Losers")
        if top_losers_nifty50:
            df_losers_nifty50 = pd.DataFrame(top_losers_nifty50, columns=['Ticker', 'Percentage Change'])
            st.dataframe(format_df(df_losers_nifty50))

    with col3:
        st.write("### Large Cap Top Gainers")
        if top_gainers_large_cap:
            df_gainers_large_cap = pd.DataFrame(top_gainers_large_cap, columns=['Ticker', 'Percentage Change'])
            st.dataframe(format_df(df_gainers_large_cap))

    with col4:
        st.write("### Large Cap Top Losers")
        if top_losers_large_cap:
            df_losers_large_cap = pd.DataFrame(top_losers_large_cap, columns=['Ticker', 'Percentage Change'])
            st.dataframe(format_df(df_losers_large_cap))

# Fetch and display stock profile
def fetch_stock_profile(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        profile = {
            "Name": info.get('shortName', 'N/A'),
            "Current Price": f"₹ {info.get('currentPrice', 'N/A')}",
            "Market Cap": f"₹ {info.get('marketCap', 'N/A') / 1e7:.2f} Cr.",
            "P/E Ratio": info.get('forwardEps', 'N/A'),
            "Book Value": info.get('bookValue', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "ROCE": info.get('returnOnCapitalEmployed', 'N/A'),
            "ROE": info.get('returnOnEquity', 'N/A'),
            "Face Value": info.get('faceValue', 'N/A')
        }
        return profile
    except Exception as e:
        st.error(f"Error fetching profile for {ticker}: {e}")
        return {}




# Display stock profile as a table
def display_profile(profile):
    st.subheader("Stock Profile")
    profile_df = pd.DataFrame([profile])
    st.table(profile_df)

# Fetch and display quarterly results
def display_quarterly_results(ticker):
    st.subheader("Quarterly Results Summary")
    try:
        stock = yf.Ticker(ticker)
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
        st.write(f"Error fetching quarterly results: {e}")

# Fetch and display shareholding pattern
def display_shareholding_pattern(ticker):
    st.subheader("Shareholding Pattern")
    
    # Placeholder values; replace with actual data source or API call
    data = {
        'Category': ['Promoters', 'FIIs (Foreign Institutional Investors)', 'DIIs (Domestic Institutional Investors)', 'Public'],
        'Holding (%)': [45.0, 20.0, 15.0, 20.0]
    }
    
    df = pd.DataFrame(data)
    st.table(df)


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

















# Main application
# def main():
#     st.title("Stock Analysis Dashboard")

#     # Select ticker input
#     ticker = st.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)")

#     if ticker:
#         profile = fetch_stock_profile(ticker)
#         if profile:
#             display_profile(profile)
        
#         display_quarterly_results(ticker)
#         display_shareholding_pattern(ticker)

#     # Show dashboard
#     if st.button("Show Dashboard"):
#         display_dashboard()

# if __name__ == "__main__":
#     main()
