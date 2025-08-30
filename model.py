
import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


from logger import get_logger

logger = get_logger(__name__)
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# # Example usage of logger
# logger.info("This is an info message")

# Fetch historical data
# def fetch_data(ticker, start_date, end_date):
#     logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
#     data = yf.download(ticker, start=start_date, end=end_date)
#     if data.empty:
#         logger.warning(f"No data returned for {ticker}.")
#         return None
    
#     # Reset index to ensure Date is a column
#     data.reset_index(inplace=True)
#     logger.info(f"Data fetched successfully for {ticker}.")
#     return data



def fetch_data(ticker, start_date, end_date):
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        logger.warning(f"No data returned for {ticker}.")
        return None

    # Reset index to ensure Date is a column
    data.reset_index(inplace=True)

    # 🔑 Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]

    # Ensure "Date" is named correctly
    if 'Date' not in data.columns:
        data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

    logger.info(f"Data fetched successfully for {ticker}.")
    return data














# def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Calculating indicators with fixed parameters.")
    
#     # Check if required columns are present
#     required_columns = ['Close', 'High', 'Low', 'Volume']
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         logger.error(f"Missing columns in data: {', '.join(missing_columns)}")
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     # Calculate fixed moving averages
#     ma_period = 50  # Fixed period for moving averages
#     try:
#         data[f'SMA_{ma_period}'] = data['Close'].rolling(window=ma_period).mean()
#         data[f'EMA_{ma_period}'] = data['Close'].ewm(span=ma_period, adjust=False).mean()
#     except Exception as e:
#         logger.error(f"Error calculating moving averages: {e}")
#         raise
    
#     # Calculate other indicators
#     try:
#         data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
#         macd = ta.trend.MACD(data['Close'])
#         data['MACD'] = macd.macd()
#         data['MACD_Signal'] = macd.macd_signal()
#         bollinger = ta.volatility.BollingerBands(data['Close'])
#         data['Bollinger_High'] = bollinger.bollinger_hband()
#         data['Bollinger_Low'] = bollinger.bollinger_lband()
#         data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
#         data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
#     except Exception as e:
#         logger.error(f"Error calculating other indicators: {e}")
#         raise
    
#     # Debugging line to check the columns
#     logger.debug("Columns after calculating indicators: %s", data.columns)
    
#     data = data.dropna()
#     logger.info("Indicators calculated successfully.")
#     return data

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating indicators with fixed parameters.")
    
    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing columns in data: {', '.join(missing_columns)}")
        raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
    # Ensure Series are 1D
    close = pd.Series(data['Close'].values.flatten(), index=data.index)
    high = pd.Series(data['High'].values.flatten(), index=data.index)
    low = pd.Series(data['Low'].values.flatten(), index=data.index)
    volume = pd.Series(data['Volume'].values.flatten(), index=data.index)

    ma_period = 50
    try:
        data[f'SMA_{ma_period}'] = close.rolling(window=ma_period).mean()
        data[f'EMA_{ma_period}'] = close.ewm(span=ma_period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating moving averages: {e}")
        raise
    
    try:
        data['RSI'] = ta.momentum.RSIIndicator(close).rsi()
        
        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        
        bollinger = ta.volatility.BollingerBands(close)
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        
        atr = ta.volatility.AverageTrueRange(high, low, close)
        data['ATR'] = atr.average_true_range()
        
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
        data['OBV'] = obv.on_balance_volume()
    except Exception as e:
        logger.error(f"Error calculating other indicators: {e}")
        raise
    
    data = data.dropna()
    logger.info("Indicators calculated successfully.")
    return data


    
# def calculate_indicators(data: pd.DataFrame, ma_type='SMA', ma_period=50) -> pd.DataFrame:
#     logger.info(f"Calculating indicators with {ma_type} of period {ma_period}.")
    
#     # Check if required columns are present
#     required_columns = ['Close', 'High', 'Low', 'Volume']
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         logger.error(f"Missing columns in data: {', '.join(missing_columns)}")
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     # Calculate moving averages
#     if ma_type == 'SMA':
#         data[f'SMA_{ma_period}'] = data['Close'].rolling(window=ma_period).mean()
#     elif ma_type == 'EMA':
#         data[f'EMA_{ma_period}'] = data['Close'].ewm(span=ma_period, adjust=False).mean()
#     else:
#         logger.error(f"Unknown moving average type: {ma_type}")
#         raise ValueError(f"Unknown moving average type: {ma_type}")
    
#     # Calculate other indicators
#     try:
#         data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
#         macd = ta.trend.MACD(data['Close'])
#         data['MACD'] = macd.macd()
#         data['MACD_Signal'] = macd.macd_signal()
#         bollinger = ta.volatility.BollingerBands(data['Close'])
#         data['Bollinger_High'] = bollinger.bollinger_hband()
#         data['Bollinger_Low'] = bollinger.bollinger_lband()
#         data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
#         data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
#     except Exception as e:
#         logger.error(f"Error calculating indicators: {e}")
#         raise
    
    # Debugging line to check the columns
    logger.debug("Columns after calculating indicators: %s", data.columns)
    
    data = data.dropna()
    logger.info("Indicators calculated successfully.")
    return data


# # Calculate technical indicators
# def calculate_indicators(data, ma_type='SMA', ma_period=50):
#     logger.info(f"Calculating indicators with {ma_type} of period {ma_period}.")
    
#     if ma_type == 'SMA':
#         data[f'SMA_{ma_period}'] = data['Close'].rolling(window=ma_period).mean()
#     elif ma_type == 'EMA':
#         data[f'EMA_{ma_period}'] = data['Close'].ewm(span=ma_period, adjust=False).mean()
    
#     data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
#     macd = ta.trend.MACD(data['Close'])
#     data['MACD'] = macd.macd()
#     data['MACD_Signal'] = macd.macd_signal()
#     bollinger = ta.volatility.BollingerBands(data['Close'])
#     data['Bollinger_High'] = bollinger.bollinger_hband()
#     data['Bollinger_Low'] = bollinger.bollinger_lband()
#     data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
#     data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    
#     # Debugging line to check the columns
#     logger.debug("Columns after calculating indicators: %s", data.columns)
    
#     data = data.dropna()
#     logger.info("Indicators calculated successfully.")
#     return data

# Calculate support and resistance levels
def calculate_support_resistance(data, window=30):
    logger.info(f"Calculating support and resistance with a window of {window}.")
    
    recent_data = data.tail(window)
    rolling_max = data['Close'].rolling(window=window).max()
    rolling_min = data['Close'].rolling(window=window).min()
    recent_max = recent_data['Close'].max()
    recent_min = recent_data['Close'].min()
    
    support = min(rolling_min.iloc[-1], recent_min)
    resistance = max(rolling_max.iloc[-1], recent_max)
    
    logger.debug("Support: %f, Resistance: %f", support, resistance)
    return support, resistance

# Prepare data for LSTM model
def prepare_lstm_data(data):
    logger.info("Preparing data for LSTM model.")
    
    features = data[['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']].values
    target = data['Close'].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(len(features) - 60):
        X.append(features[i:i+60])
        y.append(target[i+60])
        
    logger.info("Data preparation for LSTM completed.")
    return np.array(X), np.array(y)


# def predict_future_prices(data, algorithm, days=10):
#     logger.info(f"Predicting future prices using {algorithm}.")
    
#     # Check if required columns are present
#     required_columns = ['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']
#     missing_columns = [col for col in required_columns if col not in data.columns]
    
#     if missing_columns:
#         logger.error("Missing columns in data: %s", ', '.join(missing_columns))
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     features = data[required_columns]
#     target = data['Close']
    
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
#     mae, r2 = None, None  # Initialize variables for metrics
    
#     if algorithm == 'Linear Regression':
#         model = LinearRegression()
        
#     elif algorithm == 'Decision Tree':
#         model = DecisionTreeRegressor()
        
#     elif algorithm == 'Random Forest':
#         model = RandomForestRegressor(n_estimators=100)
        
#     elif algorithm == 'XGBoost':
#         model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        
#     elif algorithm == 'CatBoost':
#         model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0)
        
#     elif algorithm == 'LSTM':
#         X, y = prepare_lstm_data(data)
#         model = Sequential()
#         model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#         model.add(LSTM(50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)
#         last_data_point = np.expand_dims(X[-1], axis=0)
#         future_prices = [model.predict(last_data_point)[0][0] for _ in range(days)]
#         logger.info("Future prices predicted using LSTM model.")
#         return future_prices, None, None, None, None
        
#     elif algorithm == 'ARIMA':
#         model = ARIMA(data['Close'], order=(5, 1, 0))
#         model_fit = model.fit()
#         future_prices = model_fit.forecast(steps=days)
        
#     elif algorithm == 'SARIMA':
#         model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
#         model_fit = model.fit()
#         future_prices = model_fit.forecast(steps=days)
        
#     else:
#         logger.error("Algorithm not recognized: %s", algorithm)
#         return None, None, None, None, None

#     if algorithm in ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost']:
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         mae = mean_absolute_error(y_test, predictions)
#         r2 = r2_score(y_test, predictions)
        
#         future_prices = []
#         last_data_point = features.iloc[-1].values.reshape(1, -1)  # Ensure it's 2D
        
#         for _ in range(days):
#             future_price = model.predict(last_data_point)[0]
#             future_prices.append(future_price)
#             last_data_point = last_data_point + 1  # Update last data point (simplified, better methods should be used)
        
#     logger.info("Future prices predicted using %s model.", algorithm)
#     return future_prices, mae, r2, None, None

def predict_future_prices(data, algorithm, days=10):
    logger.info(f"Predicting future prices using {algorithm}.")

    required_columns = ['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 
                        'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 
                        'ATR', 'OBV']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error("Missing columns in data: %s", ', '.join(missing_columns))
        raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
    features = data[required_columns]
    target = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    mae, r2 = None, None

    # ---------------- Classical ML ----------------
    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif algorithm == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100)
    elif algorithm == 'XGBoost':
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    elif algorithm == 'CatBoost':
        model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0)
    # ---------------- Deep Learning ----------------
    elif algorithm == 'LSTM':
        X, y = prepare_lstm_data(data)
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        last_data_point = np.expand_dims(X[-1], axis=0)
        future_prices = []
        for _ in range(days):
            pred = model.predict(last_data_point, verbose=0)[0][0]
            future_prices.append(pred)
            # append pred to sequence (sliding window)
            last_data_point = np.roll(last_data_point, -1, axis=1)
            last_data_point[0, -1, 0] = pred
        return future_prices, None, None, None, None
    # ---------------- Time Series ----------------
    elif algorithm == 'ARIMA':
        model = ARIMA(target, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        mae = mean_absolute_error(target[-days:], forecast[:len(target[-days:])])
        r2 = r2_score(target[-days:], forecast[:len(target[-days:])])
        return forecast.tolist(), mae, r2, None, None
    elif algorithm == 'SARIMA':
        model = SARIMAX(target, order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        mae = mean_absolute_error(target[-days:], forecast[:len(target[-days:])])
        r2 = r2_score(target[-days:], forecast[:len(target[-days:])])
        return forecast.tolist(), mae, r2, None, None
    else:
        logger.error("Algorithm not recognized: %s", algorithm)
        return None, None, None, None, None

    # ---------------- Train classical models ----------------
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Generate simple future predictions using last row
    future_prices = []
    last_data_point = features.iloc[-1].values.reshape(1, -1)
    for _ in range(days):
        pred = model.predict(last_data_point)[0]
        future_prices.append(pred)
        # (⚠️ currently not updating indicators — just repeating last row with new Close)
        last_data_point[0, 0] = pred   # replace "Open" or "Close"-proxy with prediction

    logger.info("Future prices predicted using %s model.", algorithm)
    return future_prices, mae, r2, predictions, y_test


# def predict_future_prices(data, algorithm, days=10):
#     logger.info(f"Predicting future prices using {algorithm}.")
    
#     # Check if required columns are present
#     required_columns = ['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']
#     missing_columns = [col for col in required_columns if col not in data.columns]
    
#     if missing_columns:
#         logger.error("Missing columns in data: %s", ', '.join(missing_columns))
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     features = data[required_columns]
#     target = data['Close']
    
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
#     if algorithm == 'Linear Regression':
#         model = LinearRegression()
        
#     elif algorithm == 'Decision Tree':
#         model = DecisionTreeRegressor()
        
#     elif algorithm == 'Random Forest':
#         model = RandomForestRegressor(n_estimators=100)
        
#     elif algorithm == 'XGBoost':
#         model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        
#     elif algorithm == 'CatBoost':
#         model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0)
        
#     elif algorithm == 'LSTM':
#         X, y = prepare_lstm_data(data)
#         model = Sequential()
#         model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#         model.add(LSTM(50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)
#         last_data_point = np.expand_dims(X[-1], axis=0)
#         future_prices = [model.predict(last_data_point)[0][0] for _ in range(days)]
#         logger.info("Future prices predicted using LSTM model.")
#         return future_prices, None, None, None, None
        
#     elif algorithm == 'ARIMA':
#         model = ARIMA(data['Close'], order=(5, 1, 0))
#         model_fit = model.fit()
#         future_prices = model_fit.forecast(steps=days)
        
#     elif algorithm == 'SARIMA':
#         model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
#         model_fit = model.fit()
#         future_prices = model_fit.forecast(steps=days)
        
#     else:
#         logger.error("Algorithm not recognized: %s", algorithm)
#         return None, None, None, None, None

#     if algorithm in ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost']:
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         mae = mean_absolute_error(y_test, predictions)
#         r2 = r2_score(y_test, predictions)
        
#         future_prices = []
#         last_data_point = features.iloc[-1].values.reshape(1, -1)  # Ensure it's 2D
        
#         for _ in range(days):
#             future_price = model.predict(last_data_point)[0]
#             future_prices.append(future_price)
#             last_data_point = last_data_point + 1  # Update last data point (simplified, better methods should be used)
        
#     logger.info("Future prices predicted using %s model.", algorithm)
#     return future_prices, mae, r2, None, None












# # Predict future prices using the selected algorithm
# def predict_future_prices(data, algorithm, days=10):
#     logger.info(f"Predicting future prices using {algorithm}.")
    
#     # Check if required columns are present
#     required_columns = ['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']
#     missing_columns = [col for col in required_columns if col not in data.columns]
    
#     if missing_columns:
#         logger.error("Missing columns in data: %s", ', '.join(missing_columns))
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     features = data[required_columns]
#     target = data['Close']
    
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
#     if algorithm == 'Linear Regression':
#         model = LinearRegression()
#     elif algorithm == 'Decision Tree':
#         model = DecisionTreeRegressor()
#     elif algorithm == 'Random Forest':
#         model = RandomForestRegressor(n_estimators=100)
#     elif algorithm == 'XGBoost':
#         model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
#     elif algorithm == 'CatBoost':
#         model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0)
#     elif algorithm == 'LSTM':

#         X, y = prepare_lstm_data(data)
#         model = Sequential()
#         model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#         model.add(LSTM(50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)
#         last_data_point = np.expand_dims(X[-1], axis=0)
#         future_prices = [model.predict(last_data_point)[0][0] for _ in range(days)]
        
#     elif algorithm == 'ARIMA':
#             model = ARIMA(data['Close'], order=(5, 1, 0))
#             model_fit = model.fit()
#             future_prices = model_fit.forecast(steps=10)

#     elif algorithm == 'SARIMA':
#             model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
#             model_fit = model.fit()
#             forecast = model_fit.forecast(steps=10)

#         logger.info("Future prices predicted using LSTM model.")
#         return future_prices, None, None, None, None
#     else:
#         logger.error("Algorithm not recognized: %s", algorithm)
#         return None, None, None, None, None

#     model.fit(X_train, y_train)
    
#     predictions = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
    
#     future_prices = []
#     last_data_point = features.iloc[-1].values.reshape(1, -1)  # Ensure it's 2D
    
#     for _ in range(days):
#         future_price = model.predict(last_data_point)[0]
#         future_prices.append(future_price)
#         last_data_point = last_data_point + 1  # Update last data point (simplified, better methods should be used)
    
#     logger.info("Future prices predicted using %s model.", algorithm)
#     return future_prices, mae, r2, None, None

# import pandas as pd
# import numpy as np
# import yfinance as yf
# import ta
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import xgboost as xgb
# from catboost import CatBoostRegressor
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# from logger import get_logger

# logger = get_logger(__name__)

# # Fetch historical data
# def fetch_data(ticker, start_date, end_date):
#     logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
#     data = yf.download(ticker, start=start_date, end=end_date)
#     if data.empty:
#         logger.warning(f"No data returned for {ticker}.")
#         return None
    
#     # Reset index to ensure Date is a column
#     data.reset_index(inplace=True)
#     logger.info(f"Data fetched successfully for {ticker}.")
#     return data

# def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Calculating indicators with fixed parameters.")
    
#     # Check if required columns are present
#     required_columns = ['Close', 'High', 'Low', 'Volume']
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         logger.error(f"Missing columns in data: {', '.join(missing_columns)}")
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     # Calculate fixed moving averages
#     ma_period = 50  # Fixed period for moving averages
#     try:
#         data[f'SMA_{ma_period}'] = data['Close'].rolling(window=ma_period).mean()
#         data[f'EMA_{ma_period}'] = data['Close'].ewm(span=ma_period, adjust=False).mean()
#     except Exception as e:
#         logger.error(f"Error calculating moving averages: {e}")
#         raise
    
#     # Calculate other indicators
#     try:
#         data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
#         macd = ta.trend.MACD(data['Close'])
#         data['MACD'] = macd.macd()
#         data['MACD_Signal'] = macd.macd_signal()
#         bollinger = ta.volatility.BollingerBands(data['Close'])
#         data['Bollinger_High'] = bollinger.bollinger_hband()
#         data['Bollinger_Low'] = bollinger.bollinger_lband()
#         data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
#         data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
#     except Exception as e:
#         logger.error(f"Error calculating other indicators: {e}")
#         raise
    
#     # Debugging line to check the columns
#     logger.debug("Columns after calculating indicators: %s", data.columns)
    
#     data = data.dropna()
#     logger.info("Indicators calculated successfully.")
#     return data

# # Calculate support and resistance levels
# def calculate_support_resistance(data, window=30):
#     logger.info(f"Calculating support and resistance with a window of {window}.")
    
#     recent_data = data.tail(window)
#     rolling_max = data['Close'].rolling(window=window).max()
#     rolling_min = data['Close'].rolling(window=window).min()
#     recent_max = recent_data['Close'].max()
#     recent_min = recent_data['Close'].min()
    
#     support = min(rolling_min.iloc[-1], recent_min)
#     resistance = max(rolling_max.iloc[-1], recent_max)
    
#     logger.debug("Support: %f, Resistance: %f", support, resistance)
#     return support, resistance

# # Prepare data for LSTM model
# def prepare_lstm_data(data):
#     logger.info("Preparing data for LSTM model.")
    
#     features = data[['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']].values
#     target = data['Close'].values
#     scaler = MinMaxScaler()
#     features = scaler.fit_transform(features)
    
#     X, y = [], []
#     for i in range(len(features) - 60):
#         X.append(features[i:i+60])
#         y.append(target[i+60])
        
#     logger.info("Data preparation for LSTM completed.")
#     return np.array(X), np.array(y)

# # Predict future prices using the selected algorithm
# def predict_future_prices(data, algorithm, days=10):
#     logger.info(f"Predicting future prices using {algorithm}.")
    
#     # Check if required columns are present
#     required_columns = ['Open', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV']
#     missing_columns = [col for col in required_columns if col not in data.columns]
    
#     if missing_columns:
#         logger.error("Missing columns in data: %s", ', '.join(missing_columns))
#         raise KeyError(f"Missing columns in data: {', '.join(missing_columns)}")
    
#     features = data[required_columns]
#     target = data['Close']
    
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
#     if algorithm == 'Linear Regression':
#         model = LinearRegression()
#     elif algorithm == 'Decision Tree':
#         model = DecisionTreeRegressor()
#     elif algorithm == 'Random Forest':
#         model = RandomForestRegressor(n_estimators=100)
#     elif algorithm == 'XGBoost':
#         model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
#     elif algorithm == 'CatBoost':
#         model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0)
#     elif algorithm == 'LSTM':
#         X, y = prepare_lstm_data(data)
#         model = Sequential()
#         model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#         model.add(LSTM(50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)
#         last_data_point = np.expand_dims(X[-1], axis=0)
#         future_prices = [model.predict(last_data_point)[0][0] for _ in range(days)]
        
#         logger.info("Future prices predicted using LSTM model.")
#         return future_prices, None, None, None, None
#     elif algorithm == 'ARIMA':
#         model = ARIMA(data['Close'], order=(5, 1, 0))
#         model_fit = model.fit(disp=0)
#         forecast = model_fit.forecast(steps=days)[0]
        
#         mae = mean_absolute_error(target[-days:], forecast[:days])
#         r2 = r2_score(target[-days:], forecast[:days])
        
#         logger.info("Future prices predicted using ARIMA model.")
#         return forecast.tolist(), mae, r2, None, None
#     elif algorithm == 'SARIMA':
#         model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
#         model_fit = model.fit(disp=0)
#         forecast = model_fit.forecast(steps=days)
        
#         mae = mean_absolute_error(target[-days:], forecast[:days])
#         r2 = r2_score(target[-days:], forecast[:days])
        
#         logger.info("Future prices predicted using SARIMA model.")
#         return forecast.tolist(), mae, r2, None, None
#     else:
#         logger.error("Algorithm not recognized: %s", algorithm)
#         return None, None, None, None, None

#     model.fit(X_train, y_train)
    
#     predictions = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
    
#     future_prices = []
#     last_data_point = features.iloc[-1].values.reshape(1, -1)  # Ensure it's 2D
    
#     for _ in range(days):
#         future_price = model.predict(last_data_point)[0]
#         future_prices.append(future_price)
#         last_data_point = last_data_point + 1  # Update last data point (simplified, better methods should be used)
    
#     logger.info("Future prices predicted using %s model.", algorithm)
#     return future_prices, mae, r2, predictions, y_test




# # model.py

# import pandas as pd
# import numpy as np
# import yfinance as yf
# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error, r2_score

# def fetch_data(ticker, start_date, end_date):
#     try:
#         df = yf.download(ticker, start=start_date, end=end_date)
#         return df
#     except Exception as e:
#         print(f"An error occurred while fetching data: {e}")
#         return None

# def calculate_indicators(data):
#     # Example indicators - these should be tailored to your requirements
#     data['SMA_50'] = data['Close'].rolling(window=50).mean()
#     data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
#     data['RSI'] = calculate_rsi(data['Close'])
#     data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
#     data['Bollinger_High'], data['Bollinger_Low'] = calculate_bollinger_bands(data['Close'])
#     data['ATR'] = calculate_atr(data)
#     data['OBV'] = calculate_obv(data)
#     return data

# def calculate_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# def calculate_macd(series):
#     macd = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
#     macd_signal = macd.ewm(span=9, adjust=False).mean()
#     return macd, macd_signal

# def calculate_bollinger_bands(series, window=20):
#     rolling_mean = series.rolling(window=window).mean()
#     rolling_std = series.rolling(window=window).std()
#     high = rolling_mean + (rolling_std * 2)
#     low = rolling_mean - (rolling_std * 2)
#     return high, low

# def calculate_atr(data, window=14):
#     high_low = data['High'] - data['Low']
#     high_close = np.abs(data['High'] - data['Close'].shift())
#     low_close = np.abs(data['Low'] - data['Close'].shift())
#     tr = np.max(np.array([high_low, high_close, low_close]), axis=0)
#     atr = tr.rolling(window=window).mean()
#     return atr

# def calculate_obv(data):
#     obv = (data['Volume'] * np.sign(data['Close'].diff())).fillna(0).cumsum()
#     return obv

# def calculate_support_resistance(data):
#     # Example calculation - you may need to refine this based on your requirements
#     support = data['Close'].min()
#     resistance = data['Close'].max()
#     return support, resistance

# def predict_future_prices(data, model_type='ARIMA'):
#     try:
#         # Use ARIMA
#         if model_type == 'ARIMA':
#             model = ARIMA(data['Close'], order=(5, 1, 0))
#             model_fit = model.fit()
#             forecast = model_fit.forecast(steps=10)
#         # Use SARIMA
#         elif model_type == 'SARIMA':
#             model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
#             model_fit = model.fit()
#             forecast = model_fit.forecast(steps=10)
#         else:
#             raise ValueError("Unsupported model type. Use 'ARIMA' or 'SARIMA'.")
        
#         # Calculate MAE and R2 for evaluation
#         y_true = data['Close'][-10:]  # last 10 days as true values for comparison
#         mae = mean_absolute_error(y_true, forecast[:len(y_true)])
#         r2 = r2_score(y_true, forecast[:len(y_true)])
        
#         # Return results
#         return forecast, mae, r2
#     except Exception as e:
#         print(f"An error occurred while predicting future prices: {e}")
#         return None, None, None





