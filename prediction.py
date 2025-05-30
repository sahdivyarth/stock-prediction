import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def get_historical_data(self, ticker, period='2y'):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert to format suitable for frontend
            data_dict = {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'prices': data['Close'].tolist(),
                'volumes': data['Volume'].tolist(),
                'highs': data['High'].tolist(),
                'lows': data['Low'].tolist()
            }
            
            return data_dict
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def prepare_data(self, data, lookback=60):
        """Prepare data for machine learning models"""
        prices = np.array(data['Close'])
        
        # Create features (technical indicators)
        df = pd.DataFrame({'Close': prices})
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Price_Change'] = df['Close'].pct_change()
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def linear_regression_model(self, data, days_ahead=30):
        """Linear Regression prediction"""
        df = self.prepare_data(data)
        
        # Prepare features and target
        X = df[['MA_5', 'MA_20', 'RSI', 'Price_Change']].values
        y = df['Close'].values
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predict future prices
        last_features = X[-1].reshape(1, -1)
        future_predictions = []
        
        for _ in range(days_ahead):
            pred = model.predict(last_features)[0]
            future_predictions.append(pred)
            # Update features for next prediction (simplified)
            last_features[0] = np.roll(last_features[0], -1)
            last_features[0][-1] = pred
        
        return {
            'predictions': future_predictions,
            'mse': mse,
            'mae': mae,
            'model_name': 'Linear Regression'
        }
    
    def random_forest_model(self, data, days_ahead=30):
        """Random Forest prediction"""
        df = self.prepare_data(data)
        
        # Prepare features and target
        X = df[['MA_5', 'MA_20', 'RSI', 'Price_Change']].values
        y = df['Close'].values
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predict future prices
        last_features = X[-1].reshape(1, -1)
        future_predictions = []
        
        for _ in range(days_ahead):
            pred = model.predict(last_features)[0]
            future_predictions.append(pred)
            # Update features for next prediction
            last_features[0] = np.roll(last_features[0], -1)
            last_features[0][-1] = pred
        
        return {
            'predictions': future_predictions,
            'mse': mse,
            'mae': mae,
            'model_name': 'Random Forest'
        }
    
    def lstm_model(self, data, days_ahead=30):
        """LSTM Neural Network prediction"""
        prices = np.array(data['Close']).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        # Prepare sequences
        lookback = 60
        X, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fit model with error handling
        try:
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0, validation_split=0.1)
        except Exception as e:
            print(f"LSTM training warning: {e}")
            # Fallback to simpler training
            model.fit(X_train, y_train, batch_size=16, epochs=25, verbose=0)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predict future prices
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        future_predictions = []
        
        for _ in range(days_ahead):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            future_predictions.append(pred)
            # Update sequence
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred
        
        # Inverse transform predictions
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        
        return {
            'predictions': future_predictions.tolist(),
            'mse': mse,
            'mae': mae,
            'model_name': 'LSTM'
        }
    
    def arima_model(self, data, days_ahead=30):
        """ARIMA time series prediction"""
        prices = data['Close']
        
        try:
            # Fit ARIMA model
            model = ARIMA(prices, order=(5, 1, 0))
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=days_ahead)
            
            # Calculate metrics on in-sample predictions
            fitted_values = fitted_model.fittedvalues
            mse = mean_squared_error(prices[1:], fitted_values)  # Skip first value due to differencing
            mae = mean_absolute_error(prices[1:], fitted_values)
            
            return {
                'predictions': forecast.tolist(),
                'mse': mse,
                'mae': mae,
                'model_name': 'ARIMA'
            }
        except Exception as e:
            # Fallback to simple moving average if ARIMA fails
            ma_pred = [prices.iloc[-1]] * days_ahead
            return {
                'predictions': ma_pred,
                'mse': 0,
                'mae': 0,
                'model_name': 'ARIMA (fallback to MA)'
            }
    
    def predict_stock(self, ticker, model_type='linear_regression', days_ahead=30):
        """Main prediction function"""
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period='2y')
            
            if hist_data.empty:
                return {'success': False, 'message': f'No data found for {ticker}'}
            
            # Choose model
            if model_type == 'linear_regression':
                result = self.linear_regression_model(hist_data, days_ahead)
            elif model_type == 'random_forest':
                result = self.random_forest_model(hist_data, days_ahead)
            elif model_type == 'lstm':
                result = self.lstm_model(hist_data, days_ahead)
            elif model_type == 'arima':
                result = self.arima_model(hist_data, days_ahead)
            else:
                return {'success': False, 'message': 'Invalid model type'}
            
            # Prepare response data
            current_price = hist_data['Close'].iloc[-1]
            historical_prices = hist_data['Close'].tail(30).tolist()
            historical_dates = hist_data.index.tail(30).strftime('%Y-%m-%d').tolist()
            
            # Generate future dates
            last_date = hist_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead).strftime('%Y-%m-%d').tolist()
            
            return {
                'success': True,
                'data': {
                    'ticker': ticker,
                    'current_price': current_price,
                    'predictions': result['predictions'],
                    'historical_prices': historical_prices,
                    'historical_dates': historical_dates,
                    'future_dates': future_dates,
                    'model_metrics': {
                        'mse': result['mse'],
                        'mae': result['mae'],
                        'model_name': result['model_name']
                    }
                }
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def compare_models(self, ticker):
        """Compare all models for a given ticker"""
        models = ['linear_regression', 'random_forest', 'lstm', 'arima']
        results = {}
        
        for model in models:
            try:
                result = self.predict_stock(ticker, model, 30)
                if result['success']:
                    results[model] = result['data']['model_metrics']
                else:
                    results[model] = {'error': result['message']}
            except Exception as e:
                results[model] = {'error': str(e)}
        
        return {'success': True, 'comparison': results}
