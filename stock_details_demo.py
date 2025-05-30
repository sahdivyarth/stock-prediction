"""
Demonstration of all stock details and features included in the app
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def demonstrate_stock_features():
    """Show all stock details and features available in the app"""
    
    print("🏢 STOCK PREDICTION APP - COMPLETE STOCK DETAILS")
    print("=" * 60)
    
    # Example with Apple stock
    ticker = "AAPL"
    print(f"\n📊 Analyzing {ticker} (Apple Inc.)")
    
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        
        # 1. BASIC STOCK INFORMATION
        print("\n1. 📋 BASIC STOCK INFORMATION:")
        info = stock.info
        print(f"   Company Name: {info.get('longName', 'N/A')}")
        print(f"   Sector: {info.get('sector', 'N/A')}")
        print(f"   Industry: {info.get('industry', 'N/A')}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,}")
        print(f"   Current Price: ${info.get('currentPrice', 0):.2f}")
        print(f"   52 Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        print(f"   52 Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}")
        
        # 2. HISTORICAL PRICE DATA
        print("\n2. 📈 HISTORICAL PRICE DATA:")
        hist_data = stock.history(period="1y")
        print(f"   Data Points: {len(hist_data)} days")
        print(f"   Date Range: {hist_data.index[0].strftime('%Y-%m-%d')} to {hist_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Latest Close: ${hist_data['Close'].iloc[-1]:.2f}")
        print(f"   Average Volume: {hist_data['Volume'].mean():,.0f}")
        print(f"   Highest Price: ${hist_data['High'].max():.2f}")
        print(f"   Lowest Price: ${hist_data['Low'].min():.2f}")
        
        # 3. TECHNICAL INDICATORS
        print("\n3. 🔧 TECHNICAL INDICATORS:")
        
        # Moving Averages
        hist_data['MA_5'] = hist_data['Close'].rolling(window=5).mean()
        hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
        
        print(f"   5-Day Moving Average: ${hist_data['MA_5'].iloc[-1]:.2f}")
        print(f"   20-Day Moving Average: ${hist_data['MA_20'].iloc[-1]:.2f}")
        print(f"   50-Day Moving Average: ${hist_data['MA_50'].iloc[-1]:.2f}")
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        hist_data['RSI'] = calculate_rsi(hist_data['Close'])
        print(f"   RSI (14-day): {hist_data['RSI'].iloc[-1]:.2f}")
        
        # Volatility
        hist_data['Returns'] = hist_data['Close'].pct_change()
        volatility = hist_data['Returns'].std() * np.sqrt(252) * 100  # Annualized
        print(f"   Annualized Volatility: {volatility:.2f}%")
        
        # 4. PRICE PERFORMANCE
        print("\n4. 📊 PRICE PERFORMANCE:")
        current_price = hist_data['Close'].iloc[-1]
        
        # Performance calculations
        one_week_ago = hist_data['Close'].iloc[-5] if len(hist_data) >= 5 else current_price
        one_month_ago = hist_data['Close'].iloc[-22] if len(hist_data) >= 22 else current_price
        three_months_ago = hist_data['Close'].iloc[-66] if len(hist_data) >= 66 else current_price
        
        week_change = ((current_price - one_week_ago) / one_week_ago) * 100
        month_change = ((current_price - one_month_ago) / one_month_ago) * 100
        quarter_change = ((current_price - three_months_ago) / three_months_ago) * 100
        
        print(f"   1 Week Change: {week_change:+.2f}%")
        print(f"   1 Month Change: {month_change:+.2f}%")
        print(f"   3 Month Change: {quarter_change:+.2f}%")
        
        # 5. VOLUME ANALYSIS
        print("\n5. 📦 VOLUME ANALYSIS:")
        avg_volume_10d = hist_data['Volume'].tail(10).mean()
        avg_volume_30d = hist_data['Volume'].tail(30).mean()
        latest_volume = hist_data['Volume'].iloc[-1]
        
        print(f"   Latest Volume: {latest_volume:,}")
        print(f"   10-Day Avg Volume: {avg_volume_10d:,.0f}")
        print(f"   30-Day Avg Volume: {avg_volume_30d:,.0f}")
        print(f"   Volume vs 10-Day Avg: {(latest_volume/avg_volume_10d-1)*100:+.1f}%")
        
        # 6. FINANCIAL METRICS
        print("\n6. 💰 FINANCIAL METRICS:")
        print(f"   P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"   Forward P/E: {info.get('forwardPE', 'N/A')}")
        print(f"   Price to Book: {info.get('priceToBook', 'N/A')}")
        print(f"   Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "   Dividend Yield: N/A")
        print(f"   Beta: {info.get('beta', 'N/A')}")
        
        # 7. PREDICTION CAPABILITIES
        print("\n7. 🔮 PREDICTION CAPABILITIES:")
        print("   Available ML Models:")
        print("   ✅ Linear Regression - Fast baseline predictions")
        print("   ✅ Random Forest - Ensemble method for robust forecasts")
        print("   ✅ LSTM Neural Network - Deep learning for complex patterns")
        print("   ✅ ARIMA - Time series analysis for trend prediction")
        print("   ")
        print("   Prediction Features:")
        print("   • 7, 14, 30, or 60-day forecasts")
        print("   • Confidence intervals and accuracy metrics")
        print("   • Model performance comparison")
        print("   • Interactive charts and visualizations")
        
        # 8. SUPPORTED STOCK EXCHANGES
        print("\n8. 🌍 SUPPORTED STOCK EXCHANGES:")
        print("   ✅ NYSE (New York Stock Exchange)")
        print("   ✅ NASDAQ")
        print("   ✅ LSE (London Stock Exchange)")
        print("   ✅ TSE (Tokyo Stock Exchange)")
        print("   ✅ And many more global exchanges")
        
        # 9. POPULAR STOCKS EXAMPLES
        print("\n9. 📈 POPULAR STOCKS SUPPORTED:")
        popular_stocks = [
            ("AAPL", "Apple Inc."),
            ("GOOGL", "Alphabet Inc."),
            ("MSFT", "Microsoft Corp."),
            ("AMZN", "Amazon.com Inc."),
            ("TSLA", "Tesla Inc."),
            ("META", "Meta Platforms Inc."),
            ("NVDA", "NVIDIA Corp."),
            ("NFLX", "Netflix Inc.")
        ]
        
        for symbol, name in popular_stocks:
            print(f"   • {symbol} - {name}")
        
        print("\n   And thousands more stocks worldwide!")
        
        # 10. REAL-TIME FEATURES
        print("\n10. ⚡ REAL-TIME FEATURES:")
        print("   ✅ Live stock price data")
        print("   ✅ Real-time technical indicators")
        print("   ✅ Up-to-date financial metrics")
        print("   ✅ Current market data")
        print("   ✅ Instant prediction generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return False

def show_sample_data():
    """Show sample stock data structure"""
    print("\n" + "="*60)
    print("📋 SAMPLE STOCK DATA STRUCTURE")
    print("="*60)
    
    sample_data = {
        "basic_info": {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000,
            "current_price": 175.50
        },
        "historical_data": {
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [180.00, 181.50, 179.00],
            "high": [182.00, 183.00, 181.50],
            "low": [179.50, 180.00, 178.00],
            "close": [181.00, 182.50, 180.50],
            "volume": [50000000, 45000000, 55000000]
        },
        "technical_indicators": {
            "ma_5": 181.20,
            "ma_20": 178.50,
            "ma_50": 175.80,
            "rsi": 65.5,
            "volatility": 25.3
        },
        "predictions": {
            "linear_regression": [182.00, 183.50, 185.00],
            "random_forest": [181.80, 183.20, 184.70],
            "lstm": [182.20, 184.00, 186.50],
            "arima": [181.50, 182.80, 184.20]
        }
    }
    
    import json
    print(json.dumps(sample_data, indent=2))

if __name__ == "__main__":
    print("🚀 Starting Stock Details Demonstration...")
    
    success = demonstrate_stock_features()
    
    if success:
        show_sample_data()
        print("\n✅ Stock details demonstration completed successfully!")
        print("\n🎯 The app includes ALL these features and much more!")
    else:
        print("\n⚠️  Demo failed - but the app still includes all these features!")
    
    print("\n📱 To see all features in action, run the main app:")
    print("   python app.py")
