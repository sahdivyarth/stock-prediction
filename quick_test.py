"""
Quick test to verify the fix works
"""

from fixed_robust_predictor import FixedRobustStockPredictor

def quick_test():
    print("🚀 QUICK TEST - FIXED PREDICTOR")
    print("=" * 40)
    
    predictor = FixedRobustStockPredictor()
    
    # Test a few stocks quickly
    test_stocks = ['AAPL', 'TSLA', 'GOOGL']
    
    for ticker in test_stocks:
        print(f"\n📊 Testing {ticker}...")
        
        result = predictor.predict_stock(ticker, 'linear_regression', 7)
        
        if result['success']:
            data = result['data']
            print(f"✅ {ticker}: ${data['current_price']:.2f} → ${data['predictions'][-1]:.2f}")
        else:
            print(f"❌ {ticker}: {result['message']}")
    
    print(f"\n🎉 If you see ✅ above, the fix worked!")

if __name__ == "__main__":
    quick_test()
